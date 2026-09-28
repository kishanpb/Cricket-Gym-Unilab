"""Experimental learned arm/release task with an explicitly frozen locomotion prior."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from unilab.base import registry
from unilab.utils.rotation import np_quat_apply_batched

from .holder import build_holder_scene
from .prior import SDK_JOINTS
from .residual import FrozenPriorResidual, FrozenPriorResidualCfg
from .scene import CONTACT_FIELDS, CONTACT_WIDTH
from .task import G1CricketCfg, make_g1_cricket_env

ARM_LIMITS = np.array([2.5, 0.8, 1.2, 1.0, 0.4, 0.4, 0.4], dtype=np.float32)
HOLDER_SENSORS = ("holder_force", "holder_quat", "ball_hand")
RELEASE_THRESHOLD = 0.5


@dataclass
class G1CricketBowlingCfg(G1CricketCfg):
    def validate(self):
        super().validate()
        if self.handedness not in {"left", "right"}:
            raise ValueError("bowling requires a left or right hand")
        if not self.mujoco_observe_substeps or self.sim_dt > 0.00025:
            raise ValueError("bowling requires substep observation and physics dt <= .25 ms")

    def build_scene(self, source: Path, destination: Path) -> tuple[str, ...]:
        guards = build_holder_scene(source, destination, self.handedness)
        tree = ET.parse(destination)
        root = tree.getroot()
        wrist = root.find(f".//body[@name='{self.handedness}_wrist_yaw_link']")
        weld = root.find("equality/weld[@name='ball_holder']")
        sensors = root.find("sensor")
        key = root.find("keyframe/key")
        assert wrist is not None and weld is not None and sensors is not None and key is not None
        fixture = ET.SubElement(wrist, "body", name="holder_loadcell")
        ET.SubElement(fixture, "site", name="holder_site", size=".005", rgba=".2 .7 .9 1")
        weld.set("body1", "holder_loadcell")
        ET.SubElement(sensors, "force", name="holder_force", site="holder_site")
        ET.SubElement(
            sensors, "framequat", name="holder_quat", objtype="site", objname="holder_site"
        )
        ET.SubElement(
            sensors,
            "contact",
            name="ball_hand",
            geom1="ball_geom",
            geom2=f"{self.handedness}_hand_collision",
            num="4",
            data=CONTACT_FIELDS,
        )
        # Move the stance and its held ball together; the target wickets are downfield.
        qpos = np.fromstring(key.attrib["qpos"], sep=" ")
        shift = np.array([-1.0, -0.3, 0])
        qpos[:3] += shift
        qpos[-7:-4] += shift
        key.set("qpos", " ".join(map(str, qpos)))
        for index in range(3):
            wicket = root.find(f"worldbody/geom[@name='wicket_{index}']")
            assert wicket is not None
            pos = np.fromstring(wicket.attrib["pos"], sep=" ")
            pos[0] = 18.6
            wicket.set("pos", " ".join(map(str, pos)))
        tree.write(destination)
        return guards


@dataclass(kw_only=True)
class BowlingActionCfg(FrozenPriorResidualCfg):
    def build(self, env):
        return BowlingAction(self, env)


class BowlingAction(FrozenPriorResidual):
    sensor_names = HOLDER_SENSORS

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._bowling_raw = np.zeros((env.num_envs, 8), dtype=np.float32)
        self.released = np.zeros(env.num_envs, dtype=bool)
        self.just_released = self.released.copy()
        self.release_position = np.zeros((env.num_envs, 3))
        self.release_velocity = np.zeros((env.num_envs, 3))
        self.release_elbow = np.zeros(env.num_envs)
        self.peak_load = np.zeros(env.num_envs)
        self.impulse_world = np.zeros((env.num_envs, 3))
        self.touch_fraction = np.zeros(env.num_envs)
        self.constraints = env.equality_constraints
        self.ball = env.scene["ball"]
        model = env.get_playback_model()
        ball_joint = model.body("cricket_ball").jntadr[0]
        self.qadr = 1 + model.jnt_qposadr[ball_joint]
        self.vadr = 1 + model.nq + model.jnt_dofadr[ball_joint]
        self.elbow_qadr = 1 + model.joint(f"{env.cfg.handedness}_elbow_joint").qposadr[0]
        self.release_physics = np.zeros(
            (env.num_envs, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        )
        self.joint_limits = self._entity.data.soft_joint_pos_limits
        env.set_substep_observer(self.sensor_names, "cricket_ball", self.observe)

    @property
    def action_dim(self):
        return 8

    @property
    def raw_action(self):
        return self._bowling_raw

    def process_actions(self, actions):
        self._bowling_raw[:] = actions
        super().process_actions(actions[:, :7])
        offsets = np.tanh(actions[:, :7]) * ARM_LIMITS
        self.processed_action[:, self.arm_ids] += offsets - self.residual_radians
        self.residual_radians[:] = offsets
        self.processed_action[:, self.arm_ids] = np.clip(
            self.processed_action[:, self.arm_ids],
            self.joint_limits[self.arm_ids, 0],
            self.joint_limits[self.arm_ids, 1],
        )
        self.just_released[:] = ~self.released & (actions[:, 7] > RELEASE_THRESHOLD)
        ids = np.flatnonzero(self.just_released)
        if len(ids):
            # Solved frame sensors lag this boundary; retain the integrated state.
            state = self._env.get_physics_state_snapshot()[ids]
            self.release_physics[ids] = state
            self.release_position[ids] = state[:, self.qadr : self.qadr + 3]
            self.release_velocity[ids] = state[:, self.vadr : self.vadr + 3]
            self.release_elbow[ids] = state[:, self.elbow_qadr]
            self.constraints.set_equality_active(ids, np.zeros((len(ids), 1), dtype=bool))
            self.released[ids] = True

    def observe(self, sensors, integrated_velocity):
        del integrated_velocity
        if not np.isfinite(sensors).all():
            raise RuntimeError("non-finite bowling load sensor")
        force, orientation = sensors[..., :3], sensors[..., 3:7]
        self.peak_load[:] = np.linalg.norm(force, axis=-1).max(axis=1)
        world = np_quat_apply_batched(orientation.reshape(-1, 4), force.reshape(-1, 3))
        self.impulse_world[:] = world.reshape(force.shape).sum(axis=1) * self._env.physics_dt
        contacts = sensors[..., 7:].reshape(*sensors.shape[:2], 4, CONTACT_WIDTH)
        if np.any(contacts[..., 0] > 4):
            raise RuntimeError("bowling hand contact capacity exceeded")
        self.touch_fraction[:] = (contacts[..., 0] > 0).any(axis=-1).mean(axis=1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        if self.constraints.get_equality_names() != ("ball_holder",):
            raise ValueError("bowling requires exactly the declared ball holder")
        for values in (
            self._bowling_raw,
            self.released,
            self.just_released,
            self.release_position,
            self.release_velocity,
            self.release_elbow,
            self.release_physics,
            self.peak_load,
            self.impulse_world,
            self.touch_fraction,
        ):
            values[ids] = 0


class ResetBowler:
    def __init__(self, cfg, env):
        self.robot, self.ball = env.scene["robot"], env.scene["ball"]
        self.model = env.get_playback_model()
        self.data = mujoco.MjData(self.model)
        self.joint_qadr = [self.model.joint(name).qposadr[0] for name in SDK_JOINTS]
        self.wrist = self.model.body(f"{env.cfg.handedness}_wrist_yaw_link").id
        self.offset = np.array([0.15, 0.06 if env.cfg.handedness == "left" else -0.06, 0])

    def __call__(self, env, env_ids):
        pose = self.robot.data.default_joint_pos[env_ids] + env.rng.uniform(
            -0.005, 0.005, (len(env_ids), 29)
        )
        robot_state = self.robot.data.default_root_state[env_ids].copy()
        robot_state[:, 7:] = 0
        ball_state = self.ball.data.default_root_state[env_ids].copy()
        ball_state[:, 7:] = 0
        for row in range(len(env_ids)):
            self.data.qpos[:] = self.model.key_qpos[0]
            self.data.qpos[:7] = robot_state[row, :7]
            self.data.qpos[self.joint_qadr] = pose[row]
            mujoco.mj_kinematics(self.model, self.data)
            ball_state[row, :3] = (
                self.data.xpos[self.wrist] + self.data.xmat[self.wrist].reshape(3, 3) @ self.offset
            )
            ball_state[row, 3:7] = self.data.xquat[self.wrist]
        self.robot.write_root_state_to_sim(robot_state, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(pose, np.zeros_like(pose), env_ids=env_ids)
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)


def runup_command(env):
    command = np.zeros((env.num_envs, 3))
    command[:, 0] = 0.4
    return command


class BowlingObservation:
    def __init__(self, cfg, env):
        self.robot, self.ball = env.scene["robot"], env.scene["ball"]
        self.loads = env.scene.bind_sensor_data(("holder_force",))
        self.wrist_index = 1 if env.cfg.handedness == "left" else 2
        self.hand = 1 if env.cfg.handedness == "right" else -1

    def __call__(self, env):
        term = env.action_manager.get_term("residual")
        wrist = self.robot.data.body_link_pos_w[:, self.wrist_index]
        return np.concatenate(
            (
                self.robot.data.projected_gravity_b,
                self.robot.data.root_link_pos_w,
                self.robot.data.root_link_lin_vel_w,
                self.robot.data.root_link_ang_vel_w,
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel * 0.05,
                self.ball.data.root_link_pos_w - wrist,
                self.ball.data.root_link_lin_vel_w,
                wrist[:, 2:3],
                env.action_manager.action,
                term.baseline_action,
                term.released[:, None],
                env.episode_length_buf[:, None] * env.step_dt,
                term.peak_load[:, None] / 100,
                self.loads.read() / 100,
                term.touch_fraction[:, None],
                np.full((env.num_envs, 1), self.hand),
            ),
            axis=1,
        )


def bowling_reward(env):
    return _bowling_reward(env, ball_position_gate=True)


def bowling_release_reward_v2(env):
    """Release shaping without a ball-position proxy for foot legality."""
    return _bowling_reward(env, ball_position_gate=False)


def _bowling_reward(env, *, ball_position_gate):
    term = env.action_manager.get_term("residual")
    robot = env.scene["robot"]
    wrist = robot.data.body_link_pos_w[:, 1 if env.cfg.handedness == "left" else 2]
    height = np.clip((term.release_position[:, 2] - 0.9) / 0.4, 0, 1)
    speed = np.clip(term.release_velocity[:, 0], 0, 12) / 6
    lateral = np.exp(-np.square(term.release_velocity[:, 1] / 2))
    before_crease = term.release_position[:, 0] < 0 if ball_position_gate else 1.0
    bonus = term.just_released * 10 * height * speed * lateral * before_crease / env.step_dt
    approach = np.exp(-np.square((wrist[:, 2] - 1.3) / 0.25)) * ~term.released
    tracking = 0.5 * np.exp(-np.square((robot.data.root_link_lin_vel_w[:, 0] - 0.4) / 0.3))
    return approach + tracking + bonus


registry.register_env_config("G1CricketBowling", G1CricketBowlingCfg)
registry.register_env("G1CricketBowling", make_g1_cricket_env, sim_backend="mujoco")
