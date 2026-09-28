"""Residual PPO batting with torque-controlled fingers and an unconstrained bat."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unilab.utils.rotation import np_quat_apply_inverse_batched

from .articulated_hands import hand_joints
from .articulated_scene import build_articulated_scene
from .articulated_swing import ArticulatedSwingController
from .boundary_batting import BoundaryBattingReward, ResetVariedDelivery
from .impact import add_blade_pair
from .prior import SDK_JOINTS
from .scene import BALL_CONTACT_NAMES, CONTACT_FIELDS, CONTACT_SLOTS, CONTACT_WIDTH
from .task import G1CricketCfg
from .tracking import ankle_balance, root_position_balance

FINGER_JOINTS = tuple(name for side in ("left", "right") for name in hand_joints(side))
ARTICULATED_DELIVERY_POOL = (
    (3.0, 0.0, 1.3, 6.5),
    (2.6, 0.0, 1.3, 6.5),
    (3.4, 0.0, 1.3, 6.5),
    (3.0, -0.08, 1.3, 6.5),
    (3.0, 0.08, 1.3, 6.5),
    (3.0, 0.0, 1.2, 6.2),
    (3.0, 0.0, 1.4, 6.8),
)
BALL_ROBOT_CONTACT_SLOTS = 64
BALL_ROBOT_COLUMNS = (
    "active_contacts",
    "loaded_contacts",
    "normal_force_n",
    "tangent_force_magnitudes_n",
    "penetration_m",
)


def add_articulated_ball_contacts(root, pitch_impedance="0.9 0.95 0.001 0.5 2"):
    add_blade_pair(root, "0.002")
    ET.SubElement(
        root.find("contact"),
        "pair",
        name="articulated_pitch_bounce_v1",
        geom1="ball_geom",
        geom2="pitch",
        condim="3",
        solref="0.002 0.3",
        solimp=pitch_impedance,
        friction="0.5 0.5 0.01 0.001 0.001",
    )


class ResetArticulatedDelivery(ResetVariedDelivery):
    pool = ARTICULATED_DELIVERY_POOL
    start_x = 6.0

    def __call__(self, env, env_ids, evaluation_pool=False, delivery_index=None):
        if evaluation_pool:
            for index in range(len(self.pool)):
                ids = env_ids[env_ids % len(self.pool) == index]
                if len(ids):
                    super().__call__(env, ids, delivery_index=index)
        else:
            super().__call__(
                env, env_ids, delivery_index=delivery_index, vertical_speed_range=(6.2, 6.8)
            )


class InterceptionShaping:
    """Dense training signal only; boundary scoring still requires a real valid hit."""

    def __init__(self, cfg, env):
        self.ball = env.scene["ball"]
        self.blade = env.scene.bind_sensor_data(("bat_center_world",))

    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.blade.read(), axis=1)
        return np.exp(-(distance**2) / 0.15) * (env.action_manager.get_term("batting").elapsed < 3)


class ArticulatedBoundaryReward(BoundaryBattingReward):
    def sensor_names(self, env):
        robot = ("ball_robot",) if env.cfg.check_ball_robot_contact else ()
        return (*super().sensor_names(env), *env.cfg.bat_guard_sensor_names, *robot)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.forbidden_support = np.zeros(env.num_envs, dtype=bool)
        self.guard_capacity = np.repeat(env.cfg.guard_sensor_slots, env.cfg.guard_sensor_slots)
        self.guard_width = len(self.guard_capacity) * CONTACT_WIDTH
        self.check_ball_robot_contact = env.cfg.check_ball_robot_contact
        self.ball_robot_contact_seen = np.zeros(env.num_envs, dtype=bool)
        self.reset_arrays += (self.forbidden_support, self.ball_robot_contact_seen)

    def observe(self, sensors, integrated_velocity):
        width = len(BALL_CONTACT_NAMES) * CONTACT_SLOTS * CONTACT_WIDTH + 3
        end = width + self.guard_width
        guards = sensors[..., width:end].reshape(*sensors.shape[:2], -1, CONTACT_WIDTH)
        if np.any(guards[..., 0] > self.guard_capacity):
            raise RuntimeError("articulated support sensor capacity exceeded")
        forbidden = (guards[..., 0] > 0).any(axis=(1, 2))
        self.forbidden_support |= forbidden
        self.disqualified |= forbidden
        invalid = None
        if self.check_ball_robot_contact:
            contacts = sensors[..., end:].reshape(*sensors.shape[:2], BALL_ROBOT_CONTACT_SLOTS, 5)
            if not np.isfinite(contacts).all():
                raise RuntimeError("nonfinite ball/robot contact sensors")
            if np.any(contacts[..., 0] > BALL_ROBOT_CONTACT_SLOTS):
                raise RuntimeError("ball/robot contact sensor capacity exceeded")
            active = contacts[..., 0] > 0
            loaded = active & (contacts[..., 1] > 0)
            depth = np.maximum(0, np.where(active, -contacts[..., 4], 0).max(axis=-1))
            self.ball_robot_metrics = np.stack(
                (
                    active.sum(axis=-1),
                    loaded.sum(axis=-1),
                    np.where(active, contacts[..., 1], 0).sum(axis=-1),
                    np.where(active, np.linalg.norm(contacts[..., 2:4], axis=-1), 0).sum(axis=-1),
                    depth,
                ),
                axis=-1,
            )
            invalid = loaded.any(axis=-1) | (depth > 0.006)
            self.ball_robot_contact_seen |= invalid.any(axis=1)
        super().observe(sensors[..., :width], integrated_velocity, invalid_contacts=invalid)


def invalid_support(env):
    return env.reward_manager.get_term_cfg("boundary").func.forbidden_support.copy()


@dataclass
class G1ArticulatedBattingCfg(G1CricketCfg):
    hand_assets: str = "src/unilab/assets/robots/g1_hands"
    reference_file: str = "g1_cricket_results/articulated_swing_compact/right_reference.npz"
    grasp_sensor_names: tuple[str, ...] = ()
    guard_sensor_slots: tuple[int, ...] = ()
    pitch_impedance: str = "0.9 0.95 0.001 0.5 2"
    physics_integrator: str = "implicitfast"
    check_ball_robot_contact: bool = False

    def build_scene(self, source, destination):
        build_articulated_scene(source, self.hand_assets, destination)
        tree = ET.parse(destination)
        root = tree.getroot()
        add_articulated_ball_contacts(root, self.pitch_impedance)
        root.find("option").set("integrator", self.physics_integrator)
        with np.load(self.reference_file) as reference:
            pose = reference["qpos"][0]
        model = mujoco.MjModel.from_xml_path(str(destination))
        if pose.shape != (model.nq,) or model.nu != 43:
            raise ValueError("articulated reference must match the 43-actuator scene")
        key = root.find("keyframe/key")
        key.set("name", "articulated_guard")
        key.set("qpos", " ".join(map(str, pose)))
        sensors = root.find("sensor")
        if self.check_ball_robot_contact:
            ET.SubElement(
                sensors,
                "contact",
                name="ball_robot",
                body1="cricket_ball",
                subtree2="pelvis",
                data="found force dist",
                num=str(BALL_ROBOT_CONTACT_SLOTS),
                reduce="none",
            )
        # A sphere/box pair has one contact; preserve the boundary scorer's four-slot layout.
        sensors.find("contact[@name='ball_bat']").set("num", "4")
        ET.SubElement(
            sensors, "framepos", name="ball_world_position", objtype="body", objname="cricket_ball"
        )
        ET.SubElement(
            sensors, "framepos", name="bat_center_world", objtype="geom", objname="bat_blade"
        )
        for side in ("left", "right"):
            ET.SubElement(
                sensors,
                "framepos",
                name=f"grip_{side}_wrist_world",
                objtype="xbody",
                objname=f"{side}_wrist_yaw_link",
            )
        self.grasp_sensor_names = tuple(
            sensor.get("name") for sensor in sensors if sensor.get("name", "").startswith("grasp_")
        )
        guards = [
            s.get("name")
            for s in sensors
            if s.tag == "contact" and s.get("name", "").startswith("prior_")
        ]
        for geom in root.findall("worldbody//geom"):
            name = geom.get("name", "")
            if not name or name.startswith("bat_") or name == "ball_geom":
                continue
            if geom.get("contype") == "0" and geom.get("conaffinity", "0") == "0":
                continue
            for bat in ("bat_blade", "bat_handle"):
                if bat == "bat_handle" and "_hand_" in name:
                    continue
                sensor_name = f"guard_{bat}_{name}"
                ET.SubElement(
                    sensors,
                    "contact",
                    name=sensor_name,
                    geom1=bat,
                    geom2=name,
                    data=CONTACT_FIELDS,
                    num="4",
                    reduce="none",
                )
                guards.append(sensor_name)
        self.guard_sensor_slots = tuple(
            int(sensors.find(f"contact[@name='{name}']").get("num")) for name in guards
        )
        tree.write(destination)
        return tuple(guards)


@dataclass(kw_only=True)
class ArticulatedBattingActionCfg(ActionTermCfg):
    scale: float = 0.1
    arm_scale: float = 0.25

    def build(self, env):
        return ArticulatedBattingAction(self, env)


class ArticulatedBattingAction(ActionTerm):
    requires_substep_state_feedback = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        with np.load(env.cfg.reference_file) as reference:
            self.controller = ArticulatedSwingController(
                model, reference["qpos"], reference["times"], inertial=True
            )
        self.body_ids, _ = self._entity.find_joints(SDK_JOINTS, preserve_order=True)
        self.finger_ids, _ = self._entity.find_joints(FINGER_JOINTS, preserve_order=True)
        self.actions = np.zeros((env.num_envs, 29), dtype=np.float32)
        self.target = np.zeros_like(self.actions)
        self.reference = np.zeros_like(self.actions)
        self.elapsed = np.zeros(env.num_envs)
        self.interval_start = np.zeros(env.num_envs)
        self.substep = 0

    @property
    def action_dim(self):
        return 29

    @property
    def raw_action(self):
        return self.actions

    def process_actions(self, actions):
        self.actions[:] = actions
        self.interval_start[:] = self.elapsed
        self.elapsed += self._env.step_dt
        self.substep = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.actions[ids] = 0
        self.elapsed[ids] = 0
        self.interval_start[ids] = 0
        self.reference[ids] = self.controller.poses[0, self.controller.q]

    def apply_actions(self):
        env, controller = self._env, self.controller
        elapsed = self.interval_start + self.substep * env.cfg.sim_dt
        self.substep += 1
        indices = np.clip(
            np.searchsorted(controller.times, elapsed, side="right") - 1,
            0,
            len(controller.times) - 2,
        )
        duration = controller.times[indices + 1] - controller.times[indices]
        blend = np.clip((elapsed - controller.times[indices]) / duration, 0, 1)[:, None]
        before, after = controller.poses[indices], controller.poses[indices + 1]
        self.reference[:] = (1 - blend) * before[:, controller.q] + blend * after[:, controller.q]
        velocity = (after[:, controller.q] - before[:, controller.q]) / duration[:, None]
        velocity[elapsed >= controller.times[-1]] = 0
        root_position = (1 - blend) * before[:, :3] + blend * after[:, :3]
        root_velocity = (after[:, :3] - before[:, :3]) / duration[:, None]
        root_velocity[elapsed >= controller.times[-1]] = 0
        force = (1 - blend) * controller.forces[indices] + blend * controller.forces[indices + 1]
        self.target[:] = (
            self.reference + controller.velocity_gain * velocity + force / controller.gain
        )
        data = self._entity.data
        for row in range(env.num_envs):
            correction = ankle_balance(
                controller.poses[0, 3:7],
                data.root_link_quat_w[row],
                data.root_link_ang_vel_b[row],
                4.0,
            )
            correction += root_position_balance(
                controller.poses[0, 3:7],
                data.root_link_pos_w[row] - env.scene.env_origins[row] - root_position[row],
                data.root_link_lin_vel_w[row] - root_velocity[row],
                2.0,
            )
            correction = np.clip(correction, -0.3, 0.3)
            self.target[row, [4, 10]] += correction[1]
            self.target[row, [5, 11]] += correction[0]
        residual = (
            np.clip(self.actions, -1, 1)
            * np.r_[np.full(15, self.cfg.scale), np.full(14, self.cfg.arm_scale)]
        )
        target = np.clip(self.target + residual, controller.limits[:, 0], controller.limits[:, 1])
        self._entity.set_joint_position_target(target, joint_ids=self.body_ids)
        fingers = controller.fingers
        finger_target = np.clip(
            controller.finger_target, fingers.limits[:, 0], fingers.limits[:, 1]
        )
        torque = (
            fingers.kp * (finger_target - data.joint_pos[:, self.finger_ids])
            - fingers.kd * data.joint_vel[:, self.finger_ids]
        )
        self._entity.set_joint_effort_target(
            np.clip(torque, fingers.force_limits[:, 0], fingers.force_limits[:, 1]),
            joint_ids=self.finger_ids,
        )


class ArticulatedObservation:
    def __init__(self, cfg, env):
        self.robot, self.bat = env.scene["robot"], env.scene["bat"]
        self.contacts = env.scene.bind_sensor_data(env.cfg.grasp_sensor_names)
        self.grip = ArticulatedGripState(cfg, env)

    def __call__(self, env):
        robot, bat = self.robot.data, self.bat.data
        rows = self.contacts.read().reshape(env.num_envs, -1, 8, 17)
        if np.any(rows[..., 0] > 8):
            raise RuntimeError("articulated grasp sensor capacity exceeded")
        force = np.where((rows[..., 0] > 0)[..., None], rows[..., 1:4], 0)
        tactile = (
            np.stack(
                (force[..., 0].sum(axis=-1), np.linalg.norm(force[..., 1:], axis=-1).sum(axis=-1)),
                axis=-1,
            )
            / 100
        )
        return np.concatenate(
            (
                robot.projected_gravity_b,
                robot.root_link_lin_vel_w,
                robot.root_link_ang_vel_b,
                robot.joint_pos,
                robot.joint_vel * 0.05,
                bat.root_link_pos_w - robot.root_link_pos_w,
                bat.root_link_quat_w,
                bat.root_link_lin_vel_w,
                self.grip.errors().reshape(env.num_envs, -1),
                tactile.reshape(env.num_envs, -1),
                env.action_manager.get_term("batting").elapsed[:, None] / 8,
                env.action_manager.action,
            ),
            axis=1,
        )


def pose_tracking(env):
    action = env.action_manager.get_term("batting")
    error = env.scene["robot"].data.joint_pos[:, action.body_ids] - action.reference
    return np.exp(-np.mean(error * error, axis=1) / 0.1)


class ArticulatedGripState:
    def __init__(self, cfg, env):
        self.bat = env.scene["bat"]
        self.wrists = env.scene.bind_sensor_data(
            ("grip_left_wrist_world", "grip_right_wrist_world")
        )
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        with np.load(env.cfg.reference_file) as reference:
            data.qpos[:] = reference["qpos"][0]
        mujoco.mj_forward(model, data)
        bat = model.body("cricket_bat").id
        positions = np.array(
            [data.xpos[model.body(f"{side}_wrist_yaw_link").id] for side in ("left", "right")]
        )
        self.offset = (positions - data.xpos[bat]) @ data.xmat[bat].reshape(3, 3)

    def errors(self):
        bat = self.bat.data
        wrists = self.wrists.read().reshape(-1, 2, 3)
        local = np.stack(
            [
                np_quat_apply_inverse_batched(
                    bat.root_link_quat_w, wrists[:, side] - bat.root_link_pos_w
                )
                for side in range(2)
            ],
            axis=1,
        )
        return local - self.offset


class ArticulatedGripFailure(ArticulatedGripState):
    def __call__(self, env, position_limit=0.04):
        return (np.linalg.norm(self.errors(), axis=-1) > position_limit).any(axis=1)
