"""Whole-body running-delivery tracking with a declared scheduled holder release."""

from dataclasses import dataclass

import mujoco
import numpy as np

from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand, MotionCommandCfg
from unilab.utils.rotation import np_quat_apply_batched

from .bowling import HOLDER_SENSORS
from .prior import SDK_JOINTS
from .running import RELEASE_TIME
from .scene import CONTACT_WIDTH, SUPPORT_NAMES, SUPPORT_SLOTS
from .tracking import (
    CricketReferenceAction,
    CricketReferenceActionCfg,
    ankle_balance,
    root_position_balance,
)


@dataclass(kw_only=True)
class RunningMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return RunningMotionCommand(self, env)


class RunningMotionCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (
            np.any(self._pose_range)
            or np.any(self._velocity_range)
            or any(self._joint_position_range)
            or any(self._joint_default_position_range)
        ):
            raise ValueError("held-ball running resets require unperturbed reference poses")
        self.ball = env.scene["ball"]
        self.wrist_index = cfg.body_names.index(f"{env.cfg.handedness}_wrist_yaw_link")
        self.offset = np.array([0.15, 0.06 if env.cfg.handedness == "left" else -0.06, 0])

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        motion = self._resample_motion
        wrist = self.wrist_index
        quaternion = motion.body_quat_w[:, wrist]
        offset = np_quat_apply_batched(quaternion, np.broadcast_to(self.offset, (len(env_ids), 3)))
        position = motion.body_pos_w[:, wrist] + offset + self._env.scene.env_origins[env_ids]
        angular = motion.body_ang_vel_w[:, wrist]
        linear = motion.body_lin_vel_w[:, wrist] + np.cross(angular, offset)
        self.ball.write_root_state_to_sim(
            np.concatenate((position, quaternion, linear, angular), axis=1), env_ids=env_ids
        )


@dataclass(kw_only=True)
class RunningReferenceActionCfg(CricketReferenceActionCfg):
    reference_file: str
    balance_gain: float = 4.0
    root_position_gain: float = 4.0
    waist_tracking_gain: float = 1.0
    use_reference_torque: bool = False
    release_time: float | None = RELEASE_TIME

    def build(self, env):
        return RunningReferenceAction(self, env)


class RunningReferenceAction(CricketReferenceAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        self.limits = model.jnt_range[joints].copy()
        self.control_limits = np.nextafter(
            self.limits.astype(np.float32), self.limits[:, ::-1].astype(np.float32)
        )
        self.velocity_gain = -model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0]
        with np.load(cfg.reference_file) as saved:
            self.poses = saved["qpos"].copy()
            torque = saved["torque"].copy() if cfg.use_reference_torque else None
        motion = self.command.motion.get_motion_at_frame(np.arange(len(self.poses)))
        np.testing.assert_allclose(
            motion.joint_pos, self.poses[:, model.jnt_qposadr[joints]], atol=1e-6, rtol=0
        )
        self.root_velocity = motion.body_lin_vel_w[:, 0]
        if torque is not None:
            if torque.shape != (len(self.poses), 29) or not np.isfinite(torque).all():
                raise ValueError("reference torque must contain finite 29-joint frame values")
            self.bias_offset = torque / model.actuator_gainprm[:, 0]
        else:
            data = mujoco.MjData(model)
            self.bias_offset = np.empty((len(self.poses), 29))
            for i, pose in enumerate(self.poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                self.bias_offset[i] = (
                    data.qfrc_bias[model.jnt_dofadr[joints]] / model.actuator_gainprm[:, 0]
                )
        self.released = np.zeros(env.num_envs, dtype=bool)
        self.just_released = self.released.copy()
        self.release_physics = np.zeros(
            (env.num_envs, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        )
        self.peak_load = np.zeros(env.num_envs)
        self.impulse_world = np.zeros((env.num_envs, 3))
        self.touch_fraction = np.zeros(env.num_envs)
        self.constraints = env.equality_constraints
        env.set_substep_observer(HOLDER_SENSORS, "cricket_ball", self.observe)

    @property
    def processed_action(self):
        return self.target

    def process_actions(self, actions):
        super().process_actions(actions)
        frames = self.command.time_steps
        reference = self.poses[frames]
        self.target += self.velocity_gain * self.command.joint_vel + self.bias_offset[frames]
        robot = self._entity.data
        position_error = robot.root_link_pos_w - (reference[:, :3] + self._env.scene.env_origins)
        velocity_error = robot.root_link_lin_vel_w - self.root_velocity[frames]
        for i in range(self.num_envs):
            correction = ankle_balance(
                reference[i, 3:7],
                robot.root_link_quat_w[i],
                robot.root_link_ang_vel_b[i],
                self.cfg.balance_gain,
            )
            correction += root_position_balance(
                reference[i, 3:7], position_error[i], velocity_error[i], self.cfg.root_position_gain
            )
            correction = np.clip(correction, -0.3, 0.3)
            self.target[i, [4, 10]] += correction[1]
            self.target[i, [5, 11]] += correction[0]
        self.target[:, 14] += self.cfg.waist_tracking_gain * (
            self.command.joint_pos[:, 14] - robot.joint_pos[:, 14]
        )
        np.clip(self.target, self.control_limits[:, 0], self.control_limits[:, 1], out=self.target)
        self.just_released[:] = False
        if self.cfg.release_time is not None:
            self.just_released[:] = ~self.released & (
                frames / self.command.motion.fps >= self.cfg.release_time
            )
        ids = np.flatnonzero(self.just_released)
        if len(ids):
            self.release_physics[ids] = self._env.get_physics_state_snapshot()[ids]
            self.constraints.set_equality_active(ids, np.zeros((len(ids), 1), dtype=bool))
            self.released[ids] = True

    def observe(self, sensors, integrated_velocity):
        if not np.isfinite(sensors).all() or not np.isfinite(integrated_velocity).all():
            raise RuntimeError("non-finite running-delivery load sensor")
        force, quaternion = sensors[..., :3], sensors[..., 3:7]
        self.peak_load[:] = np.linalg.norm(force, axis=-1).max(axis=1)
        world = np_quat_apply_batched(quaternion.reshape(-1, 4), force.reshape(-1, 3))
        self.impulse_world[:] = world.reshape(force.shape).sum(axis=1) * self._env.physics_dt
        contacts = sensors[..., 7:].reshape(*sensors.shape[:2], 4, CONTACT_WIDTH)
        if np.any(contacts[..., 0] > 4):
            raise RuntimeError("running-delivery hand contact capacity exceeded")
        self.touch_fraction[:] = (contacts[..., 0] > 0).any(axis=-1).mean(axis=1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if self.constraints.get_equality_names() != ("ball_holder",):
            raise ValueError("running delivery requires exactly the declared ball holder")
        ids = slice(None) if env_ids is None else env_ids
        for array in (
            self.released,
            self.just_released,
            self.release_physics,
            self.peak_load,
            self.impulse_world,
            self.touch_fraction,
        ):
            array[ids] = 0


def running_contact_observation(env):
    action = env.action_manager.get_term("reference")
    return np.concatenate(
        (
            action.peak_load[:, None] / 100,
            action.impulse_world / 10,
            action.touch_fraction[:, None],
            action.released[:, None],
        ),
        axis=1,
    )


class RunningSupportObservation:
    """Foot contact snapshots and reference phase, not calibrated tactile taxels."""

    def __init__(self, cfg, env):
        self.support = env.scene.bind_sensor_data(SUPPORT_NAMES)

    def __call__(self, env):
        rows = self.support.read().reshape(env.num_envs, 2, SUPPORT_SLOTS, CONTACT_WIDTH)
        if np.any(rows[..., 0] > SUPPORT_SLOTS):
            raise RuntimeError("running foot contact sensor capacity exceeded")
        force = np.where((rows[..., 0] > 0)[..., None], rows[..., 1:4], 0)
        normal = force[..., 0].sum(axis=-1)
        shear = np.linalg.norm(force[..., 1:], axis=-1).sum(axis=-1)
        feet = np.stack((normal > 1, normal / 100, shear / 100), axis=-1)
        command = env.command_manager.get_term("motion")
        duration = env.max_episode_length * env.cfg.ctrl_dt
        phase = command.time_steps / command.motion.fps / duration
        return np.concatenate((feet.reshape(env.num_envs, 6), phase[:, None]), axis=1)
