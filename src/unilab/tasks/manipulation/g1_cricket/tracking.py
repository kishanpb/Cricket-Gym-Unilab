"""Whole-body reference residuals for cricket, without a frozen walking policy."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from unilab.managers.action_manager import ActionTerm, ActionTermCfg

from .bimanual import build_bimanual_scene, support_feedforward
from .prior import SDK_JOINTS
from .task import G1CricketCfg


def ankle_balance(
    reference_quaternion, quaternion, angular_velocity, gain, reference_angular_velocity=None
):
    tilt = np.empty(3)
    mujoco.mju_subQuat(tilt, quaternion, reference_quaternion)
    reference_rotation, rotation = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(reference_rotation, reference_quaternion)
    mujoco.mju_quat2Mat(rotation, quaternion)
    velocity = reference_rotation.reshape(3, 3).T @ rotation.reshape(3, 3) @ angular_velocity
    if reference_angular_velocity is not None:
        velocity -= reference_angular_velocity
    return np.clip(gain * (0.7 * tilt[:2] + 0.1 * velocity[:2]), -0.3, 0.3)


def root_position_balance(reference_quaternion, position_error, velocity_error, gain):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, reference_quaternion)
    error = rotation.reshape(3, 3).T @ (position_error + 0.2 * velocity_error)
    return gain * np.array([-error[1], error[0]])


@dataclass
class G1BimanualTrackingCfg(G1CricketCfg):
    def build_scene(self, source: Path, destination: Path) -> tuple[str, ...]:
        return build_bimanual_scene(source, destination, self.handedness)


@dataclass(kw_only=True)
class CricketReferenceActionCfg(ActionTermCfg):
    scale: float = 0.25
    command_name: str = "motion"

    def build(self, env):
        return CricketReferenceAction(self, env)


class CricketReferenceAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = env.command_manager.get_term(cfg.command_name)
        self._raw = np.zeros((env.num_envs, len(SDK_JOINTS)), dtype=np.float32)
        self.target = self._entity.data.default_joint_pos.copy()

    @property
    def action_dim(self):
        return len(SDK_JOINTS)

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        self.target[:] = self.command.joint_pos + self.cfg.scale * np.clip(actions, -1, 1)

    def apply_actions(self):
        self._entity.set_joint_position_target(self.target)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.target[ids] = self._entity.data.default_joint_pos[ids]


@dataclass(kw_only=True)
class SupportedCricketReferenceActionCfg(CricketReferenceActionCfg):
    reference_file: str
    lookahead_frames: int = 0

    def build(self, env):
        return SupportedCricketReferenceAction(self, env)


class SupportedCricketReferenceAction(CricketReferenceAction):
    def __init__(self, cfg, env):
        if type(cfg.lookahead_frames) is not int or cfg.lookahead_frames < 0:
            raise ValueError("lookahead_frames must be a nonnegative integer")
        super().__init__(cfg, env)
        model = env.get_playback_model()
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        self.limits = model.jnt_range[joints].copy()
        # Round inward before writing float32 controls at a hard joint limit.
        self.control_limits = np.nextafter(
            self.limits.astype(np.float32), self.limits[:, ::-1].astype(np.float32)
        )
        self.velocity_gain = -model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0]
        with np.load(cfg.reference_file) as saved:
            poses = saved["qpos"]
        self.reference_root_quaternion = poses[:, 3:7].copy()
        self.reference_root_position = poses[:, :3].copy()
        self.reference_root_velocity = np.gradient(
            self.reference_root_position, 1 / self.command.motion.fps, axis=0
        )
        motion = self.command.motion.get_motion_at_frame(np.arange(len(poses)))
        np.testing.assert_allclose(
            motion.joint_pos, poses[:, model.jnt_qposadr[joints]], atol=1e-6, rtol=0
        )
        computed = [support_feedforward(model, pose) for pose in poses]
        if max(np.linalg.norm(residual) for _, residual in computed) > 1e-6:
            raise ValueError("reference has no static nonnegative foot-support solution")
        self.gravity_offset = (
            np.asarray([torque for torque, _ in computed], dtype=np.float32)
            / model.actuator_gainprm[:, 0]
        )

    def _reference_with_feedforward(self, actions):
        frames = self.command.time_steps
        if self.cfg.lookahead_frames:
            frames = np.minimum(frames + self.cfg.lookahead_frames, len(self.gravity_offset) - 1)
            motion = self.command.motion.get_motion_at_frame(frames)
            self._raw[:] = actions
            self.target[:] = motion.joint_pos + self.cfg.scale * np.clip(actions, -1, 1)
            self.target += self.velocity_gain * motion.joint_vel
        else:
            super().process_actions(actions)
            self.target += self.velocity_gain * self.command.joint_vel
        self.target += self.gravity_offset[frames]

    def process_actions(self, actions):
        self._reference_with_feedforward(actions)
        np.clip(self.target, self.control_limits[:, 0], self.control_limits[:, 1], out=self.target)


@dataclass(kw_only=True)
class BalancedCricketReferenceActionCfg(SupportedCricketReferenceActionCfg):
    balance_gain: float = 4.0
    waist_tracking_gain: float = 0.0
    root_position_gain: float = 0.0

    def build(self, env):
        return BalancedCricketReferenceAction(self, env)


class BalancedCricketReferenceAction(SupportedCricketReferenceAction):
    def process_actions(self, actions):
        self._reference_with_feedforward(actions)
        reference = self.reference_root_quaternion[self.command.time_steps]
        quaternion = self._entity.data.root_link_quat_w
        angular_velocity = self._entity.data.root_link_ang_vel_b
        position_error = self._entity.data.root_link_pos_w - (
            self.reference_root_position[self.command.time_steps] + self._env.scene.env_origins
        )
        velocity_error = (
            self._entity.data.root_link_lin_vel_w
            - self.reference_root_velocity[self.command.time_steps]
        )
        for i in range(self.num_envs):
            correction = ankle_balance(
                reference[i], quaternion[i], angular_velocity[i], self.cfg.balance_gain
            )
            correction += root_position_balance(
                reference[i], position_error[i], velocity_error[i], self.cfg.root_position_gain
            )
            correction = np.clip(correction, -0.3, 0.3)
            self.target[i, [4, 10]] += correction[1]
            self.target[i, [5, 11]] += correction[0]
        self.target[:, 14] += self.cfg.waist_tracking_gain * (
            self.command.joint_pos[:, 14] - self._entity.data.joint_pos[:, 14]
        )
        np.clip(self.target, self.control_limits[:, 0], self.control_limits[:, 1], out=self.target)


def export_reference(model: mujoco.MjModel, qpos: np.ndarray, fps: int, destination: Path):
    """Export offline FK in the shared MotionLoader's compiled body-ID layout."""
    data = mujoco.MjData(model)
    velocity = np.empty((len(qpos), model.nv))
    for index in range(len(qpos)):
        before, after = max(index - 1, 0), min(index + 1, len(qpos) - 1)
        mujoco.mj_differentiatePos(
            model, velocity[index], (after - before) / fps, qpos[before], qpos[after]
        )
    position = np.empty((len(qpos), model.nbody, 3))
    quaternion = np.empty((len(qpos), model.nbody, 4))
    body_velocity = np.zeros((len(qpos), model.nbody, 6))
    for index, pose in enumerate(qpos):
        data.qpos[:], data.qvel[:] = pose, velocity[index]
        mujoco.mj_forward(model, data)
        position[index], quaternion[index] = data.xpos, data.xquat
        for body in range(1, model.nbody):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body, body_velocity[index, body], 0
            )
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    np.savez_compressed(
        destination,
        fps=np.array([fps], dtype=np.int32),
        joint_pos=qpos[:, model.jnt_qposadr[joints]],
        joint_vel=velocity[:, model.jnt_dofadr[joints]],
        body_pos_w=position,
        body_quat_w=quaternion,
        body_lin_vel_w=body_velocity[..., 3:],
        body_ang_vel_w=body_velocity[..., :3],
    )
