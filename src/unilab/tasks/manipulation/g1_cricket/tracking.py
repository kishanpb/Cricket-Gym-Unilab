"""Whole-body reference residuals for cricket, without a frozen walking policy."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from unilab.managers.action_manager import ActionTerm, ActionTermCfg

from .bimanual import build_bimanual_scene, support_feedforward
from .prior import SDK_JOINTS
from .task import G1CricketCfg


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

    def build(self, env):
        return SupportedCricketReferenceAction(self, env)


class SupportedCricketReferenceAction(CricketReferenceAction):
    def __init__(self, cfg, env):
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

    def process_actions(self, actions):
        super().process_actions(actions)
        self.target += self.velocity_gain * self.command.joint_vel
        self.target += self.gravity_offset[self.command.time_steps]
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
