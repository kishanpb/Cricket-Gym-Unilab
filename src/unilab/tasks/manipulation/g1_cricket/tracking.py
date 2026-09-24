"""Whole-body reference residuals for cricket, without a frozen walking policy."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from unilab.managers.action_manager import ActionTerm, ActionTermCfg

from .bimanual import build_bimanual_scene
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
