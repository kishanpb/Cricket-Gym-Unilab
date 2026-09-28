"""Shared bat-pose commands mapped to both arms without constraining the free bat."""
from dataclasses import dataclass
import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from .batting_projection import rotation_log_jacobian
from .incoming_swing import FrozenGuardSwingAction, FrozenGuardSwingActionCfg, IncomingSwingAction, IncomingSwingObservation
from .prior import SDK_JOINTS

class SharedBatPoseIK:

    def __init__(self, model, joint_offset_limit, *, include_waist=False):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(name).id for name in SDK_JOINTS[12 if include_waist else 15:]]
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints].copy()
        self.offset_limit = joint_offset_limit
        self.wrists = [model.body(f'{hand}_wrist_yaw_link').id for hand in ('left', 'right')]
        self.bat = model.body('cricket_bat').id

    def solve(self, pose, offset, *, previous=None):
        offset = np.asarray(offset)
        if offset.shape != (6,) or not np.isfinite(offset).all():
            raise ValueError('shared bat offset must contain six finite pose coordinates')
        if not np.any(offset):
            return (np.zeros(len(self.q)), {'feasible': True, 'position_error_m': 0.0, 'rotation_error_rad': 0.0})
        model, data = (self.model, self.data)
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        origin = data.xpos[self.bat].copy()
        delta = Rotation.from_rotvec(offset[3:]).as_matrix()
        positions = origin + offset[:3] + (data.xpos[self.wrists] - origin) @ delta.T
        rotations = delta @ data.xmat[self.wrists].reshape(2, 3, 3)
        return self.solve_targets(pose, positions, rotations, previous=previous)

    def solve_targets(self, pose, positions, rotations, *, previous=None):
        model, data = (self.model, self.data)
        data.qpos[:] = pose
        nominal = pose[self.q].copy()
        lower = np.maximum(self.limits[:, 0] + 0.02, nominal - self.offset_limit)
        upper = np.minimum(self.limits[:, 1] - 0.02, nominal + self.offset_limit)

        def errors(value, jacobian=False):
            data.qpos[self.q] = value
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            values, derivatives = ([], [])
            for wrist, position, rotation in zip(self.wrists, positions, rotations, strict=True):
                angular_error = Rotation.from_matrix(rotation.T @ data.xmat[wrist].reshape(3, 3)).as_rotvec()
                values.extend((data.xpos[wrist] - position, 0.1 * angular_error))
                if jacobian:
                    linear, angular = (np.empty((3, model.nv)), np.empty((3, model.nv)))
                    mujoco.mj_jacBody(model, data, linear, angular, wrist)
                    derivatives.extend((linear[:, self.v], 0.1 * rotation_log_jacobian(angular_error) @ rotation.T @ angular[:, self.v]))
            if jacobian:
                return np.vstack((*derivatives, 0.0001 * np.eye(len(self.q))))
            return np.r_[np.concatenate(values), 0.0001 * (value - nominal)]
        initial = nominal if previous is None else nominal + previous
        result = least_squares(errors, np.clip(initial, lower, upper), jac=lambda value: errors(value, jacobian=True), bounds=(lower, upper), max_nfev=30, ftol=1e-08, xtol=1e-08, gtol=1e-08)
        residual = errors(result.x)[:12].reshape(2, 6)
        position_error = float(np.linalg.norm(residual[:, :3], axis=1).max())
        rotation_error = float(np.linalg.norm(residual[:, 3:], axis=1).max() / 0.1)
        return (result.x - nominal, {'feasible': position_error <= 0.001 and rotation_error <= 0.01, 'position_error_m': position_error, 'rotation_error_rad': rotation_error})

@dataclass(kw_only=True)
class SharedBatPoseActionCfg(FrozenGuardSwingActionCfg):
    arm_residual_scale: float = 0.4
    translation_scale: float = 0.1
    rotation_scale: float = 0.35
    translation_rate: float = 0.3
    rotation_rate: float = 2.0

    def build(self, env):
        return SharedBatPoseAction(self, env)

class SharedBatPoseAction(FrozenGuardSwingAction):

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        values = (cfg.translation_scale, cfg.rotation_scale, cfg.translation_rate, cfg.rotation_rate)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('shared pose scales and rates must be positive and finite')
        self.ik = SharedBatPoseIK(env.get_playback_model(), cfg.arm_residual_scale)
        self.pose_commands = np.zeros((env.num_envs, 7), dtype=np.float32)
        self.pose_offset = np.zeros((env.num_envs, 6))
        self.arm_goal = np.zeros((env.num_envs, 14))
        self.ik_rejected = np.zeros(env.num_envs, dtype=bool)
        self.rejected_steps = np.zeros(env.num_envs, dtype=int)

    @property
    def action_dim(self):
        return 7

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.pose_commands[ids] = 0
        self.pose_offset[ids] = 0
        self.arm_goal[ids] = 0
        self.ik_rejected[ids] = False
        self.rejected_steps[ids] = 0

    def process_actions(self, actions):
        self.pose_commands[:] = actions
        scales = np.r_[np.full(3, self.cfg.translation_scale), np.full(3, self.cfg.rotation_scale)]
        desired = np.clip(actions[:, 1:], -1, 1) * scales
        change = desired - self.pose_offset
        for selected, rate in ((slice(0, 3), self.cfg.translation_rate), (slice(3, 6), self.cfg.rotation_rate)):
            length = np.linalg.norm(change[:, selected], axis=1, keepdims=True)
            change[:, selected] *= np.minimum(1, rate * self._env.step_dt / np.maximum(length, 1e-12))
        for row in range(self._env.num_envs):
            nominal = self.swing.path.pose(self.ik.model, self.phase[row])
            requested = self.pose_offset[row] + change[row]
            goal, result = self.ik.solve(nominal, requested, previous=self.arm_goal[row])
            self.ik_rejected[row] = not result['feasible']
            if not result['feasible']:
                requested = self.pose_offset[row].copy()
                goal, result = self.ik.solve(nominal, requested, previous=self.arm_goal[row])
                if not result['feasible']:
                    requested[:] = 0
                    goal = np.zeros(14)
            self.pose_offset[row], self.arm_goal[row] = (requested, goal)
        self.rejected_steps += self.ik_rejected
        observation = self._env.observation_manager.get_term_cfg('policy', 'body_grip')
        values = IncomingSwingObservation.__call__(observation.func, self._env, reference_relative_legs=True).astype(np.float32)
        obs = TensorDict({'policy': torch.from_numpy(values)}, batch_size=[self._env.num_envs])
        with torch.inference_mode():
            legs = self.guard(obs).numpy()[:, :12]
        commands = np.column_stack((legs, actions[:, 0], self.arm_goal / self.cfg.arm_residual_scale))
        IncomingSwingAction.process_actions(self, commands)

    def update_arm_residual(self, dt):
        change = self.arm_goal - self.arm_residual
        largest = np.max(np.abs(change), axis=1, keepdims=True)
        scale = np.minimum(1, self.cfg.arm_residual_rate * dt / np.maximum(largest, 1e-12))
        self.arm_residual += scale * change

class SharedBatPoseObservation(IncomingSwingObservation):

    def __call__(self, env, **kwargs):
        action = env.action_manager.get_term('batting')
        return np.column_stack((super().__call__(env, **kwargs), action.pose_offset, action.pose_commands, action.ik_rejected))
