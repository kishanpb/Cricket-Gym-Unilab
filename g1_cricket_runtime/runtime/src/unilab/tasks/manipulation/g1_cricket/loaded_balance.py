"""Loaded stance adaptation before the full cricket swing curriculum."""

from dataclasses import dataclass

import mujoco
import numpy as np

from unilab.managers.action_manager import ActionTerm, ActionTermCfg

from .articulated_batting import ArticulatedStanceController
from .articulated_learning import FINGER_JOINTS, ArticulatedGripState, ArticulatedObservation
from .leg_locomotion import LegLocomotionPolicy
from .prior import SDK_JOINTS


@dataclass(kw_only=True)
class LoadedBalanceActionCfg(ActionTermCfg):
    assets: str = "g1_cricket_results/locomotion_assets"
    finger_preload: float = 0.2

    def build(self, env):
        return LoadedBalanceAction(self, env)


class LoadedBalanceAction(ActionTerm):
    requires_substep_state_feedback = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        with np.load(env.cfg.reference_file) as saved:
            pose = saved["qpos"][0]
        self.controller = ArticulatedStanceController(model, pose, preload=cfg.finger_preload)
        self.legs = LegLocomotionPolicy(cfg.assets)
        self.legs.validate_position_servos(model)
        self.body_ids, _ = self._entity.find_joints(SDK_JOINTS, preserve_order=True)
        self.finger_ids, _ = self._entity.find_joints(FINGER_JOINTS, preserve_order=True)
        self.actions = np.zeros((env.num_envs, 12), dtype=np.float32)
        self.elapsed = np.zeros(env.num_envs)
        self.reference = np.tile(pose[self.controller.q], (env.num_envs, 1))
        self.target = self.reference.copy()
        self.limits = np.concatenate((self.controller.limits, self.controller.fingers.limits))

    @property
    def action_dim(self):
        return 12

    @property
    def raw_action(self):
        return self.actions

    def process_actions(self, actions):
        self.actions[:] = actions
        self.elapsed += self._env.step_dt

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.actions[ids] = 0
        self.elapsed[ids] = 0

    def apply_actions(self):
        controller = self.controller
        self.target[:] = np.clip(
            self.reference + controller.offset, controller.limits[:, 0], controller.limits[:, 1]
        )
        self.target[:, :12] = self.legs.default + self.legs.config["action_scale"] * self.actions
        self._entity.set_joint_position_target(self.target, joint_ids=self.body_ids)
        fingers = controller.fingers
        data = self._entity.data
        desired = np.clip(controller.finger_target, fingers.limits[:, 0], fingers.limits[:, 1])
        torque = fingers.kp * (desired - data.joint_pos[:, self.finger_ids])
        torque -= fingers.kd * data.joint_vel[:, self.finger_ids]
        self._entity.set_joint_effort_target(
            np.clip(torque, fingers.force_limits[:, 0], fingers.force_limits[:, 1]),
            joint_ids=self.finger_ids,
        )


class LoadedBalanceObservation(ArticulatedObservation):
    def __call__(self, env):
        action = env.action_manager.get_term("batting")
        robot = self.robot.data
        first = []
        for row in range(env.num_envs):
            values = action.legs.observation(
                robot.root_link_quat_w[row],
                robot.root_link_ang_vel_b[row],
                robot.joint_pos[row, action.body_ids[:12]],
                robot.joint_vel[row, action.body_ids[:12]],
                [0, 0, 0],
                action.elapsed[row],
            )
            values[33:45] = action.actions[row]
            first.append(values)
        return np.column_stack((first, super().__call__(env)))


class ResetLoadedGuard:
    def __init__(self, cfg, env):
        self.ball = env.scene["ball"]

    def __call__(self, env, env_ids):
        state = self.ball.data.default_root_state[env_ids].copy()
        state[:, :3] = env.scene.env_origins[env_ids] + [20, -20, 1]
        state[:, 3:7] = [1, 0, 0, 0]
        state[:, 7:] = 0
        self.ball.write_root_state_to_sim(state, env_ids=env_ids)


class LoadedBalanceReward(ArticulatedGripState):
    def __call__(self, env, follow_swing_reference=False):
        action = env.action_manager.get_term("batting")
        robot = env.scene["robot"].data
        offset = robot.root_link_pos_w - env.scene.env_origins - action.controller.reference[:3]
        if follow_swing_reference:
            path = action.swing.path
            phase = np.clip(action.phase, path.times[0], path.times[-1])
            offset -= path.spline(phase)[:, :3]
        upright = np.clip(-robot.projected_gravity_b[:, 2], 0, 1)
        position = np.exp(-np.sum(offset**2, axis=1) / 0.04**2)
        grip = np.exp(-np.sum(self.errors() ** 2, axis=(1, 2)) / 0.01**2)
        return 0.4 * upright + 0.3 * position + 0.3 * grip


class PlantedFootReward:
    def __init__(self, cfg, env):
        self.robot = env.scene["robot"]
        names = ("left_ankle_roll_link", "right_ankle_roll_link")
        self.feet, _ = self.robot.find_bodies(names, preserve_order=True)
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        data.qpos[:] = env.action_manager.get_term("batting").controller.reference
        mujoco.mj_kinematics(model, data)
        self.reference = np.array([data.body(name).xpos for name in names])

    def __call__(self, env):
        offset = self.robot.data.body_link_pos_w[:, self.feet] - env.scene.env_origins[:, None]
        offset -= self.reference
        worst_squared = np.sum(offset**2, axis=-1).max(axis=1)
        return np.exp(-worst_squared / 0.03**2) - 1


def loaded_joint_limit_failure(env):
    action = env.action_manager.get_term("batting")
    q = env.scene["robot"].data.joint_pos[:, np.r_[action.body_ids, action.finger_ids]]
    return ((q < action.limits[:, 0] - 0.02) | (q > action.limits[:, 1] + 0.02)).any(axis=1)
