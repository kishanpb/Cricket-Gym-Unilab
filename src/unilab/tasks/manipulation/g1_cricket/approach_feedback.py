"""Whole-body cricket residuals over the explicit external locomotion feedback."""

from dataclasses import dataclass

import numpy as np

from .approach import approach_command
from .approach_learning import FOOT_MOTION, HOLDER_WIDTH, SUPPORT_WIDTH, loaded_contact_slip
from .bowling import HOLDER_SENSORS, BowlingAction, BowlingActionCfg
from .scene import CONTACT_WIDTH, SUPPORT_NAMES, SUPPORT_SLOTS


@dataclass(kw_only=True)
class ApproachFeedbackActionCfg(BowlingActionCfg):
    scale: float = 0.1

    def build(self, env):
        return ApproachFeedbackAction(self, env)


class ApproachFeedbackAction(BowlingAction):
    sensor_names = HOLDER_SENSORS + SUPPORT_NAMES + FOOT_MOTION

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._whole_body_raw = np.zeros((env.num_envs, 29), dtype=np.float32)
        self._hold_action = np.zeros((env.num_envs, 8), dtype=np.float32)
        self.slip_squared = np.zeros((env.num_envs, 2))
        self.slip_peak = np.zeros((env.num_envs, 2))
        self.normal_load = np.zeros((env.num_envs, 2))

    @property
    def action_dim(self):
        return 29

    @property
    def raw_action(self):
        return self._whole_body_raw

    def process_actions(self, actions):
        super().process_actions(self._hold_action)
        self._whole_body_raw[:] = actions
        self.processed_action[:] += self.cfg.scale * np.clip(actions, -1, 1)

    def observe(self, sensors, integrated_velocity):
        super().observe(sensors[..., :HOLDER_WIDTH], integrated_velocity)
        support = sensors[..., HOLDER_WIDTH : HOLDER_WIDTH + SUPPORT_WIDTH].reshape(
            *sensors.shape[:2], 2, SUPPORT_SLOTS, CONTACT_WIDTH
        )
        if np.any(support[..., 0] > SUPPORT_SLOTS):
            raise RuntimeError("approach support contact capacity exceeded")
        motion = sensors[..., HOLDER_WIDTH + SUPPORT_WIDTH :].reshape(*sensors.shape[:2], 2, 9)
        speed, load = loaded_contact_slip(support, motion)
        self.slip_squared[:] = np.square(speed).mean(axis=1)
        self.slip_peak[:] = speed.max(axis=1)
        self.normal_load[:] = load.mean(axis=1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self._whole_body_raw, self.slip_squared, self.slip_peak, self.normal_load):
            value[ids] = 0


def lane_error(env):
    robot = env.scene["robot"]
    return robot.data.root_link_pos_w[:, 1] - robot.data.default_root_state[:, 1]


def gait_observation(env):
    action = env.action_manager.get_term("residual")
    return np.column_stack(
        (np.sqrt(action.slip_squared), action.normal_load / 100, lane_error(env))
    )


def loaded_slip_cost(env):
    return env.action_manager.get_term("residual").slip_squared.sum(axis=1)


def lane_tracking(env):
    return np.exp(-np.square(lane_error(env) / 0.1))


def velocity_tracking(env):
    error = env.scene["robot"].data.root_link_lin_vel_w[:, :2] - approach_command(env)[:, :2]
    return np.exp(-np.square(error / 0.25).sum(axis=1))
