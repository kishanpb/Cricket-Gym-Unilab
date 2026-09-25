"""Continuous delivery prototype: reference arms over live locomotion feedback."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .approach_feedback import ApproachFeedbackAction, ApproachFeedbackActionCfg
from .prior import SDK_JOINTS
from .running import END_TIME, GATHER_TIME, RELEASE_TIME

REFERENCE_START = 2.8


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def arm_weight(time):
    local = np.asarray(time) - REFERENCE_START
    return smoothstep(local / GATHER_TIME) * (1 - smoothstep((local - END_TIME) / 0.6))


def moving_command(env):
    time = env.episode_length_buf * env.step_dt
    command = np.zeros((env.num_envs, 3))
    command[:, 0] = np.interp(time, [0, 1, 2, 4.8, 5.8, 8], [0, 0, 1, 1, 0, 0])
    return command


@dataclass(kw_only=True)
class MovingDeliveryActionCfg(ApproachFeedbackActionCfg):
    reference_directory: str

    def build(self, env):
        return MovingDeliveryAction(self, env)


class MovingDeliveryAction(ApproachFeedbackAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        joints = np.array([model.joint(name).id for name in SDK_JOINTS[15:]])
        path = Path(cfg.reference_directory) / f"{env.cfg.handedness}_dense_reference.npz"
        with np.load(path) as reference:
            self.arm_reference = PchipInterpolator(
                reference["times"], reference["qpos"][:, model.jnt_qposadr[joints]], axis=0
            )
        self.arm_limits = model.jnt_range[joints].copy()

    def process_actions(self, actions):
        time = self._env.episode_length_buf * self._env.step_dt
        self._hold_action[:, 7] = time >= REFERENCE_START + RELEASE_TIME
        super().process_actions(actions)
        target = self.arm_reference(np.clip(time - REFERENCE_START, 0, END_TIME))
        weight = arm_weight(time)[:, None]
        # Only motor targets change; legs and trunk retain the live prior's feedback.
        current = self.processed_action[:, 15:]
        current[:] = np.clip(
            current + weight * (target - current), self.arm_limits[:, 0], self.arm_limits[:, 1]
        )
