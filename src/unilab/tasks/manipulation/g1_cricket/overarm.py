"""Selected-arm absolute references with unchanged locomotion and motor authority."""

from dataclasses import dataclass

import numpy as np

from .bowling import BowlingAction, BowlingActionCfg


def absolute_targets(actions, neutral, limits):
    fraction = np.tanh(actions)
    scale = np.where(fraction >= 0, limits[:, 1] - neutral, neutral - limits[:, 0])
    return neutral + fraction * scale


def actions_for_targets(targets, neutral, limits):
    offset = targets - neutral
    scale = np.where(offset >= 0, limits[:, 1] - neutral, neutral - limits[:, 0])
    fraction = offset / scale
    if np.any(np.abs(fraction) >= 1):
        raise ValueError("absolute arm target must lie strictly inside soft joint limits")
    return np.arctanh(fraction)


@dataclass(kw_only=True)
class OverarmActionCfg(BowlingActionCfg):
    def build(self, env):
        return OverarmAction(self, env)


class OverarmAction(BowlingAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        neutral = self._entity.data.default_joint_pos[:, self.arm_ids]
        target = absolute_targets(actions[:, :7], neutral, self.joint_limits[self.arm_ids])
        self.processed_action[:, self.arm_ids] = target
        self.residual_radians[:] = target - neutral
