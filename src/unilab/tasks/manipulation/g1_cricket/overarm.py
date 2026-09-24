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
    prior_target_limit_fraction: float | None = None

    def build(self, env):
        fraction = self.prior_target_limit_fraction
        if fraction is not None and not 0 < fraction <= 1:
            raise ValueError("prior target limit fraction must be in (0, 1]")
        return OverarmAction(self, env)


class OverarmAction(BowlingAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.prior_ids = np.setdiff1d(np.arange(29), self.arm_ids)
        self.prior_target_limits = None
        if cfg.prior_target_limit_fraction is not None:
            bounds = self.joint_limits[self.prior_ids]
            center = bounds.mean(axis=1)
            half = (bounds[:, 1] - bounds[:, 0]) * cfg.prior_target_limit_fraction / 2
            self.prior_target_limits = np.stack((center - half, center + half), axis=1)

    def process_actions(self, actions):
        super().process_actions(actions)
        neutral = self._entity.data.default_joint_pos[:, self.arm_ids]
        target = absolute_targets(actions[:, :7], neutral, self.joint_limits[self.arm_ids])
        self.processed_action[:, self.arm_ids] = target
        self.residual_radians[:] = target - neutral
        if self.prior_target_limits is not None:
            self.processed_action[:, self.prior_ids] = np.clip(
                self.processed_action[:, self.prior_ids],
                self.prior_target_limits[:, 0],
                self.prior_target_limits[:, 1],
            )
