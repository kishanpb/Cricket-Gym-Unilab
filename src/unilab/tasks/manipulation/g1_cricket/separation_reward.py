"""Reward-only follow-up to the failed contact-seeking residual experiment."""

import numpy as np

from .residual import BattingReward


class SeparationReward(BattingReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.scored = np.zeros(env.num_envs, dtype=bool)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.scored[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.bat.read(), axis=1)
        velocity = self.ball.data.root_link_lin_vel_w[:, 0]
        approach = 5 * np.exp(-((distance / 0.18) ** 2)) * (velocity < 0)
        touching = (self.contact.read().reshape(env.num_envs, 4, 17)[..., 0] > 0).any(axis=1)
        self.hit_seen |= touching
        separated = self.hit_seen & ~touching & ~self.scored
        self.scored |= separated
        return approach + separated * 10 * np.tanh(velocity - 1) / env.step_dt
