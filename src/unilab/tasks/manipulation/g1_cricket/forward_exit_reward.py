"""Nonnegative separation shaping; the independent shot gate stays unchanged."""

import numpy as np

from .separation_reward import SeparationReward


class ForwardExitReward(SeparationReward):
    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.bat.read(), axis=1)
        velocity = self.ball.data.root_link_lin_vel_w[:, 0]
        approach = 5 * np.exp(-((distance / 0.18) ** 2)) * (velocity < 0)
        touching = (self.contact.read().reshape(env.num_envs, 4, 17)[..., 0] > 0).any(axis=1)
        self.hit_seen |= touching
        separated = self.hit_seen & ~touching & ~self.scored
        self.scored |= separated
        return approach + separated * 5 * (1 + np.tanh(velocity - 1)) / env.step_dt
