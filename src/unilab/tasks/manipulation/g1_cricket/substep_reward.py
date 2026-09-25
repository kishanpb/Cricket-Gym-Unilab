"""The v3 reward curve with first separation acquired at physics-substep rate."""

import numpy as np

from .separation_reward import SeparationReward


class SubstepForwardExitReward(SeparationReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.pending = np.zeros(env.num_envs, dtype=bool)
        self.separation_velocity = np.zeros(env.num_envs)
        env.set_substep_observer(("ball_bat",), "cricket_ball", self.observe)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.pending[ids] = False
        self.separation_velocity[ids] = 0

    def contact_rows(self, sensors):
        rows = sensors.reshape(*sensors.shape[:2], 4, 17)
        if np.any(rows[..., 0] > 4):
            raise RuntimeError("G1 cricket contact sensor capacity exceeded")
        return rows

    def touching(self, sensors):
        return (self.contact_rows(sensors)[..., 0] > 0).any(axis=-1)

    def observe(self, sensors, integrated_velocity):
        touching = self.touching(sensors)
        seen = np.maximum.accumulate(touching, axis=1) | self.hit_seen[:, None]
        separated = seen & ~touching & ~(self.scored | self.pending)[:, None]
        fresh = separated.any(axis=1)
        rows = np.flatnonzero(fresh)
        self.separation_velocity[rows] = integrated_velocity[
            rows, separated.argmax(axis=1)[rows], 0
        ]
        self.pending |= fresh
        self.hit_seen |= touching.any(axis=1)

    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.bat.read(), axis=1)
        velocity = self.ball.data.root_link_lin_vel_w[:, 0]
        approach = 5 * np.exp(-((distance / 0.18) ** 2)) * (velocity < 0)
        bonus = self.pending * 5 * (1 + np.tanh(self.separation_velocity - 1)) / env.step_dt
        self.scored |= self.pending
        self.pending[:] = False
        return approach + bonus
