"""Pre-contact forward blade motion with the existing first-separation bonus."""

import numpy as np

from .substep_reward import SubstepForwardExitReward


class ForwardSwingReward(SubstepForwardExitReward):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.bat_body = self.robot.body_names.index("cricket_bat")

    def blade_velocity(self):
        data, body = self.robot.data, self.bat_body
        offset = self.bat.read() - data.body_link_pos_w[:, body]
        return data.body_link_lin_vel_w[:, body] + np.cross(
            data.body_link_ang_vel_w[:, body], offset
        )

    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.bat.read(), axis=1)
        incoming = self.ball.data.root_link_lin_vel_w[:, 0] < 0
        forward = np.clip(self.blade_velocity()[:, 0], 0, 1)
        approach = 5 * np.exp(-((distance / 0.18) ** 2)) * forward * incoming * ~self.hit_seen
        bonus = self.pending * 5 * (1 + np.tanh(self.separation_velocity - 1)) / env.step_dt
        self.scored |= self.pending
        self.pending[:] = False
        return approach + bonus
