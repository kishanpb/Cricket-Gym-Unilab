"""Bounded lane feedback for the external locomotion command, not a new policy."""

import numpy as np

from .approach import approach_command


def lane_command(env):
    command = approach_command(env)
    robot = env.scene["robot"]
    error = robot.data.root_link_pos_w[:, 1] - robot.data.default_root_state[:, 1]
    command[:, 1] = np.clip(-error, -0.25, 0.25)
    return command
