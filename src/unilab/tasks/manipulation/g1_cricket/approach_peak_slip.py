"""Landing-peak objective for the unchanged closed-loop whole-body residual."""

import numpy as np


def peak_slip_cost(env):
    action = env.action_manager.get_term("residual")
    return np.square(action.slip_peak).sum(axis=1)
