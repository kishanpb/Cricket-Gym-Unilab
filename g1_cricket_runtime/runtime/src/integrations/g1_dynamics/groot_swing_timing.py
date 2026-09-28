"""Launch-state timing heuristic for a shared swing, not a learned policy."""

import numpy as np
from scipy.optimize import brentq

from unilab.tasks.manipulation.g1_cricket.one_pitch_delivery import (
    ResetOnePitchDelivery,
)
from unilab.tasks.manipulation.g1_cricket.strike_timing import StrikeTiming

REFERENCE_SPEED_M_S = 3.0
STRIKE_X_M = 0.0
MAX_DELAY_S = 0.2


def retime_swing_knots(times, power, delay):
    """Preserve every shared pose; invert the existing tempo law at its knots."""
    timing = StrikeTiming(power, center=2.0 + delay)
    result = times.copy()
    if power == 0:
        return result
    start = timing.center - timing.half_width
    end = timing.center + timing.half_width
    for index in np.flatnonzero((times > start) & (times < end)):
        result[index] = brentq(
            lambda time: timing.sample(time)[0] - times[index],
            start,
            end,
            xtol=1e-14,
        )
    return result


def arrival_timing(model, data, nominal_delay):
    ball = model.joint("ball_free")
    position = float(data.qpos[int(ball.qposadr[0])])
    velocity = float(data.qvel[int(ball.dofadr[0])])
    distance = position - STRIKE_X_M
    if (
        not np.isfinite([position, velocity, nominal_delay]).all()
        or distance <= 0
        or velocity >= 0
        or not 0 <= nominal_delay <= MAX_DELAY_S
    ):
        raise ValueError(
            "Arrival timing requires an approaching finite launch and a bounded nominal delay"
        )
    arrival = distance / -velocity
    reference_arrival = (
        ResetOnePitchDelivery.start_x - STRIKE_X_M
    ) / REFERENCE_SPEED_M_S
    raw_delay = nominal_delay + (arrival - reference_arrival)
    delay = float(np.clip(raw_delay, 0, MAX_DELAY_S))
    return {
        "mode": "launch_velocity",
        "launch_x_m": position,
        "launch_vx_m_s": velocity,
        "strike_x_m": STRIKE_X_M,
        "reference_speed_m_s": REFERENCE_SPEED_M_S,
        "horizontal_arrival_estimate_s": arrival,
        "reference_arrival_estimate_s": reference_arrival,
        "nominal_delay_s": nominal_delay,
        "unclipped_delay_s": raw_delay,
        "delay_s": delay,
        "clipped": bool(raw_delay < 0 or raw_delay > MAX_DELAY_S),
        "scope": "Reset-state horizontal estimate; ignores bounce friction and vertical geometry. No feed ID, future outcome or learned timing policy.",
    }
