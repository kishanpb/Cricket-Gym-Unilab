"""Offline run-up momentum target from a planted-foot ground wrench."""

import numpy as np


class RunningGroundMomentum:
    def __init__(self, lateral_com, mass, mean_momentum, *, constant=False):
        parent = lateral_com.parent
        self.force_z = mass * (parent.gravity + float(parent.stance(0, 2)))
        self.velocity_x = float(parent.horizontal_velocity[0])
        self.initial = np.asarray(mean_momentum).copy()
        self.constant = constant
        # Preserve the parent's cycle-average momentum, not its inconsistent torque.
        mean_impulse = -self.force_z * self.velocity_x * 0.22**3 / (12 * 0.3)
        if not constant:
            self.initial[1] -= mean_impulse

    def __call__(self, time):
        phase = min(max(0.0, time - np.floor((time + 1e-12) / 0.3) * 0.3), 0.22)
        result = self.initial.copy()
        if not self.constant:
            result[1] += 0.5 * self.force_z * self.velocity_x * phase * (phase - 0.22)
        return result
