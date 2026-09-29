"""Trigger the bowling-arm sweep from recent, loaded, legal front-foot support."""
import numpy as np
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import feet_failures

def validate_cocked_pause(reference):
    times = np.linspace(1.5, 1.7, 41)
    values = reference(times)
    if not (np.allclose(values, values[0], rtol=0, atol=1e-10) and np.allclose(reference(times, nu=1), 0, rtol=0, atol=1e-10)):
        raise ValueError('Support-triggered sweep requires a stationary 1.5-1.7 s bowling-arm pause')

class SupportSwingClock:

    def __init__(self):
        self.started_at = None

    def can_release(self, time, events, *, enforce_footwork=True):
        if self.started_at is None:
            return False
        if not enforce_footwork:
            return True
        front = events.landings[events.front][-1] if events.landings[events.front] else None
        back = events.landings[events.hand][-1] if events.landings[events.hand] else None
        return not feet_failures(front, back, events.side, time)

    def advance(self, time, local, events, front_load):
        if self.started_at is not None:
            return (1.7 + time - self.started_at, 1.0)
        if local < 1.5:
            return (local, 1.0)
        front = events.landings[events.front][-1] if events.landings[events.front] else None
        back = events.landings[events.hand][-1] if events.landings[events.hand] else None
        if front_load > 1 and front is not None and (0 <= time - front['time'] <= 0.1) and (not feet_failures(front, back, events.side, time)):
            self.started_at = time
            return (1.7, 1.0)
        return (1.5, 0.0)
