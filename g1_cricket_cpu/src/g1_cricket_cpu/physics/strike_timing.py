"""Fixed phase-local timing of the common two-hand swing path."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StrikeTiming:
    power: float
    center: float = 2.0
    half_width: float = 0.75

    def __post_init__(self):
        if not np.isfinite([self.power, self.center, self.half_width]).all():
            raise ValueError('strike timing parameters must be finite')
        if not 0 <= self.power <= 1 or not 0 < self.half_width <= self.center:
            raise ValueError('strike timing requires power in [0, 1] and a positive nonnegative-start window')

    def sample(self, time):
        time = np.asarray(time)
        z = np.clip((time - self.center) / self.half_width, -1, 1)
        remaining = 1 - z ** 2
        phase = time + self.power * self.half_width * z * remaining ** 3
        rate = 1 + self.power * remaining ** 2 * (1 - 7 * z ** 2)
        acceleration = self.power * 6 * z * remaining * (7 * z ** 2 - 3) / self.half_width
        return (phase, rate, acceleration)

class StrikeTimedPath:
    """Compose a fixed timing law with every consumer of the nominal path."""

    def __init__(self, path, timing):
        self.path = path
        self.timing = timing
        self.initial = path.initial
        self.times = path.times

    def spline(self, phase, nu=0):
        phase, rate, acceleration = self.timing.sample(phase)
        if nu == 0:
            return self.path.spline(phase)
        first = self.path.spline(phase, 1)
        if nu == 1:
            return first * rate[..., None]
        if nu == 2:
            return self.path.spline(phase, 2) * rate[..., None] ** 2 + first * acceleration[..., None]
        raise ValueError('strike-timed spline supports derivatives zero through two')

    def derivatives(self, phase, rate, rate_dot):
        phase, timing_rate, timing_acceleration = self.timing.sample(phase)
        return self.path.derivatives(phase, timing_rate * rate, timing_acceleration * np.asarray(rate) ** 2 + timing_rate * rate_dot)

    def pose(self, model, phase):
        return self.path.pose(model, self.timing.sample(phase)[0])
