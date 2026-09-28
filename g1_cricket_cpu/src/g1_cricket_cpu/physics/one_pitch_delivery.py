"""Versioned batting feeds with a prescribed first pitch location."""
import numpy as np
from .articulated_learning import ARTICULATED_DELIVERY_POOL
from .incoming_swing import ResetIncomingSwingDelivery

def launch_vertical_speed(speed, height, *, distance, surface_height, gravity):
    flight_time = distance / np.asarray(speed)
    return (surface_height - height) / flight_time + 0.5 * gravity * flight_time

class ResetOnePitchDelivery(ResetIncomingSwingDelivery):
    """Set launch velocity once at reset; subsequent flight is unconstrained."""
    version = 'one_pitch_v2'
    bounce_x = 1.5

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        self.gravity = -model.opt.gravity[2]
        self.surface_height = model.geom('pitch').pos[2] + model.geom('ball_geom').size[0]
        self.pool = tuple(((speed, line, height, float(self.vertical_speed(speed, height))) for speed, line, height, _ in ARTICULATED_DELIVERY_POOL))

    def vertical_speed(self, speed, height):
        return launch_vertical_speed(speed, height, distance=self.start_x - self.bounce_x, surface_height=self.surface_height, gravity=self.gravity)
