"""Measured boundary outcomes for reset-randomized, one-bounce G1 batting."""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import numpy as np
from .palm_grip import G1PalmGripCfg
from .scene import BALL_CONTACT_NAMES, CONTACT_SLOTS, CONTACT_WIDTH
BOUNDARY_RADIUS = 55.0
DELIVERY_POOL = ((3.0, 0.0, 1.3, 4.0), (2.6, 0.0, 1.3, 4.0), (3.4, 0.0, 1.3, 4.0), (3.0, -0.08, 1.3, 4.0), (3.0, 0.08, 1.3, 4.0), (3.0, 0.0, 1.2, 3.7), (3.0, 0.0, 1.4, 4.3))

@dataclass
class G1BoundaryBattingCfg(G1PalmGripCfg):

    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        root = tree.getroot()
        ET.SubElement(root.find('sensor'), 'framepos', name='ball_world_position', objtype='body', objname='cricket_ball')
        angles = np.linspace(0, 2 * np.pi, 129)
        points = BOUNDARY_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))
        for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
            ET.SubElement(root.find('worldbody'), 'geom', name=f'boundary_rope_{index}', type='capsule', fromto=f'{start[0]} {start[1]} 0.04 {end[0]} {end[1]} 0.04', size='0.04', rgba='0.96 0.96 0.90 1', contype='0', conaffinity='0')
        tree.write(destination)
        return guards

class ResetVariedDelivery:
    """The pool uses canonical right-hand lines; left-hand feeds are mirrored."""
    pool = DELIVERY_POOL
    start_x = 4.0

    def __init__(self, cfg, env):
        self.ball = env.scene['ball']
        self.line_sign = 1 if env.cfg.handedness == 'right' else -1
        self.settings = np.zeros((env.num_envs, 4))
        self.delivery_indices = np.full(env.num_envs, -1, dtype=int)

    def __call__(self, env, env_ids, delivery_index=None, speed_range=(2.6, 3.4), line_range=(-0.08, 0.08), height_range=(1.2, 1.4), vertical_speed_range=(3.7, 4.3)):
        if delivery_index is None:
            bounds = np.array((speed_range, line_range, height_range, vertical_speed_range))
            settings = env.rng.uniform(bounds[:, 0], bounds[:, 1], size=(len(env_ids), 4))
            self.delivery_indices[env_ids] = -1
        else:
            if not isinstance(delivery_index, int) or not 0 <= delivery_index < len(self.pool):
                raise ValueError('delivery_index must name a declared delivery-pool row')
            settings = np.broadcast_to(self.pool[delivery_index], (len(env_ids), 4)).copy()
            self.delivery_indices[env_ids] = delivery_index
        settings[:, 1] *= self.line_sign
        self.settings[env_ids] = settings
        states = self.ball.data.default_root_state[env_ids].copy()
        states[:, :3] = np.column_stack((np.full(len(env_ids), self.start_x), settings[:, 1:3]))
        states[:, :3] += env.scene.env_origins[env_ids]
        states[:, 3:7] = [1, 0, 0, 0]
        states[:, 7:] = 0
        states[:, 7] = -settings[:, 0]
        states[:, 9] = settings[:, 3]
        self.ball.write_root_state_to_sim(states, env_ids=env_ids)

class BoundaryBattingReward:
    """Score measured center crossings, separately from whole-robot qualification."""

    def sensor_names(self, env):
        return (*BALL_CONTACT_NAMES, 'ball_world_position')

    def __init__(self, cfg, env):
        self.origins = env.scene.env_origins.copy()
        self.valid_hit = np.zeros(env.num_envs, dtype=bool)
        self.separated = np.zeros(env.num_envs, dtype=bool)
        self.separation_velocity_m_s = np.zeros((env.num_envs, 3))
        self.disqualified = np.zeros(env.num_envs, dtype=bool)
        self.pitch_touching = np.zeros(env.num_envs, dtype=bool)
        self.bat_touching = np.zeros(env.num_envs, dtype=bool)
        self.incoming_bounces = np.zeros(env.num_envs, dtype=int)
        self.loaded_hit_count = np.zeros(env.num_envs, dtype=int)
        self.outgoing_ground_contact = np.zeros(env.num_envs, dtype=bool)
        self.maximum_radius_m = np.zeros(env.num_envs)
        self.boundary_runs = np.zeros(env.num_envs, dtype=int)
        self.pending_distance_m = np.zeros(env.num_envs)
        self.pending_runs = np.zeros(env.num_envs)
        self.reset_arrays = (self.valid_hit, self.separated, self.separation_velocity_m_s, self.disqualified, self.pitch_touching, self.bat_touching, self.incoming_bounces, self.loaded_hit_count, self.outgoing_ground_contact, self.maximum_radius_m, self.boundary_runs, self.pending_distance_m, self.pending_runs)
        env.set_substep_observer(self.sensor_names(env), 'cricket_ball', self.observe)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for array in self.reset_arrays:
            array[ids] = 0

    def observe(self, sensors, integrated_velocity, *, invalid_contacts=None):
        width = len(BALL_CONTACT_NAMES) * CONTACT_SLOTS * CONTACT_WIDTH
        contacts = sensors[..., :width].reshape(*sensors.shape[:2], 5, 4, 17)
        if np.any(contacts[..., 0] > CONTACT_SLOTS):
            raise RuntimeError('G1 cricket contact sensor capacity exceeded')
        active = contacts[..., 0] > 0
        loaded = (active & (contacts[..., 1] > 0)).any(axis=-1)
        penetration = (active & (contacts[..., 7] < -0.006)).any(axis=(-1, -2))
        invalid = penetration | loaded[..., 2:].any(axis=-1)
        if invalid_contacts is not None:
            invalid |= invalid_contacts
        positions = sensors[..., width:] - self.origins[:, None, :]
        radii = np.linalg.norm(positions[..., :2], axis=-1)
        for row in range(len(sensors)):
            bat, pitch = (loaded[row, :, 0], loaded[row, :, 1])
            previous_pitch = np.r_[self.pitch_touching[row], pitch[:-1]]
            completed = previous_pitch & ~pitch
            previous_bat = np.r_[self.bat_touching[row], bat[:-1]]
            self.loaded_hit_count[row] += np.count_nonzero(bat & ~previous_bat)
            self.pitch_touching[row], self.bat_touching[row] = (pitch[-1], bat[-1])
            if self.disqualified[row] or self.boundary_runs[row]:
                continue
            stop = int(np.flatnonzero(invalid[row])[0]) if invalid[row].any() else len(bat)
            start = 0
            if not self.valid_hit[row]:
                hits = np.flatnonzero(bat[:stop])
                if not len(hits):
                    self.incoming_bounces[row] += np.count_nonzero(completed[:stop])
                    self.disqualified[row] |= stop < len(bat)
                    continue
                start = int(hits[0])
                self.incoming_bounces[row] += np.count_nonzero(completed[:start])
                if self.incoming_bounces[row] != 1 or pitch[start] or radii[row, start] >= BOUNDARY_RADIUS:
                    self.disqualified[row] = True
                    continue
                self.valid_hit[row] = True
            if not self.separated[row]:
                free = np.flatnonzero(~bat[start:stop])
                if not len(free):
                    self.outgoing_ground_contact[row] |= pitch[start:stop].any()
                    self.disqualified[row] |= stop < len(bat)
                    continue
                separation = start + int(free[0])
                self.outgoing_ground_contact[row] |= pitch[start:separation].any()
                self.separated[row] = True
                self.separation_velocity_m_s[row] = integrated_velocity[row, separation]
                self.maximum_radius_m[row] = radii[row, separation]
                if radii[row, separation] >= BOUNDARY_RADIUS:
                    self.disqualified[row] = True
                    continue
                start = separation
            crossings = np.flatnonzero(radii[row, start:stop] >= BOUNDARY_RADIUS)
            end = start + int(crossings[0]) + 1 if len(crossings) else stop
            if end > start:
                self.outgoing_ground_contact[row] |= pitch[start:end].any()
                maximum = min(float(radii[row, start:end].max()), BOUNDARY_RADIUS)
                self.pending_distance_m[row] += max(0, maximum - self.maximum_radius_m[row])
                self.maximum_radius_m[row] = max(maximum, self.maximum_radius_m[row])
            if len(crossings):
                self.boundary_runs[row] = 4 if self.outgoing_ground_contact[row] else 6
                self.pending_runs[row] = self.boundary_runs[row]
            elif stop < len(bat):
                self.disqualified[row] = True
                self.pending_distance_m[row] = 0
                self.pending_runs[row] = 0

    def __call__(self, env):
        reward = (self.pending_distance_m / BOUNDARY_RADIUS + self.pending_runs) / env.step_dt
        self.pending_distance_m[:] = 0
        self.pending_runs[:] = 0
        return reward
