"""Completed-state grip telemetry around native bat/ball impact."""
from collections import deque
import mujoco
import numpy as np
from g1_cricket_cpu.dynamics.impact_schedule import blade_gap
from g1_cricket_cpu.physics.articulated_hands import hand_contact_state
GROUPS = [f'{side}_{part}' for side in ('left', 'right') for part in ('palm', 'thumb', 'index', 'middle')]

def aggregate_contacts(contacts):
    normal, slip, count = (np.zeros(8), np.zeros(8), np.zeros(8, dtype=int))
    force = np.zeros((8, 3))
    for contact in contacts:
        if contact['normal_force_n'] <= 0.1:
            continue
        name = contact['geom']
        side = name.split('_', 1)[0]
        part = next((p for p in ('thumb', 'index', 'middle') if f'_hand_{p}_' in name), 'palm')
        group = GROUPS.index(f'{side}_{part}')
        normal[group] += contact['normal_force_n']
        slip[group] = max(slip[group], contact['tangential_slip_m_s'])
        count[group] += 1
        sign = 1 if contact['geom_order'][1] == name else -1
        force[group] += sign * np.asarray(contact['frame']).T @ np.asarray(contact['wrench_contact_frame'][:3])
    return (normal, force, slip, count)

class ImpactGripRecorder:

    def __init__(self, model, initial, fingers):
        self.model, self.fingers = (model, fingers)
        self.wrists = [model.body(f'{s}_wrist_yaw_link').id for s in ('left', 'right')]
        self.bat = model.body('cricket_bat').id
        self.blade = model.geom('bat_blade').id
        self.ball = model.geom('ball_geom').id
        geometry = mujoco.MjData(model)
        geometry.qpos[:] = initial
        mujoco.mj_kinematics(model, geometry)
        self.grip = (geometry.xpos[self.wrists] - geometry.xpos[self.bat]) @ geometry.xmat[self.bat].reshape(3, 3)
        self.pending = deque()
        self.rows = []
        self.first_contact = None

    def __call__(self, data, ball_force):
        time = float(data.time)
        if self.first_contact is not None and time > self.first_contact + 0.15 + 1e-10:
            return
        while self.pending and self.pending[0]['time_s'] < time - 0.05:
            self.pending.popleft()
        if self.first_contact is None and ball_force <= 0 and (blade_gap(self.model, data) > 0.2):
            return
        if self.first_contact is None and ball_force > 0:
            self.first_contact = time
            self.rows.extend(self.pending)
            self.pending.clear()
        normal, force, slip, count = aggregate_contacts(hand_contact_state(self.model, data))
        rotation = data.xmat[self.bat].reshape(3, 3)
        relative = (data.xpos[self.wrists] - data.xpos[self.bat]) @ rotation
        row = {'time_s': time, 'dt_s': float(self.model.opt.timestep), 'integrator_code': int(self.model.opt.integrator), 'ball_normal_force_n': float(ball_force), 'ball_center_blade_frame_m': (data.geom_xpos[self.ball] - data.geom_xpos[self.blade]) @ data.geom_xmat[self.blade].reshape(3, 3), 'wrist_error_bat_frame_m': relative - self.grip, 'finger_position_rad': data.qpos[self.fingers.q].copy(), 'finger_velocity_rad_s': data.qvel[self.fingers.v].copy(), 'finger_target_rad': data.ctrl[self.fingers.actuators].copy(), 'finger_actuator_force_nm': data.actuator_force[self.fingers.actuators].copy(), 'contact_normal_force_n': normal, 'contact_force_on_hand_world_n': force, 'contact_max_slip_m_s': slip, 'loaded_contact_count': count}
        if self.first_contact is None:
            self.pending.append(row)
        else:
            self.rows.append(row)

    def arrays(self):
        return {'impact_grip_' + name: np.asarray([row[name] for row in self.rows]) for name in self.rows[0]} if self.rows else {}

    def report(self):
        return {'scope': 'Completed-state native telemetry, not RK4 stage averages or calibrated real-robot forces; no controller or acceptance changes', 'first_loaded_bat_contact_s': self.first_contact, 'pre_contact_window_s': 0.05, 'post_contact_window_s': 0.15, 'pre_contact_capture_requires_blade_gap_below_m': 0.2, 'samples': len(self.rows), 'window_start_s': self.rows[0]['time_s'] if self.rows else None, 'window_end_s': self.rows[-1]['time_s'] if self.rows else None, 'contact_groups': GROUPS, 'blade_half_sizes_m': self.model.geom_size[self.blade].tolist(), 'loaded_contact_threshold_n': 0.1, 'wrist_order': ['left', 'right'], 'finger_joint_names': [self.model.joint(int(j)).name for j in self.fingers.joints], 'finger_force_limits_nm': self.fingers.force_limits.tolist()}
