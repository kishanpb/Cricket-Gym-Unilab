"""Continuous two-hand motion references for the floating, articulated G1."""
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation
from .articulated_batting import ArticulatedStanceController, body_support_torque
from .batting_dynamics import reference_derivatives
from .bimanual import batting_targets
from .prior import SDK_JOINTS
from .tracking import ankle_balance, root_position_balance
SWING_AMPLITUDE = 0.75

def strike_window(times):

    def ramp(start):
        u = np.clip((np.asarray(times) - start) / 0.9, 0, 1)
        u = np.where(np.asarray(times) >= start + 0.9, 1, u)
        return u ** 3 * (10 + u * (-15 + 6 * u))
    return ramp(0.5) - ramp(2.6)

def strike_lift(times, height):
    if not np.isfinite(height) or not -0.08 <= height <= 0.25:
        raise ValueError('strike lift must be finite and between -0.08 and 0.25 m')
    return height * strike_window(times)

def swing_targets(times, hand, *, lift_height=0.0, pitch_bias=0.0, lateral_offset=0.0):
    if not np.isfinite(lateral_offset) or not -0.08 <= lateral_offset <= 0.08:
        raise ValueError('strike lateral offset must be finite and within 0.08 m')
    if not np.isfinite(pitch_bias) or not -np.pi / 2 <= pitch_bias <= 0:
        raise ValueError('strike pitch bias must be finite and between -pi/2 and 0 rad')
    knots = np.array([0, 0.55, 1.15, 1.45, 1.8, 2.45, 3.0])
    positions, rotations = batting_targets(knots, hand)
    angles = Rotation.from_matrix(rotations).as_rotvec()[:, 1]
    phase = np.clip(np.asarray(times) - 0.5, 0, 3)
    position = positions[0] + SWING_AMPLITUDE * (CubicSpline(knots, positions, bc_type='clamped')(phase) - positions[0])
    position[:, 2] += strike_lift(times, lift_height)
    if lateral_offset:
        position[:, 1] += lateral_offset * strike_window(times)
    angle = angles[0] + SWING_AMPLITUDE * (CubicSpline(knots, angles, bc_type='clamped')(phase) - angles[0])
    angle += pitch_bias * strike_window(times)
    rotation = Rotation.from_euler('y', angle[:, None]).as_matrix()
    if hand == 'right':
        rotation = rotation @ Rotation.from_euler('z', np.pi).as_matrix()
    return (position, rotation)

def retarget_swing(model, initial, times, hand, *, lock_root_height=False, acceleration_weight=0.0, lift_height=0.0, pitch_bias=0.0, lateral_offset=0.0):
    """Offline motor reference; qpos placement here is never a physics rollout."""
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    q = model.jnt_qposadr[joints]
    feet = [model.body(f'{side}_ankle_roll_link').id for side in ('left', 'right')]
    wrists = [model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
    bat = model.body('cricket_bat').id
    bat_q = model.joint('bat_free').qposadr[0]
    pelvis = model.body('pelvis').id
    foot_positions = data.xpos[feet].copy()
    foot_rotations = data.xmat[feet].reshape(2, 3, 3).copy()
    bat_rotation = data.xmat[bat].reshape(3, 3)
    local_positions = (data.xpos[wrists] - data.xpos[bat]) @ bat_rotation
    local_rotations = bat_rotation.T @ data.xmat[wrists].reshape(2, 3, 3)
    masses = np.array([model.body_subtreemass[pelvis], model.body_mass[bat]])
    support_center = (masses[0] * data.subtree_com[pelvis] + masses[1] * data.xipos[bat])[:2] / masses.sum()
    positions, rotations = swing_targets(times, hand, lift_height=lift_height, pitch_bias=pitch_bias, lateral_offset=lateral_offset)
    lower = np.r_[model.jnt_range[joints, 0] + 0.04, initial[:3] - [0.15, 0.1, 0.08]]
    upper = np.r_[model.jnt_range[joints, 1] - 0.04, initial[:3] + [0.15, 0.1, 0.05]]
    lower[[3, 9]] = np.maximum(lower[[3, 9]], 0.25)
    neutral = np.r_[initial[q], initial[:3]]
    if lock_root_height:
        lower, upper, neutral = (lower[:-1], upper[:-1], neutral[:-1])
    previous = neutral.copy()
    previous_velocity = np.zeros_like(previous)
    acceleration_scale = np.r_[np.ones(29), np.full(len(previous) - 29, 10.0)]
    poses, rows = ([], [])
    for index, (position, rotation) in enumerate(zip(positions, rotations, strict=True)):
        dt = (times[index] - times[index - 1] if index else times[1] - times[0]) if acceleration_weight else 1.0
        target_position = position + local_positions @ rotation.T
        target_rotation = rotation @ local_rotations
        data.qpos[:] = initial
        data.qpos[bat_q:bat_q + 3] = position
        mujoco.mju_mat2Quat(data.qpos[bat_q + 3:bat_q + 7], rotation.ravel())

        def residual(value):
            data.qpos[q] = value[:29]
            data.qpos[:3] = np.r_[value[29:], initial[2]] if lock_root_height else value[29:]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            com = (masses[0] * data.subtree_com[pelvis] + masses[1] * data.xipos[bat]) / masses.sum()
            error = np.r_[50 * (data.xpos[wrists] - target_position).ravel(), 5 * Rotation.from_matrix(target_rotation.transpose(0, 2, 1) @ data.xmat[wrists].reshape(2, 3, 3)).as_rotvec().ravel(), 50 * (data.xpos[feet] - foot_positions).ravel(), 5 * Rotation.from_matrix(foot_rotations.transpose(0, 2, 1) @ data.xmat[feet].reshape(2, 3, 3)).as_rotvec().ravel(), 15 * (com[:2] - support_center), 0.025 * (value - previous), 0.005 * (value - neutral)]
            if acceleration_weight:
                return np.r_[error, acceleration_weight * acceleration_scale * ((value - previous) / dt - previous_velocity) / dt]
            return error
        unchanged = index == 0 or (np.array_equal(position, positions[index - 1]) and np.array_equal(rotation, rotations[index - 1]))
        if unchanged:
            solved_success = True
            value = previous
        else:
            step = np.r_[np.full(29, 0.12), np.full(len(previous) - 29, 0.005)]
            solved = least_squares(residual, previous, bounds=(np.maximum(lower, previous - step), np.minimum(upper, previous + step)), max_nfev=150, ftol=1e-09, xtol=1e-09, gtol=1e-09)
            value, solved_success = (solved.x, bool(solved.success))
        error = residual(value)
        mujoco.mj_forward(model, data)
        rows.append({'time_s': float(times[index]), 'optimizer_success': solved_success, 'max_wrist_position_error_m': float(np.linalg.norm(error[:6].reshape(2, 3), axis=1).max() / 50), 'max_wrist_orientation_error_rad': float(np.linalg.norm(error[6:12].reshape(2, 3), axis=1).max() / 5), 'max_foot_position_error_m': float(np.linalg.norm(error[12:18].reshape(2, 3), axis=1).max() / 50), 'max_foot_orientation_error_rad': float(np.linalg.norm(error[18:24].reshape(2, 3), axis=1).max() / 5), 'support_com_error_m': float(np.linalg.norm(error[24:26]) / 15), 'max_joint_frame_change_rad': float(np.max(np.abs(value[:29] - previous[:29])))})
        previous_velocity = (value - previous) / dt
        previous = value.copy()
        poses.append(data.qpos.copy())
    return (np.asarray(poses), rows)

def transfer_bat_wrench(model, data, required):
    """Split the free-bat wrench between wrists, preserving its origin and frame."""
    result = required.copy()
    bat = model.body('cricket_bat').id
    bat_dof = model.joint('bat_free').dofadr[0]
    rotation = data.xmat[bat].reshape(3, 3)
    force = required[bat_dof:bat_dof + 3]
    torque = rotation @ required[bat_dof + 3:bat_dof + 6]
    for side in ('left', 'right'):
        linear, angular = (np.empty((3, model.nv)), np.empty((3, model.nv)))
        mujoco.mj_jac(model, data, linear, angular, data.xpos[bat], model.body(f'{side}_wrist_yaw_link').id)
        result += (linear.T @ force + angular.T @ torque) / 2
    return result

def support_motion_increment(model, position, velocity, acceleration):
    """Motor feedforward with free-bat wrench transfer and frictional foot support."""
    data = mujoco.MjData(model)
    data.qpos[:] = position
    mujoco.mj_forward(model, data)
    static = transfer_bat_wrench(model, data, data.qfrc_bias - data.qfrc_passive)
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    dynamic = transfer_bat_wrench(model, data, mass @ acceleration + data.qfrc_bias - data.qfrc_passive)
    basis = []
    pitch = model.geom('pitch').id
    for side in ('left', 'right'):
        for i in range(1, 8):
            geom = model.geom(f'{side}_foot{i}_collision')
            friction = min(model.geom_friction[geom.id, 0], model.geom_friction[pitch, 0])
            rays = np.array([[friction, 0, 1], [-friction, 0, 1], [0, friction, 1], [0, -friction, 1]])
            axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
            for sign in (-1, 1):
                point = data.geom_xpos[geom.id] + sign * geom.size[1] * axis
                point[2] = 0
                jacobian = np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jacobian, None, point, int(geom.bodyid[0]))
                basis.extend(rays @ jacobian)
    basis = np.asarray(basis)
    residuals, torques = ([], [])
    for required in (static, dynamic):
        loads = lsq_linear(basis[:, :6].T, required[:6], bounds=(0, np.inf), tol=1e-10, max_iter=200)
        remaining = required - basis.T @ loads.x
        residuals.append(float(np.linalg.norm(remaining[:6])))
        torques.append(remaining)
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    return ((torques[1] - torques[0])[model.jnt_dofadr[joints]], max(residuals))

class ArticulatedSwingController(ArticulatedStanceController):

    def __init__(self, model, poses, times, *, inertial=False, preload=0.2):
        super().__init__(model, poses[0], preload=preload)
        self.poses, self.times = (poses.copy(), times.copy())
        self.gravity = np.asarray([body_support_torque(model, p)[0] for p in poses])
        self.motion_increment = np.zeros_like(self.gravity)
        self.motion_support_residual = np.zeros(len(poses))
        if inertial:
            velocity, acceleration, _ = reference_derivatives(model, poses, times)
            for index, pose in enumerate(poses):
                self.motion_increment[index], self.motion_support_residual[index] = support_motion_increment(model, pose, velocity[index], acceleration[index])
        self.forces = self.gravity + self.motion_increment
        self.gain = model.actuator_gainprm[self.actuators, 0]
        self.velocity_gain = -model.actuator_biasprm[self.actuators, 2] / self.gain

    def apply(self, data):
        index = int(np.clip(np.searchsorted(self.times, data.time, side='right') - 1, 0, len(self.times) - 2))
        duration = self.times[index + 1] - self.times[index]
        blend = np.clip((data.time - self.times[index]) / duration, 0, 1)
        before, after = (self.poses[index], self.poses[index + 1])
        reference = (1 - blend) * before[self.q] + blend * after[self.q]
        velocity = (after[self.q] - before[self.q]) / duration
        root_position = (1 - blend) * before[:3] + blend * after[:3]
        root_velocity = (after[:3] - before[:3]) / duration
        root_rotation = self.poses[0, 3:7]
        force = (1 - blend) * self.forces[index] + blend * self.forces[index + 1]
        target = reference + self.velocity_gain * velocity + force / self.gain
        correction = ankle_balance(root_rotation, data.qpos[3:7], data.qvel[3:6], 4.0)
        correction += root_position_balance(root_rotation, data.qpos[:3] - root_position, data.qvel[:3] - root_velocity, 2.0)
        correction = np.clip(correction, -0.3, 0.3)
        target[[4, 10]] += correction[1]
        target[[5, 11]] += correction[0]
        data.ctrl[self.actuators] = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        self.fingers.apply(data, self.finger_target)
