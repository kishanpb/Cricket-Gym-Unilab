"""Offline two-hand stance fitting and native control for a free cricket bat."""
import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation
from .articulated_hands import FingerController
from .prior import SDK_DEFAULT, SDK_JOINTS
from .tracking import ankle_balance, root_position_balance

def grip_frames(solution, hand):
    x, y = solution[7:9]
    heights = (-0.04, -0.15) if hand == 'right' else (-0.15, -0.04)
    positions = np.array([[-x, -y, heights[0]], [x, -y, heights[1]]])
    rotations = Rotation.from_euler('z', [[0], [np.pi]]).as_matrix()
    return (positions, rotations)

def fit_stance(model, hand, solution, *, bat_position=(0.03, 0, 0.9), lateral_offset=0.28):
    """Fit reset geometry, not a simulated or learned trajectory."""
    if hand not in {'right', 'left'}:
        raise ValueError('hand must be right or left')
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    sign = 1 if hand == 'right' else -1
    data.qpos[:3] = [0, sign * lateral_offset, 0.79]
    data.qpos[3:7] = [np.sqrt(0.5), 0, 0, -sign * np.sqrt(0.5)]
    fingers = FingerController(model)
    data.qpos[fingers.q] = np.r_[solution[:7], -np.asarray(solution[:7])]
    bat_q = model.joint('bat_free').qposadr[0]
    rotation = Rotation.from_euler('y', 0.1).as_matrix() @ Rotation.from_euler('z', np.pi if hand == 'right' else 0).as_matrix()
    data.qpos[bat_q:bat_q + 3] = bat_position
    mujoco.mju_mat2Quat(data.qpos[bat_q + 3:bat_q + 7], rotation.ravel())
    mujoco.mj_forward(model, data)
    data.qpos[2] -= min((mujoco.mj_geomDistance(model, data, model.geom(f'{side}_foot{i}_collision').id, model.geom('pitch').id, 1, None) for side in ('left', 'right') for i in range(1, 8)))
    mujoco.mj_forward(model, data)
    feet = [model.body(f'{side}_ankle_roll_link').id for side in ('left', 'right')]
    wrists = [model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
    feet_position = data.xpos[feet].copy()
    feet_rotation = data.xmat[feet].reshape(2, 3, 3).copy()
    support_center = np.mean([data.geom(f'{side}_foot4_collision').xpos[:2] for side in ('left', 'right')], axis=0)
    local_position, local_rotation = grip_frames(solution, hand)
    wrist_position = np.asarray(bat_position) + local_position @ rotation.T
    wrist_rotation = rotation @ local_rotation
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    q = model.jnt_qposadr[joints]
    initial = np.r_[data.qpos[q], data.qpos[:3]]
    lower = np.r_[model.jnt_range[joints, 0] + 0.04, [-0.1, sign * lateral_offset - 0.1, 0.66]]
    upper = np.r_[model.jnt_range[joints, 1] - 0.04, [0.1, sign * lateral_offset + 0.1, 0.84]]
    lower[[3, 9]] = 0.25
    pelvis, bat = (model.body('pelvis').id, model.body('cricket_bat').id)
    masses = np.array([model.body_subtreemass[pelvis], model.body_mass[bat]])

    def residual(value):
        data.qpos[q], data.qpos[:3] = (value[:29], value[29:])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com = (masses[0] * data.subtree_com[pelvis] + masses[1] * data.xipos[bat]) / masses.sum()
        return np.r_[30 * (data.xpos[wrists] - wrist_position).ravel(), 3 * Rotation.from_matrix(wrist_rotation.transpose(0, 2, 1) @ data.xmat[wrists].reshape(2, 3, 3)).as_rotvec().ravel(), 30 * (data.xpos[feet] - feet_position).ravel(), 3 * Rotation.from_matrix(feet_rotation.transpose(0, 2, 1) @ data.xmat[feet].reshape(2, 3, 3)).as_rotvec().ravel(), 15 * (com[:2] - support_center), 0.015 * (value - initial)]
    solved = least_squares(residual, np.clip(initial, lower, upper), bounds=(lower, upper), max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    error = residual(solved.x)
    mujoco.mj_forward(model, data)
    report = {'optimizer_success': bool(solved.success), 'max_wrist_position_error_m': float(np.linalg.norm(error[:6].reshape(2, 3), axis=1).max() / 30), 'max_wrist_rotation_error_rad': float(np.linalg.norm(error[6:12].reshape(2, 3), axis=1).max() / 3), 'max_foot_position_error_m': float(np.linalg.norm(error[12:18].reshape(2, 3), axis=1).max() / 30), 'max_foot_rotation_error_rad': float(np.linalg.norm(error[18:24].reshape(2, 3), axis=1).max() / 3), 'combined_com_support_error_m': float(np.linalg.norm(error[24:26]) / 15), 'minimum_joint_margin_rad': float(np.minimum(data.qpos[q] - model.jnt_range[joints, 0], model.jnt_range[joints, 1] - data.qpos[q]).min())}
    return (data.qpos.copy(), report)

def body_support_torque(model, pose):
    """Static feedforward through motor commands, including the held bat load."""
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    required = data.qfrc_bias.copy()
    bat = model.body('cricket_bat').id
    bat_weight = -model.body_mass[bat] * model.opt.gravity
    for side in ('left', 'right'):
        jacobian = np.empty((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, data.xipos[bat], model.body(f'{side}_wrist_yaw_link').id)
        required += jacobian.T @ (bat_weight / 2)
    normals = []
    for side in ('left', 'right'):
        for i in range(1, 8):
            geom = model.geom(f'{side}_foot{i}_collision')
            axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
            for sign in (-1, 1):
                point = data.geom_xpos[geom.id] + sign * geom.size[1] * axis
                jacobian = np.empty((3, model.nv))
                mujoco.mj_jac(model, data, jacobian, None, point, int(geom.bodyid[0]))
                normals.append(jacobian[2])
    jacobian = np.asarray(normals)
    support = lsq_linear(jacobian[:, :6].T, required[:6], bounds=(0, np.inf), tol=1e-12)
    torque = required - jacobian.T @ support.x
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    return (torque[model.jnt_dofadr[joints]], float(np.linalg.norm(torque[:6])))

def fit_neutral_leg_guard(model, reference):
    """Refit arms and bat height around the leg policy's neutral lower body."""
    data = mujoco.MjData(model)
    data.qpos[:] = reference
    mujoco.mj_forward(model, data)
    bat = model.body('cricket_bat').id
    wrists = [model.body(f'{side}_wrist_yaw_link').id for side in ('left', 'right')]
    rotation = data.xmat[bat].reshape(3, 3).copy()
    relative_position = (data.xpos[wrists] - data.xpos[bat]) @ rotation
    relative_rotation = rotation.T @ data.xmat[wrists].reshape(2, 3, 3)
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    q = model.jnt_qposadr[joints]
    data.qpos[q[:15]] = SDK_DEFAULT[:15]
    data.qpos[2] = 0.8
    mujoco.mj_forward(model, data)
    data.qpos[2] -= min((mujoco.mj_geomDistance(model, data, model.geom(f'{side}_foot{i}_collision').id, model.geom('pitch').id, 1, None) for side in ('left', 'right') for i in range(1, 8)))
    bat_q = model.joint('bat_free').qposadr[0]
    bat_initial = reference[bat_q:bat_q + 3].copy()
    initial = np.r_[data.qpos[q[15:]], bat_initial + [0, 0, 0.15]]
    lower = np.r_[model.jnt_range[joints[15:], 0] + 0.04, bat_initial + [-0.08, -0.08, 0.0]]
    upper = np.r_[model.jnt_range[joints[15:], 1] - 0.04, bat_initial + [0.08, 0.08, 0.35]]

    def residual(value):
        data.qpos[q[15:]] = value[:14]
        data.qpos[bat_q:bat_q + 3] = value[14:]
        mujoco.mj_kinematics(model, data)
        desired_position = value[14:] + relative_position @ rotation.T
        desired_rotation = rotation @ relative_rotation
        return np.r_[30 * (data.xpos[wrists] - desired_position).ravel(), 3 * Rotation.from_matrix(desired_rotation.transpose(0, 2, 1) @ data.xmat[wrists].reshape(2, 3, 3)).as_rotvec().ravel(), 0.005 * (value - initial)]
    solved = least_squares(residual, np.clip(initial, lower, upper), bounds=(lower, upper), max_nfev=500, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    error = residual(solved.x)
    return (data.qpos.copy(), {'optimizer_success': bool(solved.success), 'max_wrist_position_error_m': float(np.linalg.norm(error[:6].reshape(2, 3), axis=1).max() / 30), 'max_wrist_rotation_error_rad': float(np.linalg.norm(error[6:12].reshape(2, 3), axis=1).max() / 3), 'bat_translation_from_reference_m': (solved.x[14:] - bat_initial).tolist(), 'root_height_m': float(data.qpos[2]), 'scope': 'offline reset fit; fixed native leg/waist defaults, no runtime constraint'})

class ArticulatedStanceController:

    def __init__(self, model, reference, *, preload=0.2):
        self.model = model
        self.reference = reference.copy()
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        self.q = model.jnt_qposadr[joints]
        self.actuators = np.array([model.actuator(name).id for name in SDK_JOINTS])
        self.limits = model.jnt_range[joints]
        torque, self.support_residual = body_support_torque(model, reference)
        self.offset = torque / model.actuator_gainprm[self.actuators, 0]
        self.fingers = FingerController(model)
        delta = preload * np.array([0, 1, 1, -1, -1, -1, -1])
        self.finger_target = reference[self.fingers.q] + np.r_[delta, -delta]

    def apply(self, data):
        target = self.reference[self.q] + self.offset
        correction = ankle_balance(self.reference[3:7], data.qpos[3:7], data.qvel[3:6], 4.0)
        correction += root_position_balance(self.reference[3:7], data.qpos[:3] - self.reference[:3], data.qvel[:3], 2.0)
        correction = np.clip(correction, -0.3, 0.3)
        target[[4, 10]] += correction[1]
        target[[5, 11]] += correction[0]
        data.ctrl[self.actuators] = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        self.fingers.apply(data, self.finger_target)
