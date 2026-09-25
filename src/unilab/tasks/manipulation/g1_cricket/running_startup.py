"""Two-foot held-ball startup targets; native replay must validate their support."""

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation

from .prior import SDK_JOINTS


def place_held_ball(model, data, hand):
    mujoco.mj_kinematics(model, data)
    wrist = data.body(f"{hand}_wrist_yaw_link")
    address = int(model.joint("ball_free").qposadr[0])
    offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
    data.qpos[address : address + 3] = wrist.xpos + wrist.xmat.reshape(3, 3) @ offset
    data.qpos[address + 3 : address + 7] = wrist.xquat


def grounded_start(model, hand):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = -2.1
    data.qpos[1] = 0.7 if hand == "right" else -0.7
    mujoco.mj_forward(model, data)
    gap = min(
        mujoco.mj_geomDistance(
            model,
            data,
            model.geom(f"{side}_foot{i}_collision").id,
            model.geom("pitch").id,
            1,
            None,
        )
        for side in ("left", "right")
        for i in range(1, 8)
    )
    data.qpos[2] -= gap
    place_held_ball(model, data, hand)
    return data.qpos.copy()


def held_support_feedforward(model, pose, hand):
    """Static vertical support, including ball gravity transmitted through its holder."""
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    jacobian = np.empty((3, model.nv))
    ball = model.body("cricket_ball").id
    wrist = model.body(f"{hand}_wrist_yaw_link").id
    mujoco.mj_jac(model, data, jacobian, None, data.xipos[ball], wrist)
    bias = data.qfrc_bias + jacobian.T @ (-model.body_mass[ball] * model.opt.gravity)
    normals = []
    for side in ("left", "right"):
        for index in range(1, 8):
            geom = model.geom(f"{side}_foot{index}_collision")
            axis = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
            for sign in (-1, 1):
                point = data.geom_xpos[geom.id] + sign * geom.size[1] * axis
                mujoco.mj_jac(model, data, jacobian, None, point, int(geom.bodyid[0]))
                normals.append(jacobian[2].copy())
    normals = np.asarray(normals)
    solution = lsq_linear(normals[:, :6].T, bias[:6], bounds=(0, np.inf), tol=1e-12)
    torque = bias - normals.T @ solution.x
    joints = [model.joint(name).id for name in SDK_JOINTS]
    return torque[model.jnt_dofadr[joints]], torque[:6], solution.x.reshape(2, -1).sum(axis=1)


def startup_reference(model, hand, *, shift=0.08):
    """Settle 2 s, transfer COM toward the front foot over 1.5 s, hold 2 s."""
    times = np.arange(276) * 0.02
    initial = grounded_start(model, hand)
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    feet_position = data.xpos[feet].copy()
    feet_rotation = data.xmat[feet].reshape(2, 3, 3).copy()
    center = data.subtree_com[0].copy()
    joints = [model.joint(name).id for name in SDK_JOINTS[:12]]
    qa = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T
    lower = np.r_[lower + 0.03, initial[:3] - [0.2, 0.2, 0.05]]
    upper = np.r_[upper - 0.03, initial[:3] + [0.2, 0.2, 0.05]]
    origin = np.r_[initial[qa], initial[:3]]
    previous = origin.copy()
    poses, errors, torques, loads = [], [], [], []
    for time in times:
        fraction = np.clip((time - 2) / 1.5, 0, 1)
        blend = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        desired = center + [0, (1 if hand == "right" else -1) * shift * blend, 0]

        def residual(values):
            data.qpos[qa], data.qpos[:3] = values[:12], values[12:]
            place_held_ball(model, data, hand)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            return np.r_[
                100 * (data.xpos[feet] - feet_position).ravel(),
                10
                * Rotation.from_matrix(
                    feet_rotation.transpose(0, 2, 1) @ data.xmat[feet].reshape(2, 3, 3)
                )
                .as_rotvec()
                .ravel(),
                30 * (data.subtree_com[0] - desired),
                0.005 * (values - origin),
            ]

        if 2 < time <= 3.5:
            solved = least_squares(
                residual,
                previous,
                bounds=(lower, upper),
                max_nfev=100,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            if not solved.success:
                raise RuntimeError("startup IK did not converge")
            previous = solved.x
        residual(previous)
        pose = data.qpos.copy()
        torque, root_residual, load = held_support_feedforward(model, pose, hand)
        errors.append(
            [
                np.max(np.linalg.norm(data.xpos[feet] - feet_position, axis=1)),
                np.linalg.norm(data.subtree_com[0] - desired),
                np.linalg.norm(root_residual),
            ]
        )
        poses.append(pose)
        torques.append(torque)
        loads.append(load)
    poses = np.asarray(poses)
    velocity = np.zeros((len(times), model.nv))
    for index in range(1, len(times) - 1):
        if 2 < times[index] < 3.5:
            mujoco.mj_differentiatePos(
                model, velocity[index], 0.04, poses[index - 1], poses[index + 1]
            )
    return dict(
        times=times,
        qpos=poses,
        qvel=velocity,
        torque=np.asarray(torques),
        support_loads=np.asarray(loads),
        errors=np.asarray(errors),
    )
