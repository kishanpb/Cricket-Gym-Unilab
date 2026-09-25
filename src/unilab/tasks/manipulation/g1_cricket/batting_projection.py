"""Offline grip/foot closure without moving the declared bat trajectory."""

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .prior import SDK_JOINTS


def rotation_log_jacobian(vector):
    x, y, z = vector
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.linalg.norm(vector)
    coefficient = (
        1 / 12 + angle**2 / 720
        if angle < 1e-4
        else (1 - angle / (2 * np.tan(angle / 2))) / angle**2
    )
    return np.eye(3) - cross / 2 + coefficient * cross @ cross


class BattingProjection:
    def __init__(self, model, initial, hand):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        self.q = np.r_[model.jnt_qposadr[joints], [0, 1, 2]]
        self.v = np.r_[model.jnt_dofadr[joints], [0, 1, 2]]
        self.joint_limits = model.jnt_range[joints].copy()
        self.feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.bat = model.site("bat_center").id
        self.grip = model.site("bat_lower_grip").id
        self.palm = model.site(f"{hand}_palm").id
        self.set_pose(initial)
        self.foot_positions = self.data.xpos[self.feet].copy()
        self.foot_rotations = self.data.xmat[self.feet].reshape(2, 3, 3).copy()

    def set_pose(self, pose):
        self.data.qpos[:] = pose
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def constraint(self, variables, reference):
        self.set_pose(reference)
        target_position = self.data.site_xpos[self.bat].copy()
        target_rotation = self.data.site_xmat[self.bat].reshape(3, 3).copy()
        self.data.qpos[self.q] = variables
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        residual, jacobian = [], []
        for kind, index, position, rotation in [
            ("site", self.bat, target_position, target_rotation),
            *[
                ("body", foot, pos, rot)
                for foot, pos, rot in zip(
                    self.feet, self.foot_positions, self.foot_rotations, strict=True
                )
            ],
        ]:
            linear, angular = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
            if kind == "site":
                mujoco.mj_jacSite(self.model, self.data, linear, angular, index)
                actual_position = self.data.site_xpos[index]
                actual_rotation = self.data.site_xmat[index].reshape(3, 3)
            else:
                mujoco.mj_jacBody(self.model, self.data, linear, angular, index)
                actual_position = self.data.xpos[index]
                actual_rotation = self.data.xmat[index].reshape(3, 3)
            error = Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec()
            residual.extend((actual_position - position, error))
            jacobian.extend(
                (linear[:, self.v], rotation_log_jacobian(error) @ rotation.T @ angular[:, self.v])
            )
        grip_jac, palm_jac = np.empty((3, self.model.nv)), np.empty((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, grip_jac, None, self.grip)
        mujoco.mj_jacSite(self.model, self.data, palm_jac, None, self.palm)
        residual.append(self.data.site_xpos[self.grip] - self.data.site_xpos[self.palm])
        jacobian.append((grip_jac - palm_jac)[:, self.v])
        return np.concatenate(residual), np.vstack(jacobian)

    def project(self, reference):
        initial = reference[self.q].copy()
        weights = np.r_[np.ones(29), np.full(3, 4.0)]
        bounds = np.vstack(
            (self.joint_limits, np.column_stack((initial[-3:] - 0.05, initial[-3:] + 0.05)))
        )
        result = minimize(
            lambda value: 0.5 * np.sum(((value - initial) * weights) ** 2),
            initial,
            jac=lambda value: (value - initial) * weights**2,
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": lambda value: self.constraint(value, reference)[0],
                "jac": lambda value: self.constraint(value, reference)[1],
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 200},
        )
        error, _ = self.constraint(result.x, reference)
        pose = reference.copy()
        pose[self.q] = result.x
        return pose, {
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "iterations": int(result.nit),
            "max_constraint_error": float(np.max(np.abs(error))),
            "bat_position_error_m": float(np.linalg.norm(error[:3])),
            "bat_orientation_error_rad": float(np.linalg.norm(error[3:6])),
            "foot_position_error_m": float(
                max(np.linalg.norm(error[6:9]), np.linalg.norm(error[12:15]))
            ),
            "foot_orientation_error_rad": float(
                max(np.linalg.norm(error[9:12]), np.linalg.norm(error[15:18]))
            ),
            "grip_error_m": float(np.linalg.norm(error[18:21])),
            "maximum_joint_change_rad": float(np.max(np.abs(result.x[:29] - initial[:29]))),
            "root_translation_change_m": float(np.linalg.norm(result.x[-3:] - initial[-3:])),
        }
