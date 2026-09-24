"""Offline held-ball centroidal momentum and implicit-midpoint root rotation."""

import mujoco
import numpy as np

from .prior import SDK_JOINTS


class HeldBallMomentum:
    def __init__(self, model, hand):
        self.model = model
        self.data = mujoco.MjData(model)
        joints = [model.joint(name).id for name in SDK_JOINTS]
        self.qa, self.va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.wrist = model.body(f"{hand}_wrist_yaw_link").id
        self.ball_q = int(model.joint("ball_free").qposadr[0])
        self.ball_v = int(model.joint("ball_free").dofadr[0])
        self.offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
        self.mapping = np.zeros((model.nv, 32))
        self.mapping[3:6, :3] = np.eye(3)
        self.mapping[self.va, 3:] = np.eye(29)
        self.angular = np.empty((3, model.nv))
        self.linear_jacobian = np.empty((3, model.nv))
        self.angular_jacobian = np.empty((3, model.nv))

    def matrix(self, pose):
        """Map root-local angular and joint rates to world COM angular momentum."""
        model, data = self.model, self.data
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        rotation = data.xmat[self.wrist].reshape(3, 3).copy()
        held = data.xpos[self.wrist] + rotation @ self.offset
        data.qpos[self.ball_q : self.ball_q + 3] = held
        data.qpos[self.ball_q + 3 : self.ball_q + 7] = data.xquat[self.wrist]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jac(model, data, self.linear_jacobian, self.angular_jacobian, held, self.wrist)
        self.mapping[self.ball_v : self.ball_v + 6] = 0
        self.mapping[self.ball_v : self.ball_v + 3] = self.linear_jacobian @ self.mapping
        self.mapping[self.ball_v + 3 : self.ball_v + 6] = (
            rotation.T @ self.angular_jacobian @ self.mapping
        )
        mujoco.mj_angmomMat(model, data, self.angular, 0)
        return self.angular @ self.mapping

    def measure(self, before, after, dt):
        velocity = np.empty(self.model.nv)
        mujoco.mj_differentiatePos(self.model, velocity, dt, before, after)
        midpoint = before.copy()
        mujoco.mj_integratePos(self.model, midpoint, velocity, dt / 2)
        return self.matrix(midpoint) @ np.r_[velocity[3:6], velocity[self.va]]

    def advance(self, before, joints, momentum, dt):
        """Solve root rotation for prescribed joint motion; never step physics."""
        joint_velocity = (joints - before[self.qa]) / dt
        midpoint = before.copy()
        midpoint[self.qa] = (before[self.qa] + joints) / 2
        omega = np.zeros(3)
        for _ in range(12):
            midpoint[3:7] = before[3:7]
            mujoco.mju_quatIntegrate(midpoint[3:7], omega, dt / 2)
            matrix = self.matrix(midpoint)
            updated = np.linalg.solve(matrix[:, :3], momentum - matrix[:, 3:] @ joint_velocity)
            difference = np.linalg.norm(updated - omega)
            omega = updated
            if difference < 1e-10:
                break
        quaternion = before[3:7].copy()
        mujoco.mju_quatIntegrate(quaternion, omega, dt)
        return quaternion, omega
