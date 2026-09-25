"""Local contact-aware acceleration tracking through bounded motor commands."""

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .prior import SDK_JOINTS


class ContactAccelerationControl:
    def __init__(self, model, reference_velocity, baseline, *, foot_reference=None):
        self.model = model
        self.scratch = mujoco.MjData(model)
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        self.qa, self.va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self.limits = model.jnt_range[joints].copy()
        self.selected = np.r_[np.arange(6), self.va]
        self.weights = np.r_[np.full(6, 2.0), np.full(29, 0.2)]
        self.reference_acceleration = np.gradient(reference_velocity, 0.02, axis=0)
        self.baseline = baseline
        self.trace = []
        self.foot_reference = foot_reference
        if foot_reference is not None:
            self.feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
            reference_data = mujoco.MjData(model)
            positions, velocities = [], []
            for pose, velocity in zip(foot_reference, reference_velocity, strict=True):
                reference_data.qpos[:], reference_data.qvel[:] = pose, velocity
                mujoco.mj_forward(model, reference_data)
                positions.append(reference_data.xpos[self.feet].copy())
                velocities.append(
                    [self.foot_jacobian(reference_data, body) @ velocity for body in self.feet]
                )
            self.foot_positions = np.asarray(positions)
            self.foot_velocities = np.asarray(velocities)
            self.foot_accelerations = np.gradient(self.foot_velocities, 0.02, axis=0)

    def foot_jacobian(self, data, body):
        jacobian = np.empty((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jacobian, None, data.xpos[body], body)
        return jacobian

    def foot_tasks(self, data, frame):
        mujoco.mj_copyData(self.scratch, self.model, data)
        mujoco.mj_forward(self.model, self.scratch)
        matrices, targets = [], []
        for index, body in enumerate(self.feet):
            jacobian = self.foot_jacobian(self.scratch, body)
            derivative = np.empty_like(jacobian)
            mujoco.mj_jacDot(
                self.model, self.scratch, derivative, None, self.scratch.xpos[body], body
            )
            matrices.append(jacobian[:, self.selected])
            targets.append(
                self.foot_accelerations[frame, index]
                + 80 * (self.foot_positions[frame, index] - self.scratch.xpos[body])
                + 18 * (self.foot_velocities[frame, index] - jacobian @ data.qvel)
                - derivative @ data.qvel
            )
        return np.vstack(matrices), np.concatenate(targets)

    def acceleration(self, data, control):
        mujoco.mj_copyData(self.scratch, self.model, data)
        self.scratch.ctrl[:] = control
        mujoco.mj_forward(self.model, self.scratch)
        return self.scratch.qacc[self.selected].copy()

    def jacobian(self, data, control):
        jacobian = np.empty((len(self.selected), self.model.nu))
        for joint in range(self.model.nu):
            plus, minus = control.copy(), control.copy()
            plus[joint] = min(control[joint] + 1e-4, self.limits[joint, 1])
            minus[joint] = max(control[joint] - 1e-4, self.limits[joint, 0])
            jacobian[:, joint] = (
                self.acceleration(data, plus) - self.acceleration(data, minus)
            ) / (plus[joint] - minus[joint])
        return jacobian

    def __call__(self, data, target, velocity):
        model = self.model
        control, _ = self.baseline(model, data, target, velocity, self.qa, self.va)
        anchor = np.clip(control, self.limits[:, 0], self.limits[:, 1])
        error = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, error, 1.0, data.qpos, target)
        rotation, reference_rotation = np.empty(9), np.empty(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        mujoco.mju_quat2Mat(reference_rotation, target[3:7])
        transform = rotation.reshape(3, 3).T @ reference_rotation.reshape(3, 3)
        desired_velocity = velocity.copy()
        desired_velocity[3:6] = transform @ velocity[3:6]
        frame = int((data.time + 1e-9) / 0.02)
        feedforward = self.reference_acceleration[frame].copy()
        feedforward[3:6] = transform @ feedforward[3:6]
        desired = (feedforward + 80 * error + 18 * (desired_velocity - data.qvel))[self.selected]
        task_matrix = np.diag(self.weights)
        task_target = self.weights * desired
        if self.foot_reference is not None:
            feet, foot_target = self.foot_tasks(data, frame)
            task_matrix = np.vstack((task_matrix, 4 * feet))
            task_target = np.r_[task_target, 4 * foot_target]

        def score(command):
            residual = task_matrix @ self.acceleration(data, command) - task_target
            return float(residual @ residual + 4 * np.sum((command - anchor) ** 2))

        command = anchor.copy()
        initial = best = score(command)
        stages = []
        for _ in range(2):
            acceleration = self.acceleration(data, command)
            jacobian = self.jacobian(data, command)
            matrix = np.vstack((task_matrix @ jacobian, 2 * np.eye(model.nu)))
            rhs = np.r_[task_target - task_matrix @ acceleration, 2 * (anchor - command)]
            solution = lsq_linear(
                matrix,
                rhs,
                bounds=(
                    np.maximum(self.limits[:, 0] - command, -0.3),
                    np.minimum(self.limits[:, 1] - command, 0.3),
                ),
                method="bvls",
                tol=1e-9,
                max_iter=100,
            )
            origin = command.copy()
            for fraction in (1.0, 0.5, 0.25):
                candidate = np.clip(
                    origin + fraction * solution.x, self.limits[:, 0], self.limits[:, 1]
                )
                value = score(candidate)
                if value < best:
                    command, best = candidate, value
            stages.append(
                {
                    "solver_status": int(solution.status),
                    "solver_iterations": int(solution.nit),
                    "score": best,
                }
            )
        self.trace.append(
            {
                "time_s": float(data.time),
                "initial_score": initial,
                "final_score": best,
                "stages": stages,
            }
        )
        return command
