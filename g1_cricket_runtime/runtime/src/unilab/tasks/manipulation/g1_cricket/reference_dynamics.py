"""Native inverse-dynamics requirements for offline cricket references."""

import mujoco
import numpy as np
from scipy.optimize import linprog


def curve_state(model, curve, time, step):
    poses = [curve(time + offset * step) for offset in (-2, -1, 0, 1, 2)]
    velocities = np.empty((3, model.nv))
    for index in range(3):
        mujoco.mj_differentiatePos(
            model, velocities[index], 2 * step, poses[index], poses[index + 2]
        )
    return poses[2], velocities[1], (velocities[2] - velocities[0]) / (2 * step)


class ReferenceDynamics:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.moment = np.empty((model.nu, model.nv))
        if not (
            np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
            and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
            and np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            and np.all(model.actuator_gainprm[:, 0] > 0)
            and np.all(model.actuator_forcelimited)
        ):
            raise ValueError("reference audit requires fixed-gain affine joint actuators")
        self.control_limits = model.jnt_range[model.actuator_trnid[:, 0]].copy()
        limited = model.actuator_ctrllimited.astype(bool)
        self.control_limits[limited, 0] = np.maximum(
            self.control_limits[limited, 0], model.actuator_ctrlrange[limited, 0]
        )
        self.control_limits[limited, 1] = np.minimum(
            self.control_limits[limited, 1], model.actuator_ctrlrange[limited, 1]
        )

    def evaluate(self, position, velocity, acceleration):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = position, velocity
        mujoco.mj_forward(model, data)
        mujoco.mju_sparse2dense(
            self.moment,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
        bias = (
            model.actuator_biasprm[:, 0]
            + model.actuator_biasprm[:, 1] * data.actuator_length
            + model.actuator_biasprm[:, 2] * data.actuator_velocity
        )
        gain = model.actuator_gainprm[:, 0]
        reachable = np.clip(
            gain[:, None] * self.control_limits + bias[:, None],
            model.actuator_forcerange[:, :1],
            model.actuator_forcerange[:, 1:],
        )
        data.qacc[:] = acceleration
        mujoco.mj_inverse(model, data)
        required = data.qfrc_inverse.copy()
        forces = np.linalg.lstsq(self.moment.T, required, rcond=None)[0]
        attainable = np.clip(forces, reachable[:, 0], reachable[:, 1])
        controls = np.clip((attainable - bias) / gain, *self.control_limits.T)
        return {
            "required_generalized_force": required,
            "required_motor_force": forces,
            "unactuated_residual": required - self.moment.T @ forces,
            "reachable_motor_force_bounds": reachable,
            "motor_limit_excess": np.maximum(
                np.maximum(
                    model.actuator_forcerange[:, 0] - forces,
                    forces - model.actuator_forcerange[:, 1],
                ),
                0,
            ),
            "reachable_force_excess": np.abs(forces - attainable),
            "bounded_controls": controls,
            "bounded_force_residual": required - self.moment.T @ attainable,
            "constraint_generalized_force": data.qfrc_constraint.copy(),
        }

    def support_allocation(
        self, acceleration, patches, motor_bounds, holder_bodies=None, *, minimum_motor_scale=False
    ):
        """Optimistic ideal support test; not the native contact solution."""
        model, data = self.model, self.data
        friction_dofs = np.flatnonzero(model.dof_frictionloss)
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        required = mass @ acceleration + data.qfrc_bias - data.qfrc_passive
        matrices = [self.moment.T]
        bounds = (
            [(None, None)] * model.nu if minimum_motor_scale else list(map(tuple, motor_bounds))
        )
        inequalities = []
        holder_start = model.nu + 6 * len(patches)
        width = (
            holder_start
            + (6 if holder_bodies is not None else 0)
            + len(friction_dofs)
            + int(minimum_motor_scale)
        )
        for index, (body, box, friction) in enumerate(patches):
            center = np.r_[box.mean(axis=0)[:2], 0.0]
            linear, angular = np.empty((3, model.nv)), np.empty((3, model.nv))
            mujoco.mj_jac(model, data, linear, angular, center, body)
            matrices.append(np.column_stack((linear.T, angular.T)))
            bounds.extend([(None, None), (None, None), (0, None)] + [(None, None)] * 3)
            start = model.nu + 6 * index
            for axis in (0, 1):
                for sign in (-1, 1):
                    row = np.zeros(width)
                    row[start + axis], row[start + 2] = sign, -friction
                    inequalities.append(row)
            # A rectangular foot patch overestimates the real support polygon.
            for moment, lower, upper in (
                (3, box[0, 1] - center[1], box[1, 1] - center[1]),
                (4, center[0] - box[1, 0], center[0] - box[0, 0]),
            ):
                for sign, coefficient in ((1, -upper), (-1, lower)):
                    row = np.zeros(width)
                    row[start + moment], row[start + 2] = sign, coefficient
                    inequalities.append(row)
        if holder_bodies is not None:
            wrist, ball = holder_bodies
            position = data.xpos[ball]
            jacobians = []
            for body in (wrist, ball):
                linear, angular = np.empty((3, model.nv)), np.empty((3, model.nv))
                mujoco.mj_jac(model, data, linear, angular, position, body)
                jacobians.append(np.column_stack((linear.T, angular.T)))
            matrices.append(jacobians[0] - jacobians[1])
            bounds.extend([(None, None)] * 6)
        friction_start = holder_start + (6 if holder_bodies is not None else 0)
        if len(friction_dofs):
            matrices.append(np.eye(model.nv)[:, friction_dofs])
            bounds.extend(
                (-model.dof_frictionloss[dof], model.dof_frictionloss[dof]) for dof in friction_dofs
            )
        objective = np.zeros(width)
        if minimum_motor_scale:
            matrices.append(np.zeros((model.nv, 1)))
            bounds.append((0, None))
            objective[-1] = 1
            for index, (low, high) in enumerate(model.actuator_forcerange):
                for sign, coefficient in ((1, -high), (-1, low)):
                    row = np.zeros(width)
                    row[index], row[-1] = sign, coefficient
                    inequalities.append(row)
        matrix = np.column_stack(matrices)
        inequality = np.asarray(inequalities).reshape(-1, width)
        solution = linprog(
            objective,
            A_ub=inequality,
            b_ub=np.zeros(len(inequality)),
            A_eq=matrix,
            b_eq=required,
            bounds=bounds,
            method="highs",
        )
        if solution.status not in (0, 2):
            raise RuntimeError(f"support allocation solver failed: {solution.message}")
        result = {"solver_status": int(solution.status), "feasible": bool(solution.success)}
        if solution.success:
            residual = matrix @ solution.x - required
            violation = float(np.maximum(inequality @ solution.x, 0).max(initial=0))
            bound_error = max(
                [0.0]
                + [
                    low - value
                    for value, (low, _) in zip(solution.x, bounds, strict=True)
                    if low is not None
                ]
                + [
                    value - high
                    for value, (_, high) in zip(solution.x, bounds, strict=True)
                    if high is not None
                ]
            )
            if max(np.abs(residual).max(), violation, bound_error) > 1e-6:
                raise RuntimeError("support allocation did not satisfy its equations and bounds")
            result.update(
                generalized_residual=residual.tolist(),
                max_inequality_violation=violation,
                max_bound_violation=float(bound_error),
                motor_forces=solution.x[: model.nu].tolist(),
                support_wrenches_world=solution.x[model.nu : model.nu + 6 * len(patches)]
                .reshape(-1, 6)
                .tolist(),
                holder_wrench_world=solution.x[holder_start : holder_start + 6].tolist()
                if holder_bodies is not None
                else None,
                friction_dofs=friction_dofs.tolist(),
                friction_forces=solution.x[
                    friction_start : friction_start + len(friction_dofs)
                ].tolist(),
            )
            if minimum_motor_scale:
                result["minimum_motor_scale"] = float(solution.x[-1])
        return result
