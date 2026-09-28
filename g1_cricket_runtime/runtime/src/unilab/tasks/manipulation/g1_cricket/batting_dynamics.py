"""Offline decomposition of native inverse dynamics for a fixed batting reference."""

import mujoco
import numpy as np

from .reference_dynamics import ReferenceDynamics


def reference_derivatives(model, poses, times, half_stencil=1):
    if half_stencil not in (1, 2):
        raise ValueError("half_stencil must be one or two reference frames")
    if len(poses) != len(times) or len(times) < 5 or np.any(np.diff(times) <= 0):
        raise ValueError("reference needs matching poses and increasing times")
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        quaternions = poses[:, 3:7]
        if not np.allclose(np.abs(quaternions @ quaternions[0]), 1, atol=1e-10, rtol=0):
            raise ValueError("reference root quaternion must be constant")
    velocity = np.empty((len(poses), model.nv))
    acceleration = np.empty_like(velocity)
    for index in range(len(poses)):
        before, after = max(index - half_stencil, 0), min(index + half_stencil, len(poses) - 1)
        mujoco.mj_differentiatePos(
            model, velocity[index], times[after] - times[before], poses[before], poses[after]
        )
    for index in range(len(poses)):
        before, after = max(index - half_stencil, 0), min(index + half_stencil, len(poses) - 1)
        acceleration[index] = (velocity[after] - velocity[before]) / (times[after] - times[before])
    endpoint = (np.arange(len(poses)) < 2 * half_stencil) | (
        np.arange(len(poses)) >= len(poses) - 2 * half_stencil
    )
    return velocity, acceleration, endpoint


def independent_ball_dofs(model):
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if joint < 0:
        return np.array([], dtype=int)
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("ball_free must be an independent free joint")
    start = model.jnt_dofadr[joint]
    return np.arange(start, start + 6)


def constraint_contributions(model, data):
    kind = mujoco.mjtConstraint
    groups = {
        "equality": [kind.mjCNSTR_EQUALITY],
        "contact": [
            kind.mjCNSTR_CONTACT_FRICTIONLESS,
            kind.mjCNSTR_CONTACT_PYRAMIDAL,
            kind.mjCNSTR_CONTACT_ELLIPTIC,
        ],
        "friction_loss": [kind.mjCNSTR_FRICTION_DOF, kind.mjCNSTR_FRICTION_TENDON],
        "joint_limit": [kind.mjCNSTR_LIMIT_JOINT, kind.mjCNSTR_LIMIT_TENDON],
    }
    result, covered = {}, np.zeros(data.nefc, dtype=bool)
    for label, types in groups.items():
        selected = np.isin(data.efc_type, types)
        covered |= selected
        generalized = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model, data, generalized, np.where(selected, data.efc_force, 0))
        result[label] = generalized
    generalized = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, generalized, np.where(~covered, data.efc_force, 0))
    result["other"] = generalized
    np.testing.assert_allclose(sum(result.values()), data.qfrc_constraint, atol=1e-7, rtol=1e-10)
    return result


def evaluate_modes(helper: ReferenceDynamics, position, velocity, acceleration):
    model = helper.model
    robot = np.setdiff1d(np.arange(model.nv), independent_ball_dofs(model))
    root = (
        np.intersect1d(robot, np.arange(6))
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        else np.array([], dtype=int)
    )
    zeros = np.zeros(model.nv)
    modes, components = {}, {}
    for name, speed, accel in (
        ("static", zeros, zeros),
        ("velocity", velocity, zeros),
        ("dynamic", velocity, acceleration),
    ):
        result = helper.evaluate(position, speed, accel)
        data = helper.data
        mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, data, mass)
        inertia = mass @ accel
        unconstrained = inertia + data.qfrc_bias - data.qfrc_passive
        np.testing.assert_allclose(
            result["required_generalized_force"],
            unconstrained - data.qfrc_constraint,
            atol=1e-7,
            rtol=1e-10,
        )
        forces = result["required_motor_force"]
        force_peak = int(np.argmax(result["motor_limit_excess"]))
        control_peak = int(np.argmax(result["reachable_force_excess"]))
        bias = (
            model.actuator_biasprm[:, 0]
            + model.actuator_biasprm[:, 1] * data.actuator_length
            + model.actuator_biasprm[:, 2] * data.actuator_velocity
        )
        required_controls = (forces - bias) / model.actuator_gainprm[:, 0]
        control_excess = np.maximum(
            np.maximum(
                helper.control_limits[:, 0] - required_controls,
                required_controls - helper.control_limits[:, 1],
            ),
            0,
        )
        modes[name] = {
            "required_motor_force": forces,
            "required_controls": required_controls,
            "motor_limit_excess": result["motor_limit_excess"],
            "reachable_force_excess": result["reachable_force_excess"],
            "control_limit_excess": control_excess,
            "peak_motor_excess": float(result["motor_limit_excess"][force_peak]),
            "peak_motor_excess_joint": model.joint(int(model.actuator_trnid[force_peak, 0])).name,
            "peak_reachable_force_excess": float(result["reachable_force_excess"][control_peak]),
            "peak_reachable_force_excess_joint": model.joint(
                int(model.actuator_trnid[control_peak, 0])
            ).name,
            "root_residual": result["unactuated_residual"][root],
            "robot_unactuated_residual": result["unactuated_residual"][robot],
            "robot_required_generalized_force": result["required_generalized_force"][robot],
            "constraint_contributions": {
                key: value[robot] for key, value in constraint_contributions(model, data).items()
            },
        }
        components[name] = {
            "inertia": inertia[robot],
            "bias": data.qfrc_bias[robot].copy(),
            "passive": data.qfrc_passive[robot].copy(),
            "unconstrained": unconstrained[robot],
        }
    increments = {}
    for name, first, second in (
        ("velocity_minus_static", "velocity", "static"),
        ("dynamic_minus_velocity", "dynamic", "velocity"),
        ("dynamic_minus_static", "dynamic", "static"),
    ):
        increments[name] = {
            key: components[first][key] - components[second][key] for key in components[first]
        }
        increments[name]["required_generalized_force"] = (
            modes[first]["robot_required_generalized_force"]
            - modes[second]["robot_required_generalized_force"]
        )
    return {"modes": modes, "increments": increments, "robot_dof_indices": robot}
