"""Motor-only motion compensation, excluding native constraint stabilization."""

import mujoco
import numpy as np

from .batting_dynamics import independent_ball_dofs, reference_derivatives


def motion_increment(model, data, position, velocity, acceleration):
    data.qpos[:], data.qvel[:] = position, 0
    mujoco.mj_forward(model, data)
    static = data.qfrc_bias - data.qfrc_passive
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    return mass @ acceleration + data.qfrc_bias - data.qfrc_passive - static


def bounded_reference_forces(model, poses, times, gravity, control_limits):
    joints = model.actuator_trnid[:, 0]
    dofs = model.jnt_dofadr[joints]
    # The G1 owner uses one unit-gear position actuator per hinge.
    np.testing.assert_array_equal(model.actuator_gear[:, 0], 1)
    velocity, acceleration, endpoint = reference_derivatives(model, poses, times)
    ball = independent_ball_dofs(model)
    velocity[:, ball], acceleration[:, ball] = 0, 0
    data = mujoco.MjData(model)
    root_dofs = 6 if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
    forces, rows = [], []
    for index, pose in enumerate(poses):
        increment = motion_increment(model, data, pose, velocity[index], acceleration[index])
        bias = (
            model.actuator_biasprm[:, 0]
            + model.actuator_biasprm[:, 1] * data.actuator_length
            + model.actuator_biasprm[:, 2] * data.actuator_velocity
        )
        reachable = np.clip(
            model.actuator_gainprm[:, :1] * control_limits + bias[:, None],
            model.actuator_forcerange[:, :1],
            model.actuator_forcerange[:, 1:],
        )
        desired = gravity[index] + increment[dofs]
        bounded = np.clip(desired, *reachable.T)
        forces.append(bounded)
        rows.append(
            {
                "time_s": float(times[index]),
                "endpoint_stencil": bool(endpoint[index]),
                "motor_increment_nm": increment[dofs].tolist(),
                "unapplied_root_increment": increment[:root_dofs].tolist(),
                "desired_motor_force_nm": desired.tolist(),
                "bounded_motor_force_nm": bounded.tolist(),
                "clipping_nm": np.abs(desired - bounded).tolist(),
                "reachable_motor_force_nm": reachable.tolist(),
            }
        )
    return np.asarray(forces), rows
