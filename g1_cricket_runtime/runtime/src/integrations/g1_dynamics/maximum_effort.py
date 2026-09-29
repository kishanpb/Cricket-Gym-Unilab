"""Bounded maximum-effort interventions, not a robot capability upper bound."""

import numpy as np


def maximum_effort(request, limits):
    return np.where(request > 0, limits[:, 1], np.where(request < 0, limits[:, 0], 0.0))


def servo_effort_target(model, data, ids):
    joints = model.actuator_trnid[ids, 0]
    q, v = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    kp = model.actuator_gainprm[ids, 0]
    kd = -model.actuator_biasprm[ids, 2]
    request = kp * (data.ctrl[ids] - data.qpos[q]) - kd * data.qvel[v]
    torque = maximum_effort(request, model.actuator_forcerange[ids])
    target = data.qpos[q] + (torque + kd * data.qvel[v]) / kp
    target = np.where(model.actuator_ctrllimited[ids],
                      np.clip(target, *model.actuator_ctrlrange[ids].T), target)
    return target, request, torque
