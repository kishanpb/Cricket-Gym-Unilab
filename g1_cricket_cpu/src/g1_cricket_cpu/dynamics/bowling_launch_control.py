"""Experimental native-model shoulder steering; not a learned cricket policy."""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from .amp_running_probe import JOINTS
TARGET_VELOCITY = np.array([10.0, 0.0, 5.0])
HORIZON = 0.06

def launch_ready(position, velocity, shoulder, elbow, gravity, contact_height):
    height = position[2] - contact_height
    if height <= 0:
        return False
    flight_time = (velocity[2] + np.sqrt(velocity[2] ** 2 + 2 * gravity * height)) / gravity
    bounce = position[:2] + flight_time * velocity[:2]
    return bool(position[2] > shoulder[2] + 0.12 and elbow[2] > shoulder[2] and (velocity[0] > 6) and (abs(velocity[1]) < 2) and (8 < bounce[0] < 15) and (abs(bounce[1]) < 1.32))

class LaunchController:

    def __init__(self, model, hand):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.ids = np.array([model.actuator(name + '_joint').id for name in JOINTS])
        joints = model.actuator_trnid[self.ids, 0]
        self.q, self.v = (model.jnt_qposadr[joints], model.jnt_dofadr[joints])
        self.kp = model.actuator_gainprm[self.ids, 0]
        self.kd = -model.actuator_biasprm[self.ids, 2]
        self.caps = model.actuator_forcerange[self.ids, 1]
        self.selected = np.array([i for i, name in enumerate(JOINTS) if name.startswith(hand + '_shoulder_')])
        self.ball_v = int(model.joint('ball_free').dofadr[0])

    def command(self, data, baseline):
        model, scratch = (self.model, self.scratch)
        selected = self.selected
        base = np.clip(baseline, -self.caps, self.caps)
        desired = (TARGET_VELOCITY - data.qvel[self.ball_v:self.ball_v + 3]) / HORIZON

        def acceleration(values):
            mujoco.mj_copyData(scratch, model, data)
            torque = base.copy()
            torque[selected] = values
            scratch.ctrl[self.ids] = scratch.qpos[self.q] + (torque + self.kd * scratch.qvel[self.v]) / self.kp
            mujoco.mj_forward(model, scratch)
            return scratch.qacc[self.ball_v:self.ball_v + 3].copy()
        current = base[selected].copy()
        initial_acceleration = acceleration(current)
        actual = initial_acceleration.copy()
        for _ in range(3):
            jacobian = np.empty((3, 3))
            for index, cap in enumerate(self.caps[selected]):
                delta = -1.0 if current[index] + 1 > cap else 1.0
                perturbed = current.copy()
                perturbed[index] += delta
                jacobian[:, index] = (acceleration(perturbed) - actual) / delta
            ridge = 0.03
            solution = lsq_linear(np.vstack((jacobian, ridge * np.eye(3))), np.r_[desired - actual + jacobian @ current, ridge * base[selected]], bounds=(-self.caps[selected], self.caps[selected]))
            trial = solution.x
            trial_acceleration = acceleration(trial)
            if np.linalg.norm(trial_acceleration - desired) >= np.linalg.norm(actual - desired):
                break
            current, actual = (trial, trial_acceleration)
        return (current, np.r_[desired, initial_acceleration, actual])

def verify_launch_commands(trace, model, hand):
    controller = LaunchController(model, hand)
    data = mujoco.MjData(model)
    count = len(trace['states']) - 1
    assert trace['launch_rows'].shape == (count, 13)
    assert trace['launch_warmstart_rows'].shape == (count, model.nv)
    assert not np.any(trace['residual_rows']), 'Model launch screen is not a residual-policy trial'
    active_count = 0
    for i, state in enumerate(trace['states'][:-1]):
        active = not trace['released'][i] and 1.7 <= trace['swing_rows'][i, 0] <= 2.0
        expected = np.zeros(13)
        if active:
            mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            data.eq_active[model.equality('ball_holder').id] = True
            data.qacc_warmstart[:] = trace['launch_warmstart_rows'][i]
            target, velocity = np.split(trace['targets'][i], 2)
            kp, kd = np.split(trace['impedance_rows'][i], 2)
            baseline = kp * (target - data.qpos[controller.q]) + kd * (velocity - data.qvel[controller.v]) + trace['feedforward_rows'][i]
            torque, diagnostic = controller.command(data, baseline)
            expected = np.r_[1, torque, diagnostic]
            active_count += 1
        np.testing.assert_allclose(trace['launch_rows'][i], expected, rtol=0, atol=1e-09)
    return active_count
