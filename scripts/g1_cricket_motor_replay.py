"""Solve-phase motor telemetry alongside the unchanged full cricket replay."""

import mujoco
import numpy as np
from audit_g1_cricket_wrist_saturation import torque_request
from g1_cricket_trial import ImpactReplay


def motor_summary(tick, target, requested, applied, positions, velocities, limits):
    requested, applied = np.asarray(requested), np.asarray(applied)
    error = np.abs(applied - np.clip(requested, *limits))
    np.testing.assert_allclose(error, 0, rtol=0, atol=1e-10)
    return {
        "tick": tick,
        "target_rad": float(target),
        "physics_steps": len(requested),
        "negative_saturated_steps": int((requested <= limits[0]).sum()),
        "positive_saturated_steps": int((requested >= limits[1]).sum()),
        "requested_torque_range_nm": [float(requested.min()), float(requested.max())],
        "applied_torque_range_nm": [float(applied.min()), float(applied.max())],
        "solve_joint_position_range_rad": [min(positions), max(positions)],
        "solve_joint_velocity_range_rad_s": [min(velocities), max(velocities)],
        "maximum_clipped_torque_error_nm": float(error.max()),
    }


class MotorReplay(ImpactReplay):
    def __init__(self, env, name, limits):
        super().__init__(env)
        self.actuator = self.model.actuator(name).id
        m, a = self.model, self.actuator
        assert m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
        assert m.joint(m.actuator_trnid[a, 0]).name == name
        assert m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
        assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
        assert m.actuator_forcelimited[a] and not m.actuator_ctrllimited[a]
        np.testing.assert_array_equal(m.actuator_forcerange[a], limits)
        np.testing.assert_array_equal(m.actuator_gear[a], [1, 0, 0, 0, 0, 0])
        self.limits, self.trace = limits, []

    def step(self, env, action, tick):
        initial = env.get_physics_state_snapshot()
        result, impacts = super().step(env, action, tick)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = env.action_manager.get_term("residual").processed_action[0]
        requested, applied, positions, velocities = [], [], [], []
        for _ in range(self.steps):
            mujoco.mj_step(self.model, self.data)
            requested.append(torque_request(self.model, self.data, self.actuator))
            applied.append(float(self.data.actuator_force[self.actuator]))
            positions.append(float(self.data.actuator_length[self.actuator]))
            velocities.append(float(self.data.actuator_velocity[self.actuator]))
        endpoint = np.empty(initial.shape[1])
        mujoco.mj_getState(self.model, self.data, endpoint, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(endpoint, result[1][-1])
        if self.data.warning.number.any():
            raise RuntimeError("MuJoCo warning during motor telemetry replay")
        self.trace.append(
            motor_summary(
                tick,
                self.data.ctrl[self.actuator],
                requested,
                applied,
                positions,
                velocities,
                self.limits,
            )
        )
        return result, impacts
