"""Replay scales 1 and 0 at both timesteps; diagnose, never change, motor limits."""

import json
from importlib.metadata import version

import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import ROOT, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_trial import ImpactReplay, trial
from probe_g1_cricket_reachability import make_env
from probe_g1_cricket_wrist_pitch import DIRECTORY, action_at


def torque_request(model, data, actuator):
    gain = model.actuator_gainprm[actuator, 0]
    bias = model.actuator_biasprm[actuator, :3]
    return float(
        gain * data.ctrl[actuator]
        + bias @ [1, data.actuator_length[actuator], data.actuator_velocity[actuator]]
    )


def summarize(tick, control, requested, applied, positions, velocities):
    requested, applied = np.asarray(requested), np.asarray(applied)
    clipped = np.clip(requested, -5, 5)
    np.testing.assert_allclose(applied, clipped, rtol=0, atol=1e-10)
    return {
        "tick": tick,
        "target_rad": float(control),
        "physics_steps": len(requested),
        "saturated_steps": int((np.abs(requested) >= 5).sum()),
        "positive_saturated_steps": int((requested >= 5).sum()),
        "negative_saturated_steps": int((requested <= -5).sum()),
        "requested_torque_range_nm": [float(requested.min()), float(requested.max())],
        "applied_torque_range_nm": [float(applied.min()), float(applied.max())],
        "solve_joint_position_range_rad": [min(positions), max(positions)],
        "solve_joint_velocity_range_rad_s": [min(velocities), max(velocities)],
        "maximum_clipped_torque_error_nm": float(np.abs(applied - clipped).max()),
    }


class WristReplay(ImpactReplay):
    def __init__(self, env):
        super().__init__(env)
        self.actuator = self.model.actuator("right_wrist_pitch_joint").id
        a, m = self.actuator, self.model
        assert m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
        assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
        assert m.actuator_forcelimited[a] and not m.actuator_ctrllimited[a]
        np.testing.assert_array_equal(m.actuator_forcerange[a], [-5, 5])
        self.trace = []

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
            raise RuntimeError("MuJoCo warning during wrist diagnostic replay")
        self.trace.append(
            summarize(
                tick, self.data.ctrl[self.actuator], requested, applied, positions, velocities
            )
        )
        return result, impacts


def run():
    output = DIRECTORY / "wrist_saturation.json"
    if output.exists():
        raise FileExistsError("wrist saturation diagnostic already retained")
    path = DIRECTORY / "evaluation.json"
    parent = json.loads(path.read_text())
    pins = dict(parent["input_sha256"])
    for source in (
        str(path.relative_to(ROOT)),
        "scripts/audit_g1_cricket_wrist_saturation.py",
        "tests/scripts/test_g1_cricket_wrist_saturation.py",
    ):
        pins[source] = sha256(ROOT / source)
    check_hashes(pins)
    assert executor_manifest() == parent["executor"]
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    torch.set_num_threads(2)
    rows = []
    for dt in parent["plan"]["physics_dt_seconds"]:
        env = make_env("right", dt, "mjbatch")
        try:
            for scale in (1, 0):
                reference = next(
                    r
                    for r in parent["rows"]
                    if r["engine"] == "mjbatch" and r["sim_dt"] == dt and r["scale"] == scale
                )
                identity = {
                    k: reference["outcome"][k] for k in ("hand", "controller", "offset_m", "seed")
                }
                replay = WristReplay(env)
                result = trial(env, replay, identity, lambda tick: action_at(scale, tick), 100)
                assert result == {k: reference[k] for k in result}, "diagnostic changed parent row"
                assert len(replay.trace) == 100
                rows.append(
                    {"scale": scale, "sim_dt": dt, "trace": replay.trace, "parent_row_exact": True}
                )
                selected = replay.trace[10:20]
                print(
                    json.dumps(
                        {
                            "scale": scale,
                            "sim_dt": dt,
                            "forward_saturated_steps": sum(r["saturated_steps"] for r in selected),
                            "forward_physics_steps": sum(r["physics_steps"] for r in selected),
                        }
                    ),
                    flush=True,
                )
        finally:
            env.close()
    check_hashes(pins)
    assert executor_manifest() == parent["executor"]
    output.write_text(
        json.dumps(
            {
                "scope": "motor_saturation_diagnostic_not_control_or_training",
                "input_sha256": pins,
                "executor": parent["executor"],
                "versions": parent["versions"],
                "engine": "mjbatch",
                "joint": "right_wrist_pitch_joint",
                "rows": rows,
                "sampling": "Every physics solve, retaining min/max and saturation counts per control tick",
                "full_trial_count": 4,
                "new_training": False,
                "policy_promoted": False,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
