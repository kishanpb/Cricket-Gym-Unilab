"""Force frame/sign checks and all-row resolution evidence."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_g1_cricket_contact_resolution import ContactReplay, consistency
from evaluate_g1_cricket_residual import CricketReplay
from probe_g1_cricket_impact import probe


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("ball_type", ["sphere", "box"])
def test_signed_impulse_matches_free_ball_momentum(monkeypatch, axis, ball_type):
    other = "box" if ball_type == "sphere" else "sphere"
    position = np.zeros(3)
    position[axis] = 0.055
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep="0.0005" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <geom name="bat_blade" type="{other}" size="0.02 0.02 0.02"/>
        <body pos="{" ".join(map(str, position))}"><freejoint/>
          <geom name="ball_geom" type="{ball_type}" size="0.036 0.036 0.036"
                mass="0.156" friction="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[axis] = -2
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial = np.empty(size)
    mujoco.mj_getState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    trajectory = []
    for _ in range(40):
        mujoco.mj_step(model, data)
        snapshot = np.empty(size)
        mujoco.mj_getState(model, data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        trajectory.append(snapshot)
    expected_impulse = 0.156 * (data.qvel[:3] - initial[1 + model.nq :][:3])
    assert np.linalg.norm(expected_impulse) > 0.01
    monkeypatch.setattr(CricketReplay, "step", lambda *_: (None, np.asarray(trajectory)))
    replay = ContactReplay.__new__(ContactReplay)
    replay.model, replay.data = model, mujoco.MjData(model)
    replay.steps, replay.ball_geom = 40, model.geom("ball_geom").id
    env = SimpleNamespace(
        get_physics_state_snapshot=lambda: initial[None],
        cfg=SimpleNamespace(sim_dt=0.0005),
        action_manager=SimpleNamespace(
            get_term=lambda _: SimpleNamespace(processed_action=np.empty((1, 0)))
        ),
    )
    _, impulse = replay.step(env, None)
    np.testing.assert_allclose(impulse, expected_impulse, atol=1e-10, rtol=1e-10)


def test_consistency_rejects_missing_contact_and_threshold_changes():
    row = dict(
        blade_contact_seen=True,
        guard_contacts=[],
        seconds=2,
        terminated=False,
        blade_peak_force_norm_n=20,
        blade_force_norm_integral_ns=0.4,
        blade_contact_seconds=0.1,
        blade_max_penetration_m=0.002,
        first_separation_ball_vx_m_s=0.5,
    )
    assert not consistency(row, row)
    for key, value in (
        ("blade_peak_force_norm_n", 25),
        ("blade_contact_seen", False),
        ("first_separation_ball_vx_m_s", None),
        ("guard_contacts", ["bat_pitch"]),
        ("seconds", 1.9),
    ):
        assert key in consistency({**row, key: value}, row)


def test_complete_retained_resolution_matrix():
    directory = ROOT / "g1_cricket_results/residual_v3"
    report = json.loads((directory / "contact_resolution.json").read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert report["parent_report_sha256"] == digest(directory / "evaluation.json")
    for name, expected in report["source_sha256"].items():
        assert digest(ROOT / name) == expected
    assert not report["physical_calibration_validated"]
    assert len(report["rows"]) == report["expected_rows"] == 288
    for dt in (0.002, 0.001, 0.0005):
        rows = [r for r in report["rows"] if r["sim_dt"] == dt]
        assert {(r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in rows} == {
            (h, c, o, s)
            for h in ("right", "left")
            for c in ("zero_residual", "ppo")
            for o in (-0.12, -0.1, 0.0)
            for s in range(4301, 4309)
        }
        for row in rows:
            assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
            assert (
                0
                <= row["blade_active_load_seconds"]
                <= row["blade_contact_seconds"]
                <= row["seconds"]
            )
            assert (
                np.linalg.norm(row["blade_world_impulse_on_ball_ns"])
                <= row["blade_force_norm_integral_ns"] + 1e-12
            )
    for a, b, comparison in zip(
        report["rows"][96:192],
        report["rows"][192:],
        report["one_vs_half_ms_comparisons"],
        strict=True,
    ):
        assert comparison["failed_checks"] == consistency(a, b)
    assert report["numerically_consistent_rows"] == sum(
        not r["failed_checks"] for r in report["one_vs_half_ms_comparisons"]
    )


def test_isolated_impact_reproduces_all_design_rows():
    report = json.loads((ROOT / "g1_cricket_results/residual_v3/isolated_impact.json").read_text())
    assert (
        report["source_sha256"]
        == hashlib.sha256((ROOT / "scripts/probe_g1_cricket_impact.py").read_bytes()).hexdigest()
    )
    assert (
        report["candidate_status"] == "uncalibrated_design_probe_not_applied_to_native_environment"
    )
    assert len(report["rows"]) == 12
    for row in report["rows"]:
        assert probe(row["sim_dt"], row["solref_time_constant_s"]) == row
