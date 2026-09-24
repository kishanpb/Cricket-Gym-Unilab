"""Opt-in model identity and complete paired humanoid transfer evidence."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_residual import CricketReplay, sha256
from probe_g1_cricket_model_transfer import (
    DIRECTORY,
    PLAN,
    action,
    compare_group,
    make_env,
    model_contrasts,
)
from probe_g1_cricket_reachability import action_at
from probe_g1_cricket_terminal_residual import CANDIDATE


@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_owner_resolves_only_declared_task_and_physics_change(engine):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        new = compose("config", overrides=[f"task=g1_cricket_compliance_v2/{engine}"])
        old = compose("config", overrides=[f"task=g1_cricket_tanh_v1/{engine}"])
    assert new.training.task_name == "G1CricketImpactV2"
    assert new.env.sim_dt == 0.0000625
    old.training.task_name = new.training.task_name
    old.env.sim_dt = new.env.sim_dt
    assert old == new


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_compiled_model_changes_only_time_constant_and_preserves_reset(hand, engine):
    old = make_env("control_4ms", hand, 0.0000625, engine)
    new = make_env("candidate_2ms", hand, 0.0000625, engine)
    try:
        a, b = CricketReplay(old), CricketReplay(new)
        assert a.names == b.names
        assert a.steps == b.steps == 320
        np.testing.assert_array_equal(a.model.pair_solref, [[0.004, 1]])
        np.testing.assert_array_equal(b.model.pair_solref, [[0.002, 1]])
        for field in (
            "pair_solimp",
            "pair_friction",
            "pair_geom1",
            "pair_geom2",
            "pair_dim",
            "pair_gap",
            "pair_margin",
            "body_mass",
            "body_inertia",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_solref",
            "geom_solimp",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "key_qpos",
            "key_ctrl",
        ):
            np.testing.assert_array_equal(getattr(a.model, field), getattr(b.model, field))
        old.reset(seed=4301)
        new.reset(seed=4301)
        np.testing.assert_array_equal(
            old.get_physics_state_snapshot(), new.get_physics_state_snapshot()
        )
        for _ in range(2):
            ra = a.step(old, action("zero_residual", 0))
            rb = b.step(new, action("zero_residual", 0))
            np.testing.assert_array_equal(ra[1], rb[1])
            np.testing.assert_array_equal(ra[0].reward, rb[0].reward)
        assert a.state_error == a.sensor_error == b.state_error == b.sensor_error == 0
    finally:
        old.close()
        new.close()


def test_candidate_rejects_underresolved_physics():
    with pytest.raises(ValueError, match="timestep <=0.0625 ms"):
        make_env("candidate_2ms", "right", 0.000125, "mujoco")


def test_commands_match_parent_for_all_ticks():
    for tick in range(100):
        np.testing.assert_array_equal(
            action("zero_residual", tick), np.zeros((1, 7), dtype=np.float32)
        )
        np.testing.assert_array_equal(action(CANDIDATE["id"], tick), action_at(CANDIDATE, tick))


def test_comparator_rejects_missing_trials():
    with pytest.raises(ValueError, match="four unique"):
        compare_group([], "candidate_2ms", "right", CANDIDATE["id"])


def test_witness_rejects_fixture_resolution_and_executor_mismatch():
    outcome = dict(
        hand="right",
        controller=CANDIDATE["id"],
        offset_m=0.0,
        seed=4301,
        seconds=2,
        passed=True,
        failures=[],
        blade_contact_seen=True,
        maximum_blade_penetration_m=0.002,
        first_separation_ball_vx_m_s=1.2,
        ball_contact_peak_force_norm_n={"bat_blade": 10},
        fixture_peak_force_norm_n=20,
        fixture_peak_torque_norm_nm=0.2,
    )
    rows = [
        dict(
            model="candidate_2ms",
            engine=e,
            sim_dt=dt,
            outcome=deepcopy(outcome),
            terminated=False,
            truncated=True,
            blade_episode_count=1,
            first_impact=None,
            first_loaded_impact=None,
        )
        for e in PLAN["executors"]
        for dt in PLAN["timesteps_s"]
    ]
    assert compare_group(rows, "candidate_2ms", "right", CANDIDATE["id"])["scripted_witness"]
    with pytest.raises(ValueError, match="four unique"):
        compare_group(rows[:-1] + rows[:1], "candidate_2ms", "right", CANDIDATE["id"])
    rows[-1]["first_loaded_impact"] = {"changed": True}
    assert not compare_group(rows, "candidate_2ms", "right", CANDIDATE["id"])["scripted_witness"]
    rows[-1]["first_loaded_impact"] = None
    for row in rows:
        if row["sim_dt"] == PLAN["timesteps_s"][1]:
            row["outcome"]["fixture_peak_force_norm_n"] += 3
    result = compare_group(rows, "candidate_2ms", "right", CANDIDATE["id"])
    assert result["executor_evidence_exact"] and not result["scripted_witness"]
    assert result["fixture_resolution_failures"] == ["fixture_peak_force_norm_n"]
    rows[-1]["outcome"]["fixture_peak_force_norm_n"] += 0.01
    assert not compare_group(rows, "candidate_2ms", "right", CANDIDATE["id"])[
        "executor_evidence_exact"
    ]


def test_complete_retained_report():
    path = DIRECTORY / "evaluation.json"
    if not path.exists():
        pytest.skip("source tests before frozen transfer matrix")
    report = json.loads(path.read_text())
    assert report["plan"] == PLAN
    assert not report["new_training"] and not report["policy_promoted"]
    assert not report["physical_calibration_validated"]
    for name, expected in report["input_sha256"].items():
        assert sha256(ROOT / name) == expected
    rows = report["rows"]
    assert len(rows) == PLAN["full_trial_count"]
    assert {
        (r["model"], r["engine"], r["sim_dt"], r["outcome"]["hand"], r["outcome"]["controller"])
        for r in rows
    } == {
        (m, e, dt, h, c)
        for m in PLAN["models"]
        for e in PLAN["executors"]
        for dt in PLAN["timesteps_s"]
        for h in PLAN["hands"]
        for c in PLAN["controllers"]
    }
    for row in rows:
        o = row["outcome"]
        assert o["seed"] == PLAN["seed"] and o["offset_m"] == PLAN["offset_m"]
        assert o["maximum_endpoint_state_error"] == o["maximum_endpoint_sensor_error"] == 0
        assert o["passed"] == (not o["failures"])
    assert report["comparisons"] == [
        compare_group(rows, m, h, c)
        for m in PLAN["models"]
        for h in PLAN["hands"]
        for c in PLAN["controllers"]
    ]
    assert all(r["executor_evidence_exact"] for r in report["comparisons"])
    assert report["matched_timestep_model_contrasts"] == model_contrasts(rows)
    assert len(report["matched_timestep_model_contrasts"]) == 16
