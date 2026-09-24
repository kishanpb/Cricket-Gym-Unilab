"""Prefix diagnostics preserve phase, complete coverage and parent provenance."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_g1_cricket_impact_headroom import (
    MotionReplay,
    audit_trial,
    summarize_window,
    window_slice,
)
from evaluate_g1_cricket_impact_events_learning import validate_pool
from evaluate_g1_cricket_impact_resolution import IDENTITY
from evaluate_g1_cricket_residual import load_policy, sha256
from g1_cricket_archival_sources import check_retained_hashes

from unilab.base.config_adapter import BackendAdapter, create_env

DIRECTORY = ROOT / "g1_cricket_results/impact_events_v1"


@pytest.mark.parametrize("contact_step", [5, 960, 1005])
def test_window_excludes_impact_and_weights_partial_intervals(contact_step):
    indices = np.concatenate(
        [
            np.arange(tick * 160, (tick + 1) * 160)[window_slice(tick, 160, contact_step, 800)]
            for tick in range(contact_step // 160 + 2)
        ]
    )
    np.testing.assert_array_equal(indices, np.arange(max(0, contact_step - 800), contact_step))
    raw = np.repeat((indices // 160 == 1).astype(float)[:, None] * 2, 7, axis=1)
    stats = summarize_window(raw, np.zeros_like(raw), np.zeros_like(raw), np.zeros_like(raw))
    assert stats["raw_action_clipped_fraction"] == [float((indices // 160 == 1).mean())] * 7


def test_headroom_statistics_distinguish_clipping_and_at_bound():
    raw = np.repeat(np.array([[2.0], [1.0], [0.0]]), 7, axis=1)
    error = np.repeat(np.array([[1.0], [2.0], [3.0]]), 7, axis=1)
    force = np.repeat(np.array([[0.5], [1.0], [0.9]]), 7, axis=1)
    stats = summarize_window(raw, error, -error, force)
    assert stats["physics_samples"] == 3
    assert stats["raw_action_clipped_fraction"] == [1 / 3] * 7
    assert stats["residual_at_bound_fraction"] == [2 / 3] * 7
    assert stats["actuator_at_limit_fraction"] == [1 / 3] * 7
    assert stats["actuator_force_peak_fraction"] == [1] * 7
    np.testing.assert_allclose(stats["target_error_rms_rad"], np.sqrt(14 / 3))
    assert stats["joint_velocity_peak_abs_rad_s"] == [3] * 7


@pytest.mark.parametrize("hand,controller", [("right", "ppo"), ("left", "zero_residual")])
def test_real_g1_prefix_reproduces_parent_first_impact(hand, controller):
    parent = json.loads((DIRECTORY / "trained_evaluation.json").read_text())["reports"][1]
    reference = next(
        r
        for r in parent["rows"]
        if (r["hand"], r["controller"], r["offset_m"], r["seed"]) == (hand, controller, 0.0, 4301)
    )
    owner = OmegaConf.create(
        json.loads((DIRECTORY / "right/run_config.json").read_text())["config"]
    )
    owner.env.sim_dt = 0.000125
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    try:
        wrapped, policy = load_policy(owner, env, ROOT / parent["checkpoint"]["path"])
        row = audit_trial(env, wrapped, policy, MotionReplay(env), reference)
        diagnostic = row["diagnostic"]
        assert row["status"] == "first_impact_prefix_reproduced"
        assert diagnostic["maximum_endpoint_state_error"] == 0
        assert diagnostic["maximum_endpoint_sensor_error"] == 0
        assert diagnostic["precontact_statistics"]["physics_samples"] == 800
        assert diagnostic["first_loaded_impact"] is not None
    finally:
        env.close()


def test_misses_remain_unavailable_not_zero_velocity():
    reference = {
        "hand": "left",
        "controller": "ppo",
        "offset_m": 0.0,
        "seed": 4301,
        "blade_contact_seen": False,
        "seconds": 0.22,
        "failures": ["guarded_contact"],
    }
    row = audit_trial(None, None, None, None, reference)
    assert row["diagnostic"] is None
    assert row["parent_failures"] == ["guarded_contact"] and row["parent_seconds"] == 0.22


def test_complete_headroom_report():
    result = json.loads((DIRECTORY / "impact_headroom.json").read_text())
    parent_path = DIRECTORY / "trained_evaluation.json"
    parent = json.loads(parent_path.read_text())["reports"][1]
    assert result["parent_report_sha256"] == sha256(parent_path)
    check_retained_hashes(ROOT, result["input_sha256"])
    assert result["checkpoint"] == parent["checkpoint"]
    assert result["versions"] == parent["versions"]
    assert result["external_asset_sha256"] == parent["external_asset_sha256"]
    validate_pool(result["rows"])
    assert sum(r["diagnostic"] is not None for r in result["rows"]) == 40
    for row, original in zip(result["rows"], parent["rows"], strict=True):
        assert all(row[k] == original[k] for k in IDENTITY)
        assert row["parent_failures"] == original["failures"]
        assert row["parent_seconds"] == original["seconds"]
        if not original["blade_contact_seen"]:
            assert row["diagnostic"] is None
            continue
        d = row["diagnostic"]
        episode = d["first_episode"]
        start = next(e for e in original["blade_events"] if e["event"] == "blade_contact_start")
        end = next(e for e in original["blade_events"] if e["event"] == "first_blade_separation")
        assert episode["start"]["integrated_seconds"] == start["seconds"]
        assert episode["end"]["integrated_seconds"] == end["seconds"]
        assert episode["end"]["post_ball_velocity_world_m_s"][0] == end["ball_vx_m_s"]
        assert d["maximum_endpoint_state_error"] == d["maximum_endpoint_sensor_error"] == 0
        assert d["window_seconds"] == 0.1 and len(d["arm_joints"]) == 7
        assert any(c["normal_force_n"] > 0 for c in d["first_loaded_impact"]["contacts"])
        traces = d["precontact_control_samples"]
        indices = [
            i for s in traces for i in range(s["first_solve_substep"], s["last_solve_substep"] + 1)
        ]
        first = episode["start"]["solve_substep"]
        assert indices == list(range(first - 800, first))
        raw = np.concatenate([np.tile(s["raw_action"], (s["physics_samples"], 1)) for s in traces])
        stats = d["precontact_statistics"]
        assert stats["physics_samples"] == sum(s["physics_samples"] for s in traces) == 800
        np.testing.assert_array_equal(
            stats["raw_action_clipped_fraction"], (np.abs(raw) > 1).mean(0)
        )
        np.testing.assert_array_equal(
            stats["residual_at_bound_fraction"], (np.abs(raw) >= 1).mean(0)
        )
    assert result["policy_promoted"] is result["physical_calibration_validated"] is False
