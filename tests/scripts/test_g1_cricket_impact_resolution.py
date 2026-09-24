"""Complete finer-step comparison with unchanged provenance and thresholds."""

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_resolution import IDENTITY, compare


def row():
    return dict(
        hand="right",
        controller="ppo",
        offset_m=0.0,
        seed=4301,
        blade_contact_seen=True,
        failures=[],
        seconds=2,
        maximum_blade_penetration_m=0.004,
        first_separation_ball_vx_m_s=0.5,
        ball_contact_peak_force_norm_n={"bat_blade": 200.0},
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("blade_contact_seen", False),
        ("failures", ["no_blade_contact"]),
        ("seconds", 1.9),
        ("maximum_blade_penetration_m", 0.0043),
        ("first_separation_ball_vx_m_s", None),
        ("ball_contact_peak_force_norm_n", {"bat_blade": 211.0}),
    ],
)
def test_comparison_rejects_each_changed_gate(key, value):
    fine = row()
    coarse = {**fine, key: value}
    expected = "blade_peak_force_n" if key == "ball_contact_peak_force_norm_n" else key
    assert compare(coarse, fine)["failed_checks"] == [expected]


def test_comparison_preserves_tolerances_and_identity():
    fine = row()
    coarse = copy.deepcopy(fine)
    coarse["maximum_blade_penetration_m"] += 0.00019
    coarse["first_separation_ball_vx_m_s"] += 0.049
    coarse["ball_contact_peak_force_norm_n"]["bat_blade"] += 9.9
    assert not compare(coarse, fine)["failed_checks"]
    assert compare(coarse, fine)["differences"]["blade_peak_force_n"]["tolerance"] == 10
    with pytest.raises(ValueError, match="trial identity"):
        compare({**coarse, "seed": 4302}, fine)
    missing = {**fine, "first_separation_ball_vx_m_s": None}
    assert not compare(missing, missing)["failed_checks"]


def test_complete_resolution_extension():
    directory = ROOT / "g1_cricket_results/impact_v1"
    parent_path = directory / "frozen_transfer.json"
    parent = json.loads(parent_path.read_text())
    result = json.loads((directory / "resolution_extension.json").read_text())
    assert result["parent_report_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
    coarse, fine = parent["reports"][1], result["report"]
    assert (
        result["coarse_physics_dt_seconds"] == coarse["evaluation"]["physics_dt_seconds"] == 0.00025
    )
    assert result["fine_physics_dt_seconds"] == fine["evaluation"]["physics_dt_seconds"] == 0.000125
    for key in (
        "checkpoint",
        "run_config_sha256",
        "run_summary_sha256",
        "external_asset_sha256",
        "versions",
        "contact_models",
    ):
        assert coarse[key] == fine[key]
    for name, expected in fine["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    checkpoint = ROOT / fine["checkpoint"]["path"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == fine["checkpoint"]["sha256"]
    assert fine["evaluation_overrides"] == {
        "training": {"task_name": "G1CricketImpact"},
        "env": {"sim_dt": 0.000125},
    }
    assert fine["evaluation"]["control_dt_seconds"] == 0.02
    assert fine["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
    assert fine["evaluation"]["maximum_blade_penetration_m"] == 0.006
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    assert len(fine["rows"]) == 96
    assert {tuple(r[key] for key in IDENTITY) for r in fine["rows"]} == expected
    for r in fine["rows"]:
        assert r["maximum_endpoint_state_error"] == r["maximum_endpoint_sensor_error"] == 0
        assert r["passed"] == (not r["failures"])
        assert ("blade_penetration_above_6_mm" in r["failures"]) == (
            r["maximum_blade_penetration_m"] > 0.006
        )
        assert r["original_shot_gate_passed"] == (
            not [f for f in r["failures"] if f != "blade_penetration_above_6_mm"]
        )
    comparisons = [compare(a, b) for a, b in zip(coarse["rows"], fine["rows"], strict=True)]
    assert result["comparisons"] == comparisons
    assert result["consistent_pairs"] == sum(not c["failed_checks"] for c in comparisons)
    assert result["contact_bearing_pairs"] == sum(c["contact_bearing"] for c in comparisons)
    assert result["consistent_contact_bearing_pairs"] == sum(
        c["contact_bearing"] and not c["failed_checks"] for c in comparisons
    )
    assert result["impact_resolution_consistent"] == all(
        not c["failed_checks"] for c in comparisons
    )
    assert result["physical_calibration_validated"] is result["policy_promoted"] is False
