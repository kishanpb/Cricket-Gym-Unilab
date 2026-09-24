"""All frozen transfer outcomes survive the compliance correction audit."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

from scripts.g1_cricket_historical_sources import REVISION, legacy_source_digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_complete_frozen_transfer_and_explicit_pair_provenance():
    result = json.loads((ROOT / "g1_cricket_results/impact_v1/frozen_transfer.json").read_text())
    parent_path = ROOT / "g1_cricket_results/residual_v3/evaluation.json"
    parent = json.loads(parent_path.read_text())
    assert result["parent_report_sha256"] == digest(parent_path)
    assert result["legacy_source_revision"] == REVISION
    assert result["legacy_default_rows_exactly_reproduced"] == 96
    assert not result["physical_calibration_validated"] and not result["policy_promoted"]
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    all_rows = []
    for dt, report in zip((0.0005, 0.00025), result["reports"], strict=True):
        assert report["evaluation_overrides"] == {
            "training": {"task_name": "G1CricketImpact"},
            "env": {"sim_dt": dt},
        }
        assert report["checkpoint"] == parent["checkpoint"]
        assert digest(ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"]
        for name, expected_hash in report["source_sha256"].items():
            assert digest(ROOT / name) == expected_hash
        assert report["evaluation"]["physics_dt_seconds"] == dt
        assert report["evaluation"]["control_dt_seconds"] == 0.02
        assert report["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
        assert report["evaluation"]["maximum_blade_penetration_m"] == 0.006
        assert {m["hand"] for m in report["contact_models"]} == {"right", "left"}
        for model in report["contact_models"]:
            assert len(model["pairs"]) == 1
            pair = model["pairs"][0]
            assert set(pair["geoms"]) == {"ball_geom", "bat_blade"}
            assert pair["solref"] == [0.004, 1]
            assert pair["friction"] == [0.6, 0.6, 0.01, 0.001, 0.001]
            assert pair["dim"] == 3 and pair["margin"] == pair["gap"] == 0
        assert len(report["rows"]) == 96
        assert {
            (r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in report["rows"]
        } == expected
        for row in report["rows"]:
            assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
            assert row["passed"] == (not row["failures"])
            assert ("blade_penetration_above_6_mm" in row["failures"]) == (
                row["maximum_blade_penetration_m"] > 0.006
            )
            assert row["original_shot_gate_passed"] == (
                not [f for f in row["failures"] if f != "blade_penetration_above_6_mm"]
            )
        all_rows.extend(report["rows"])
    assert result["training_preflight_ready"] == all(
        r["maximum_blade_penetration_m"] <= 0.006
        and r["seconds"] == 2
        and not any(r["guard_contact_physics_step_counts"].values())
        and not set(r["failures"])
        & {
            "pelvis_height",
            "pelvis_orientation",
            "joint_limit",
            "actuator_limit",
            "native_episode_incomplete",
        }
        for r in all_rows
    )
    assert len(result["resolution_comparisons"]) == 96
    assert result["impact_resolution_consistent"] == all(
        not r["failed_checks"] for r in result["resolution_comparisons"]
    )


def test_historical_sources_do_not_silently_accept_other_changes(tmp_path):
    file = tmp_path / "unchanged.py"
    file.write_text("old")
    before = legacy_source_digest(tmp_path, "unchanged.py")
    file.write_text("new")
    assert legacy_source_digest(tmp_path, "unchanged.py") != before


def test_default_evaluator_keeps_legacy_row_schema(monkeypatch):
    import evaluate_g1_cricket_residual as evaluator

    monkeypatch.setattr(evaluator, "SEEDS", (4301,))
    monkeypatch.setattr(evaluator, "TOSS_OFFSETS", (0.0,))
    directory = ROOT / "g1_cricket_results/residual_v3"
    report = evaluator.evaluate(directory / "right")
    parent = json.loads((directory / "evaluation.json").read_text())
    expected = [r for r in parent["rows"] if r["seed"] == 4301 and r["offset_m"] == 0]
    assert report["rows"] == expected
    assert all("maximum_blade_penetration_m" not in row for row in report["rows"])
