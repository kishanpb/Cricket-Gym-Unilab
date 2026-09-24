"""The reward-acquisition experiment must retain its full physical comparison."""

import copy
import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_events_learning import (
    apply_penetration_gate,
    validate_baseline,
    validate_pool,
    validate_preflight,
    validate_training_config,
)
from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_residual import sha256

DIRECTORY = ROOT / "g1_cricket_results/impact_events_v1"
PREVIOUS = ROOT / "g1_cricket_results/impact_v1"


def parent():
    return json.loads((PREVIOUS / "trained_evaluation.json").read_text())


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("algo", "resume", True),
        ("algo", "resume_path", "earlier.pt"),
        ("algo", "max_iterations", 4000),
        ("env", "handedness", "left"),
        ("env", "sim_dt", 0.0005),
        ("env", "mujoco_observe_substeps", False),
        ("reward", "batting", {}),
    ],
)
def test_training_config_rejects_other_axes(section, key, value):
    reference = json.loads((PREVIOUS / "right/run_config.json").read_text())["config"]
    config = copy.deepcopy(reference)
    config["env"]["mujoco_observe_substeps"] = True
    config["reward"]["batting"]["func"] = (
        "unilab.tasks.manipulation.g1_cricket.substep_reward.SubstepForwardExitReward"
    )
    config["training"]["log_dir"] = "g1_cricket_results/impact_events_v1/right"
    validate_training_config(config, reference)
    config[section][key] = value
    with pytest.raises(ValueError, match="training config differs"):
        validate_training_config(config, reference)


def test_pool_rejects_missing_and_duplicate_trials():
    rows = parent()["reports"][0]["rows"]
    validate_pool(rows)
    with pytest.raises(ValueError, match="96 unique"):
        validate_pool(rows[:-1])
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="96 unique"):
        validate_pool(rows)


def test_preflight_rejects_unexplained_reward_change():
    preflight = json.loads((DIRECTORY / "preflight.json").read_text())
    audit = json.loads((PREVIOUS / "impact_reward_audit.json").read_text())
    validate_preflight(preflight, parent(), audit)
    preflight["report"]["rows"][0]["return"] += 0.001
    with pytest.raises(AssertionError):
        validate_preflight(preflight, parent(), audit)


def test_baseline_separates_reward_change_from_physics():
    reference = parent()["reports"][0]
    report = copy.deepcopy(reference)
    report["rows"][0]["return"] += 0.1
    validate_baseline(report, reference, same_reward=False)
    with pytest.raises(AssertionError, match="baseline changed"):
        validate_baseline(report, reference, same_reward=True)
    report["rows"][0]["minimum_pelvis_height_m"] -= 1e-10
    with pytest.raises(AssertionError, match="baseline changed"):
        validate_baseline(report, reference, same_reward=False)


@pytest.mark.parametrize("penetration,passed", [(0.006, True), (0.006000001, False)])
def test_penetration_gate_preserves_rows_and_recomputes_counts(penetration, passed):
    report = parent()["reports"][0]
    report["rows"][0].update(maximum_blade_penetration_m=penetration, passed=True, failures=[])
    apply_penetration_gate(report)
    assert len(report["rows"]) == 96
    assert report["rows"][0]["original_shot_gate_passed"] is True
    assert report["rows"][0]["passed"] is passed
    assert report["aggregates"][0]["passed"] == int(passed)


def test_complete_learning_result():
    result = json.loads((DIRECTORY / "trained_evaluation.json").read_text())
    preflight = json.loads((DIRECTORY / "preflight.json").read_text())
    previous = parent()
    saved = json.loads((DIRECTORY / "right/run_config.json").read_text())["config"]
    reference = json.loads((PREVIOUS / "right/run_config.json").read_text())["config"]
    validate_training_config(saved, reference)
    summary = json.loads((DIRECTORY / "right/run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/impact_events_v1/right/model_2079.pt"
    assert result["preflight_sha256"] == sha256(DIRECTORY / "preflight.json")
    assert result["parent_evaluation_sha256"] == sha256(PREVIOUS / "trained_evaluation.json")
    for path, expected in result["input_sha256"].items():
        assert sha256(ROOT / path) == expected
    for report in (*previous["reports"], preflight["report"]):
        assert report["run_config_sha256"] == sha256(PREVIOUS / "right/run_config.json")
    for index, (report, dt) in enumerate(zip(result["reports"], (0.00025, 0.000125), strict=True)):
        validate_pool(report["rows"])
        assert report["zero_residual_physical_rows_exactly_reproduced"] == 48
        assert report["zero_residual_returns_exactly_reproduced"] is bool(index)
        saved_report = copy.deepcopy(report)
        validate_baseline(
            report,
            preflight["report"] if index else previous["reports"][0],
            same_reward=bool(index),
        )
        assert report == saved_report
        assert report["training"]["transitions"] == 199680
        assert report["evaluation"]["physics_dt_seconds"] == dt
        assert report["evaluation"]["control_dt_seconds"] == 0.02
        assert report["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
        assert report["evaluation"]["maximum_blade_penetration_m"] == 0.006
        assert report["evaluation"]["left_hand"] == "untrained_transfer"
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == previous["reports"][index][key]
        assert report["checkpoint"]["path"] == summary["last_checkpoint"]
        assert report["checkpoint"]["sha256"] == sha256(ROOT / summary["last_checkpoint"])
        for name in ("run_config", "run_summary"):
            assert report[f"{name}_sha256"] == sha256(DIRECTORY / "right" / f"{name}.json")
        for path, expected in report["source_sha256"].items():
            assert sha256(ROOT / path) == expected
        for row in report["rows"]:
            assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
            assert row["passed"] == (not row["failures"])
            assert ("blade_penetration_above_6_mm" in row["failures"]) == (
                row["maximum_blade_penetration_m"] > 0.006
            )
            assert row["original_shot_gate_passed"] == (
                not [f for f in row["failures"] if f != "blade_penetration_above_6_mm"]
            )
        for group in report["aggregates"]:
            rows = [
                r for r in report["rows"] if all(r[k] == group[k] for k in ("hand", "controller"))
            ]
            assert group["total"] == len(rows) == 24
            assert group["passed"] == sum(r["passed"] for r in rows)
    comparisons = [
        compare(a, b)
        for a, b in zip(result["reports"][0]["rows"], result["reports"][1]["rows"], strict=True)
    ]
    assert result["comparisons"] == comparisons
    assert result["impact_resolution_consistent"] == all(
        not c["failed_checks"] for c in comparisons
    )
    assert result["physical_calibration_validated"] is result["policy_promoted"] is False
    diagnostics = json.loads((DIRECTORY / "right/training_diagnostics.json").read_text())
    assert all(v["all_finite"] for v in diagnostics["scalar_tags"].values())
    with (DIRECTORY / "right/training_scalars.csv").open() as stream:
        scalars = list(csv.DictReader(stream))
    assert [int(r["iteration"]) for r in scalars] == list(range(2080))
    for tag in ("Loss/value", "Loss/surrogate", "Train/mean_reward", "reward/batting"):
        values = [float(row[tag]) for row in scalars]
        assert all(math.isfinite(value) for value in values)
        assert diagnostics["scalar_tags"][tag] == {
            "count": 2080,
            "all_finite": True,
            "minimum": min(values),
            "maximum": max(values),
            "final": values[-1],
        }
