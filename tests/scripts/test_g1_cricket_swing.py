"""Reward-only training and all-row physical evidence stay independently checkable."""

import copy
import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_events_learning import validate_baseline, validate_pool
from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_residual import sha256
from evaluate_g1_cricket_swing import DIRECTORY, PARENT, check_hashes, validate_config


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("algo", "resume", True),
        ("algo", "resume_path", "earlier.pt"),
        ("algo", "max_iterations", 4000),
        ("env", "handedness", "left"),
        ("env", "sim_dt", 0.0005),
        ("env", "mujoco_observe_substeps", False),
        ("reward", "failure", {}),
    ],
)
def test_config_rejects_other_axes(section, key, value):
    reference = json.loads((PARENT / "right/run_config.json").read_text())["config"]
    config = copy.deepcopy(reference)
    config["reward"]["batting"]["func"] = (
        "unilab.tasks.manipulation.g1_cricket.swing_reward.ForwardSwingReward"
    )
    config["training"]["log_dir"] = "g1_cricket_results/swing_v1/right"
    validate_config(config, reference)
    config[section][key] = value
    with pytest.raises(ValueError, match="swing config differs"):
        validate_config(config, reference)


def test_pretraining_contract():
    contract = json.loads((DIRECTORY / "preflight.json").read_text())
    check_hashes({**contract["input_sha256"], **contract["source_sha256"]})
    validate_config(
        contract["config"], json.loads((PARENT / "right/run_config.json").read_text())["config"]
    )
    assert contract["policy_promoted"] is False


def test_complete_learning_result():
    result = json.loads((DIRECTORY / "trained_evaluation.json").read_text())
    preflight = json.loads((DIRECTORY / "preflight.json").read_text())
    parent = json.loads((PARENT / "trained_evaluation.json").read_text())
    saved = json.loads((DIRECTORY / "right/run_config.json").read_text())["config"]
    assert saved == preflight["config"]
    summary = json.loads((DIRECTORY / "right/run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/swing_v1/right/model_2079.pt"
    check_hashes(result["input_sha256"])
    assert result["input_sha256"]["g1_cricket_results/swing_v1/preflight.json"] == sha256(
        DIRECTORY / "preflight.json"
    )
    for index, (report, dt) in enumerate(zip(result["reports"], (0.00025, 0.000125), strict=True)):
        validate_pool(report["rows"])
        before = copy.deepcopy(report)
        validate_baseline(report, parent["reports"][index], same_reward=False)
        assert report == before
        assert report["training"]["transitions"] == 199680
        assert report["evaluation"]["physics_dt_seconds"] == dt
        assert report["evaluation"]["control_dt_seconds"] == 0.02
        assert report["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
        assert report["evaluation"]["maximum_blade_penetration_m"] == 0.006
        assert report["evaluation"]["left_hand"] == "untrained_transfer"
        assert report["checkpoint"]["path"] == summary["last_checkpoint"]
        assert report["checkpoint"]["sha256"] == sha256(ROOT / summary["last_checkpoint"])
        check_hashes(report["source_sha256"])
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == parent["reports"][index][key]
        for name in ("run_config", "run_summary"):
            assert report[f"{name}_sha256"] == sha256(DIRECTORY / "right" / f"{name}.json")
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
    with (DIRECTORY / "right/training_scalars.csv").open() as stream:
        scalars = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in scalars] == list(range(2080))
    for tag, observed in diagnostics["scalar_tags"].items():
        values = [float(row[tag]) for row in scalars if row[tag] != ""]
        assert all(math.isfinite(v) for v in values)
        assert observed == {
            "count": len(values),
            "all_finite": True,
            "minimum": min(values),
            "maximum": max(values),
            "final": values[-1],
        }
        if tag in ("Loss/value", "Loss/surrogate", "Train/mean_reward", "reward/batting"):
            assert len(values) == 2080
