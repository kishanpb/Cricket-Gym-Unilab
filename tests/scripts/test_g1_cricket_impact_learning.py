"""The corrected-contact PPO result retains the full declared comparison."""

import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_learning import validate_preflight, validate_training_config
from evaluate_g1_cricket_impact_resolution import IDENTITY, compare


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("algo", "resume", True),
        ("algo", "resume_path", "earlier.pt"),
        ("algo", "max_iterations", 4000),
        ("env", "handedness", "left"),
        ("env", "sim_dt", 0.0005),
        ("reward", "batting", {}),
    ],
)
def test_fresh_training_config_rejects_undeclared_changes(section, key, value):
    reference = json.loads(
        (ROOT / "g1_cricket_results/residual_v3/right/run_config.json").read_text()
    )["config"]
    cfg = copy.deepcopy(reference)
    cfg["env"]["sim_dt"] = 0.00025
    cfg["training"]["task_name"] = "G1CricketImpact"
    cfg["training"]["log_dir"] = "g1_cricket_results/impact_v1/right"
    validate_training_config(cfg, reference)
    cfg[section][key] = value
    with pytest.raises(ValueError, match="training config differs"):
        validate_training_config(cfg, reference)


@pytest.mark.parametrize(
    "key,value",
    [
        ("maximum_blade_penetration_m", 0.0061),
        ("seconds", 1.9),
        ("guard_contact_physics_step_counts", {"bat_pitch": 1}),
        ("failures", ["joint_limit"]),
        ("maximum_endpoint_state_error", 0.001),
    ],
)
def test_training_preflight_rejects_physical_failures(key, value):
    extension = json.loads(
        (ROOT / "g1_cricket_results/impact_v1/resolution_extension.json").read_text()
    )
    validate_preflight(extension)
    extension["report"]["rows"][0][key] = value
    with pytest.raises(ValueError, match="physical preflight failed"):
        validate_preflight(extension)


def test_complete_corrected_contact_learning_result():
    directory = ROOT / "g1_cricket_results/impact_v1"
    result = json.loads((directory / "trained_evaluation.json").read_text())
    parent = json.loads((directory / "frozen_transfer.json").read_text())
    extension = json.loads((directory / "resolution_extension.json").read_text())
    saved = json.loads((directory / "right/run_config.json").read_text())["config"]
    reference_config = json.loads(
        (ROOT / "g1_cricket_results/residual_v3/right/run_config.json").read_text()
    )["config"]
    validate_training_config(saved, reference_config)
    summary = json.loads((directory / "right/run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/impact_v1/right/model_2079.pt"
    assert result["frozen_transfer_sha256"] == digest(directory / "frozen_transfer.json")
    assert result["resolution_extension_sha256"] == digest(directory / "resolution_extension.json")
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    for report, reference, dt in zip(
        result["reports"],
        (parent["reports"][1], extension["report"]),
        (0.00025, 0.000125),
        strict=True,
    ):
        assert report["training"]["transitions"] == 199680
        assert report["evaluation"]["physics_dt_seconds"] == dt
        assert report["evaluation"]["control_dt_seconds"] == 0.02
        assert report["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
        assert report["evaluation"]["maximum_blade_penetration_m"] == 0.006
        assert report["evaluation"]["left_hand"] == "untrained_transfer"
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == reference[key]
        assert report["checkpoint"]["sha256"] == digest(ROOT / report["checkpoint"]["path"])
        for name, expected_hash in report["source_sha256"].items():
            assert digest(ROOT / name) == expected_hash
        for name in ("run_config", "run_summary"):
            assert digest(directory / "right" / f"{name}.json") == report[f"{name}_sha256"]
        assert len(report["rows"]) == 96
        assert {tuple(row[k] for k in IDENTITY) for row in report["rows"]} == expected
        baseline = [r for r in report["rows"] if r["controller"] == "zero_residual"]
        assert len(baseline) == report["zero_residual_rows_exactly_reproduced"] == 48
        assert baseline == [r for r in reference["rows"] if r["controller"] == "zero_residual"]
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
    diagnostics = json.loads((directory / "right/training_diagnostics.json").read_text())
    assert all(v["all_finite"] for v in diagnostics["scalar_tags"].values())
    for tag in ("Loss/value", "Loss/surrogate", "Train/mean_reward", "reward/batting"):
        assert diagnostics["scalar_tags"][tag]["count"] == 2080
