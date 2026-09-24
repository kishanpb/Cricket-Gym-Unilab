"""Full two-resolution evaluation of fresh PPO trained on the impact-v1 model."""

import copy
import itertools
import json

from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_residual import ROOT, evaluate, sha256


def validate_training_config(saved, reference):
    expected = copy.deepcopy(reference)
    expected["env"]["sim_dt"] = 0.00025
    expected["training"]["task_name"] = "G1CricketImpact"
    expected["training"]["log_dir"] = "g1_cricket_results/impact_v1/right"
    if saved != expected:
        raise ValueError(
            "training config differs beyond declared contact task, timestep and log directory"
        )
    assert saved["algo"]["resume"] is False and saved["algo"]["resume_path"] is None


def validate_preflight(extension):
    report = extension["report"]
    assert (
        sha256(ROOT / "g1_cricket_results/impact_v1/frozen_transfer.json")
        == extension["parent_report_sha256"]
    )
    assert sha256(ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert (
            sha256(ROOT / "g1_cricket_results/residual_v3/right" / f"{name}.json")
            == report[f"{name}_sha256"]
        )
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    assert len(report["rows"]) == 96
    assert {
        (r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in report["rows"]
    } == expected
    for row in report["rows"]:
        if (
            row["maximum_blade_penetration_m"] > 0.006
            or row["seconds"] != 2
            or any(row["guard_contact_physics_step_counts"].values())
            or set(row["failures"])
            & {
                "pelvis_height",
                "pelvis_orientation",
                "joint_limit",
                "actuator_limit",
                "native_episode_incomplete",
            }
            or row["maximum_endpoint_state_error"] != 0
            or row["maximum_endpoint_sensor_error"] != 0
        ):
            raise ValueError("fine-resolution physical preflight failed")
    for name, expected_hash in report["source_sha256"].items():
        if sha256(ROOT / name) != expected_hash:
            raise ValueError(f"preflight source changed: {name}")


def run():
    directory = ROOT / "g1_cricket_results/impact_v1"
    parent_path = directory / "frozen_transfer.json"
    extension_path = directory / "resolution_extension.json"
    parent = json.loads(parent_path.read_text())
    extension = json.loads(extension_path.read_text())
    assert extension["parent_report_sha256"] == sha256(parent_path)
    validate_preflight(extension)
    references = (parent["reports"][1], extension["report"])
    run_dir = directory / "right"
    summary = json.loads((run_dir / "run_summary.json").read_text())
    saved = json.loads((run_dir / "run_config.json").read_text())["config"]
    reference_config = json.loads(
        (ROOT / "g1_cricket_results/residual_v3/right/run_config.json").read_text()
    )["config"]
    validate_training_config(saved, reference_config)
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert saved["training"]["task_name"] == "G1CricketImpact"
    assert saved["env"]["sim_dt"] == 0.00025 and saved["env"]["handedness"] == "right"
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/impact_v1/right/model_2079.pt"
    outputs = []
    for reference, dt in zip(references, (0.00025, 0.000125), strict=True):
        assert reference["evaluation"]["physics_dt_seconds"] == dt
        for name, expected in reference["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"reference source changed: {name}")
        report = evaluate(
            run_dir, {"training": {"task_name": "G1CricketImpact"}, "env": {"sim_dt": dt}}
        )
        report["scope"] = "fresh_right_ppo_corrected_contact_development_not_promoted"
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == reference[key]
        for row in report["rows"]:
            row["original_shot_gate_passed"] = row["passed"]
            if row["maximum_blade_penetration_m"] > 0.006:
                row["failures"].append("blade_penetration_above_6_mm")
            row["passed"] = not row["failures"]
        assert len(report["rows"]) == len(reference["rows"]) == 96
        baseline = [r for r in report["rows"] if r["controller"] == "zero_residual"]
        assert baseline == [r for r in reference["rows"] if r["controller"] == "zero_residual"]
        report["zero_residual_rows_exactly_reproduced"] = len(baseline)
        for group in report["aggregates"]:
            group["passed"] = sum(
                r["passed"]
                for r in report["rows"]
                if all(r[k] == group[k] for k in ("hand", "controller"))
            )
        report["evaluation"]["maximum_blade_penetration_m"] = 0.006
        report["evaluation"]["gate"] += "; maximum blade penetration <=6 mm"
        report["source_sha256"].update(reference["source_sha256"])
        for name in (
            "scripts/evaluate_g1_cricket_impact_learning.py",
            "docs/g1_cricket_impact_learning_v1.md",
        ):
            report["source_sha256"][name] = sha256(ROOT / name)
        outputs.append(report)
        print(json.dumps({"dt": dt, "aggregates": report["aggregates"]}), flush=True)
    comparisons = [
        compare(a, b) for a, b in zip(outputs[0]["rows"], outputs[1]["rows"], strict=True)
    ]
    result = {
        "scope": "fresh_ppo_corrected_contact_complete_development_evaluation",
        "frozen_transfer_sha256": sha256(parent_path),
        "resolution_extension_sha256": sha256(extension_path),
        "reports": outputs,
        "comparisons": comparisons,
        "impact_resolution_consistent": all(not c["failed_checks"] for c in comparisons),
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (directory / "trained_evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    run()
