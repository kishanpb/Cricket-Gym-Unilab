"""Evaluate the final fresh PPO checkpoint with physics-rate separation rewards."""

import copy
import itertools
import json

from evaluate_g1_cricket_impact_events import compare_row
from evaluate_g1_cricket_impact_resolution import IDENTITY, compare
from evaluate_g1_cricket_residual import ROOT, evaluate, sha256

DIRECTORY = ROOT / "g1_cricket_results/impact_events_v1"
RUN_DIR = DIRECTORY / "right"


def validate_training_config(saved, reference):
    expected = copy.deepcopy(reference)
    expected["env"]["mujoco_observe_substeps"] = True
    expected["reward"]["batting"]["func"] = (
        "unilab.tasks.manipulation.g1_cricket.substep_reward.SubstepForwardExitReward"
    )
    expected["training"]["log_dir"] = "g1_cricket_results/impact_events_v1/right"
    if saved != expected:
        raise ValueError("training config differs beyond event acquisition and log directory")
    assert saved["algo"]["resume"] is False and saved["algo"]["resume_path"] is None


def validate_pool(rows):
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    if len(rows) != 96 or {tuple(r[k] for k in IDENTITY) for r in rows} != expected:
        raise ValueError("evaluation must retain all 96 unique declared trials")


def validate_preflight(preflight, parent, audit):
    assert preflight["training_preflight_ready"] is True
    assert preflight["parent_report_sha256"] == audit["parent_report_sha256"]
    rows = preflight["report"]["rows"]
    validate_pool(rows)
    assert len(preflight["comparisons"]) == 96
    for old, new, evidence, comparison in zip(
        parent["reports"][1]["rows"], rows, audit["rows"], preflight["comparisons"], strict=True
    ):
        assert compare_row(old, new, evidence) == comparison


def apply_penetration_gate(report):
    validate_pool(report["rows"])
    for row in report["rows"]:
        row["original_shot_gate_passed"] = row["passed"]
        if row["maximum_blade_penetration_m"] > 0.006:
            row["failures"].append("blade_penetration_above_6_mm")
        row["passed"] = not row["failures"]
    for group in report["aggregates"]:
        rows = [r for r in report["rows"] if all(r[k] == group[k] for k in ("hand", "controller"))]
        assert group["total"] == len(rows) == 24
        group["passed"] = sum(r["passed"] for r in rows)
    report["evaluation"]["maximum_blade_penetration_m"] = 0.006
    report["evaluation"]["gate"] += "; maximum blade penetration <=6 mm"


def validate_baseline(report, reference, *, same_reward):
    baseline = [r for r in report["rows"] if r["controller"] == "zero_residual"]
    previous = [r for r in reference["rows"] if r["controller"] == "zero_residual"]
    assert len(baseline) == len(previous) == 48
    excluded = set() if same_reward else {"return"}
    assert [{k: v for k, v in r.items() if k not in excluded} for r in baseline] == [
        {k: v for k, v in r.items() if k not in excluded} for r in previous
    ], "zero-residual baseline changed"
    report["zero_residual_physical_rows_exactly_reproduced"] = 48
    report["zero_residual_returns_exactly_reproduced"] = same_reward
    report["baseline_return_comparison_scope"] = (
        "same repaired reward; exact native returns"
        if same_reward
        else "old reward; returns excluded, not evidence of learning improvement"
    )


def run():
    previous = ROOT / "g1_cricket_results/impact_v1"
    preflight_path = DIRECTORY / "preflight.json"
    parent_path = previous / "trained_evaluation.json"
    audit_path = previous / "impact_reward_audit.json"
    preflight, parent, audit = [
        json.loads(p.read_text()) for p in (preflight_path, parent_path, audit_path)
    ]
    assert preflight["parent_report_sha256"] == sha256(parent_path)
    assert preflight["reward_audit_sha256"] == sha256(audit_path)
    validate_preflight(preflight, parent, audit)
    sources = dict(preflight["report"]["source_sha256"])
    name = "scripts/evaluate_g1_cricket_impact_events_learning.py"
    sources[name] = sha256(ROOT / name)
    for name, expected in sources.items():
        assert sha256(ROOT / name) == expected, name
    saved = json.loads((RUN_DIR / "run_config.json").read_text())["config"]
    reference_hash = sha256(previous / "right/run_config.json")
    for report in (*parent["reports"], preflight["report"]):
        assert report["run_config_sha256"] == reference_hash
    reference = json.loads((previous / "right/run_config.json").read_text())["config"]
    validate_training_config(saved, reference)
    summary = json.loads((RUN_DIR / "run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/impact_events_v1/right/model_2079.pt"
    pinned = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (
            preflight_path,
            parent_path,
            audit_path,
            previous / "right/run_config.json",
            RUN_DIR / "run_config.json",
            RUN_DIR / "run_summary.json",
            ROOT / summary["last_checkpoint"],
        )
    }
    outputs = []
    for index, dt in enumerate((0.00025, 0.000125)):
        for name, expected in {**sources, **pinned}.items():
            assert sha256(ROOT / name) == expected, name
        report = evaluate(RUN_DIR, {"env": {"sim_dt": dt}})
        assert report["checkpoint"]["sha256"] == pinned[summary["last_checkpoint"]]
        for name in ("run_config", "run_summary"):
            assert (
                report[f"{name}_sha256"]
                == pinned[f"g1_cricket_results/impact_events_v1/right/{name}.json"]
            )
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == parent["reports"][index][key]
        apply_penetration_gate(report)
        validate_baseline(
            report,
            preflight["report"] if index else parent["reports"][0],
            same_reward=bool(index),
        )
        report["scope"] = "fresh_right_ppo_physics_rate_reward_development_not_promoted"
        report["source_sha256"].update(sources)
        outputs.append(report)
        print(json.dumps({"dt": dt, "aggregates": report["aggregates"]}), flush=True)
    for name, expected in {**sources, **pinned}.items():
        assert sha256(ROOT / name) == expected, name
    comparisons = [
        compare(a, b) for a, b in zip(outputs[0]["rows"], outputs[1]["rows"], strict=True)
    ]
    result = {
        "scope": "fresh_ppo_physics_rate_reward_complete_development_evaluation",
        "preflight_sha256": sha256(preflight_path),
        "parent_evaluation_sha256": sha256(parent_path),
        "input_sha256": pinned,
        "reports": outputs,
        "comparisons": comparisons,
        "impact_resolution_consistent": all(not c["failed_checks"] for c in comparisons),
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (DIRECTORY / "trained_evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    run()
