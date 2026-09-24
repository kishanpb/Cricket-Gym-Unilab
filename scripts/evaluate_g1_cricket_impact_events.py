"""Frozen-policy preflight: repair event capture without changing any physical row."""

import json

import numpy as np
from evaluate_g1_cricket_residual import ROOT, evaluate, sha256

SOURCES = (
    "scripts/evaluate_g1_cricket_impact_events.py",
    "docs/g1_cricket_impact_events_v1.md",
    "docs/sphinx/source/adr/ADR-0010-experimental-substep-observation.md",
    "src/unilab/base/backend_substeps.py",
    "src/unilab/base/mujoco_substeps.py",
    "src/unilab/base/base.py",
    "src/unilab/base/backend_factory.py",
    "src/unilab/base/np_env.py",
    "src/unilab/tasks/manipulation/g1_cricket/substep_reward.py",
    "src/unilab/conf/ppo/task/g1_cricket_impact_events_v1/mujoco.yaml",
)


def compare_row(old, new, audit):
    for key in ("hand", "controller", "offset_m", "seed"):
        assert old[key] == new[key] == audit[key]
    assert {k: v for k, v in old.items() if k != "return"} == {
        k: v for k, v in new.items() if k != "return"
    }, "frozen policy physical row changed"
    samples = [
        dict(zip(audit["control_sample_fields"], s, strict=True)) for s in audit["control_samples"]
    ]
    previous_bonus = sum(s["separation_step_reward"] for s in samples)
    vx = old["first_separation_ball_vx_m_s"]
    bonus = float(5 * (1 + np.tanh(vx - 1))) if vx is not None else 0.0
    expected_return = old["return"] - previous_bonus + bonus
    np.testing.assert_allclose(new["return"], expected_return, atol=1e-5, rtol=0)
    return {
        **{key: old[key] for key in ("hand", "controller", "offset_m", "seed")},
        "non_return_fields_exact": True,
        "old_return": old["return"],
        "new_return": new["return"],
        "old_separation_bonus": previous_bonus,
        "new_separation_bonus": bonus,
        "expected_return": expected_return,
        "return_residual": new["return"] - expected_return,
        "new_first_event": vx is not None and audit["paid_separation_events"] == 0,
    }


def run():
    previous = ROOT / "g1_cricket_results/impact_v1"
    parent_path = previous / "trained_evaluation.json"
    audit_path = previous / "impact_reward_audit.json"
    audit = json.loads(audit_path.read_text())
    parent = json.loads(parent_path.read_text())["reports"][1]
    assert audit["parent_report_sha256"] == sha256(parent_path)
    sources = {**audit["source_sha256"], **{p: sha256(ROOT / p) for p in SOURCES}}
    for path, expected in sources.items():
        assert sha256(ROOT / path) == expected, path
    report = evaluate(
        previous / "right",
        {
            "env": {"sim_dt": 0.000125, "mujoco_observe_substeps": True},
            "reward": {
                "batting": {
                    "func": "unilab.tasks.manipulation.g1_cricket.substep_reward.SubstepForwardExitReward"
                }
            },
        },
    )
    for key in (
        "checkpoint",
        "versions",
        "external_asset_sha256",
        "contact_models",
        "run_config_sha256",
        "run_summary_sha256",
    ):
        assert report[key] == parent[key], key
    assert len(report["rows"]) == len(parent["rows"]) == len(audit["rows"]) == 96
    comparisons = []
    for old, new, audited in zip(parent["rows"], report["rows"], audit["rows"], strict=True):
        new["original_shot_gate_passed"] = new["passed"]
        if new["maximum_blade_penetration_m"] > 0.006:
            new["failures"].append("blade_penetration_above_6_mm")
        new["passed"] = not new["failures"]
        comparisons.append(compare_row(old, new, audited))
    for path, expected in sources.items():
        assert sha256(ROOT / path) == expected, path
    report["source_sha256"].update(sources)
    report["evaluation"]["maximum_blade_penetration_m"] = 0.006
    report["evaluation"]["gate"] += "; maximum blade penetration <=6 mm"
    report["scope"] = "frozen_policy_substep_reward_capture_preflight_not_learning"
    result = {
        "scope": report["scope"],
        "parent_report_sha256": sha256(parent_path),
        "reward_audit_sha256": sha256(audit_path),
        "report": report,
        "comparisons": comparisons,
        "training_preflight_ready": True,
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    output = ROOT / "g1_cricket_results/impact_events_v1/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(comparisons),
                "recovered_events": sum(c["new_first_event"] for c in comparisons),
                "maximum_return_residual": max(abs(c["return_residual"]) for c in comparisons),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
