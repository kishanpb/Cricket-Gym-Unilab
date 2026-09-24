"""Full frozen-policy comparison before learning against corrected compliance."""

import json

from evaluate_g1_cricket_residual import ROOT, evaluate, sha256
from g1_cricket_historical_sources import REVISION, legacy_source_digest


def run():
    parent_dir = ROOT / "g1_cricket_results/residual_v3"
    parent = json.loads((parent_dir / "evaluation.json").read_text())
    for name, expected in parent["source_sha256"].items():
        if legacy_source_digest(ROOT, name) != expected:
            raise ValueError(f"parent source changed: {name}")
    assert sha256(ROOT / parent["checkpoint"]["path"]) == parent["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert sha256(parent_dir / "right" / f"{name}.json") == parent[f"{name}_sha256"]
    original = evaluate(parent_dir / "right")
    assert len(original["rows"]) == len(parent["rows"]) == 96
    for row, old in zip(original["rows"], parent["rows"], strict=True):
        assert {k: v for k, v in row.items() if k != "maximum_blade_penetration_m"} == old
    outputs = []
    for dt in (0.0005, 0.00025):
        overrides = {"training": {"task_name": "G1CricketImpact"}, "env": {"sim_dt": dt}}
        report = evaluate(parent_dir / "right", overrides)
        report["scope"] = "frozen_v3_policy_transfer_to_impact_v1_not_retrained_or_promoted"
        for row in report["rows"]:
            row["original_shot_gate_passed"] = row["passed"]
            if row["maximum_blade_penetration_m"] > 0.006:
                row["failures"].append("blade_penetration_above_6_mm")
            row["passed"] = not row["failures"]
        for group in report["aggregates"]:
            group["passed"] = sum(
                r["passed"]
                for r in report["rows"]
                if all(r[k] == group[k] for k in ("hand", "controller"))
            )
        report["evaluation"]["maximum_blade_penetration_m"] = 0.006
        report["evaluation"]["gate"] += "; maximum blade penetration <=6 mm"
        for name in (
            "src/unilab/tasks/manipulation/g1_cricket/impact.py",
            "src/unilab/tasks/manipulation/g1_cricket/__init__.py",
            "src/unilab/tasks/manipulation/g1_cricket/forward_exit_reward.py",
            "src/unilab/tasks/manipulation/g1_cricket/separation_reward.py",
            "src/unilab/conf/ppo/task/g1_cricket_impact_v1/mujoco.yaml",
            "src/unilab/conf/ppo/task/g1_cricket_residual_v2/mujoco.yaml",
            "src/unilab/conf/ppo/task/g1_cricket_residual_v3/mujoco.yaml",
            "scripts/evaluate_g1_cricket_impact.py",
            "scripts/g1_cricket_historical_sources.py",
            "docs/g1_cricket_impact_v1.md",
        ):
            report["source_sha256"][name] = sha256(ROOT / name)
        outputs.append(report)
        print(json.dumps({"dt": dt, "aggregates": report["aggregates"]}), flush=True)
    comparisons = []
    for a, b in zip(outputs[0]["rows"], outputs[1]["rows"], strict=True):
        identity = {k: a[k] for k in ("hand", "controller", "offset_m", "seed")}
        assert all(b[k] == v for k, v in identity.items())
        failed = []
        for key in ("blade_contact_seen", "failures", "seconds"):
            if a[key] != b[key]:
                failed.append(key)
        for key, tolerance in (
            ("maximum_blade_penetration_m", 0.0001),
            ("first_separation_ball_vx_m_s", 0.05),
        ):
            x, y = a[key], b[key]
            if x is None or y is None:
                if x != y:
                    failed.append(key)
            elif abs(x - y) > max(tolerance, abs(y) * 0.05):
                failed.append(key)
        x = a["ball_contact_peak_force_norm_n"].get("bat_blade", 0)
        y = b["ball_contact_peak_force_norm_n"].get("bat_blade", 0)
        if abs(x - y) > max(1, abs(y) * 0.05):
            failed.append("blade_peak_force")
        comparisons.append({**identity, "failed_checks": failed})
    result = {
        "scope": "versioned_contact_correction_frozen_policy_preflight",
        "parent_report_sha256": sha256(parent_dir / "evaluation.json"),
        "legacy_default_rows_exactly_reproduced": 96,
        "legacy_source_revision": REVISION,
        "physical_calibration_validated": False,
        "reports": outputs,
        "resolution_comparisons": comparisons,
        "training_preflight_ready": all(
            r["maximum_blade_penetration_m"] <= 0.006
            and r["seconds"] == 2
            and not any(r["guard_contact_physics_step_counts"].values())
            and not any(
                x in r["failures"]
                for x in (
                    "pelvis_height",
                    "pelvis_orientation",
                    "joint_limit",
                    "actuator_limit",
                    "native_episode_incomplete",
                )
            )
            for report in outputs
            for r in report["rows"]
        ),
        "impact_resolution_consistent": all(not r["failed_checks"] for r in comparisons),
        "policy_promoted": False,
    }
    directory = ROOT / "g1_cricket_results/impact_v1"
    directory.mkdir(exist_ok=True)
    (directory / "frozen_transfer.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    run()
