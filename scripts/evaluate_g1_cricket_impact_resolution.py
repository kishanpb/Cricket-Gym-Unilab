"""Extend the complete frozen-policy impact audit to one finer physics step."""

import json

from evaluate_g1_cricket_residual import ROOT, evaluate, sha256

IDENTITY = ("hand", "controller", "offset_m", "seed")
TOLERANCES = {
    "maximum_blade_penetration_m": 0.0001,
    "first_separation_ball_vx_m_s": 0.05,
    "blade_peak_force_n": 1.0,
}


def compare(coarse, fine):
    identity = {key: coarse[key] for key in IDENTITY}
    if any(fine[key] != value for key, value in identity.items()):
        raise ValueError("paired trial identity differs")
    failures, differences = [], {}
    for key in ("blade_contact_seen", "failures", "seconds"):
        if coarse[key] != fine[key]:
            failures.append(key)
    for key, absolute in TOLERANCES.items():
        if key == "blade_peak_force_n":
            a, b = (
                row["ball_contact_peak_force_norm_n"].get("bat_blade", 0) for row in (coarse, fine)
            )
        else:
            a, b = coarse[key], fine[key]
        if a is None or b is None:
            differences[key] = {"coarse": a, "fine": b, "absolute_difference": None}
            if a != b:
                failures.append(key)
        else:
            tolerance = max(absolute, abs(b) * 0.05)
            differences[key] = {
                "coarse": a,
                "fine": b,
                "absolute_difference": abs(a - b),
                "tolerance": tolerance,
            }
            if abs(a - b) > tolerance:
                failures.append(key)
    return {
        **identity,
        "contact_bearing": coarse["blade_contact_seen"] or fine["blade_contact_seen"],
        "failed_checks": failures,
        "differences": differences,
    }


def run():
    directory = ROOT / "g1_cricket_results/impact_v1"
    parent_path = directory / "frozen_transfer.json"
    parent = json.loads(parent_path.read_text())
    coarse = parent["reports"][1]
    assert coarse["evaluation"]["physics_dt_seconds"] == 0.00025
    for name, expected in coarse["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"parent source changed: {name}")
    run_dir = ROOT / "g1_cricket_results/residual_v3/right"
    assert sha256(run_dir.parent / "evaluation.json") == parent["parent_report_sha256"]
    assert sha256(ROOT / coarse["checkpoint"]["path"]) == coarse["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert sha256(run_dir / f"{name}.json") == coarse[f"{name}_sha256"]
    overrides = {"training": {"task_name": "G1CricketImpact"}, "env": {"sim_dt": 0.000125}}
    fine = evaluate(run_dir, overrides)
    for key in (
        "checkpoint",
        "run_config_sha256",
        "run_summary_sha256",
        "external_asset_sha256",
        "versions",
        "contact_models",
    ):
        assert fine[key] == coarse[key], f"comparison provenance changed: {key}"
    fine["scope"] = "frozen_v3_policy_transfer_to_impact_v1_not_retrained_or_promoted"
    for row in fine["rows"]:
        row["original_shot_gate_passed"] = row["passed"]
        if row["maximum_blade_penetration_m"] > 0.006:
            row["failures"].append("blade_penetration_above_6_mm")
        row["passed"] = not row["failures"]
    for group in fine["aggregates"]:
        group["passed"] = sum(
            row["passed"]
            for row in fine["rows"]
            if all(row[key] == group[key] for key in ("hand", "controller"))
        )
    fine["evaluation"]["maximum_blade_penetration_m"] = 0.006
    fine["evaluation"]["gate"] += "; maximum blade penetration <=6 mm"
    fine["source_sha256"].update(coarse["source_sha256"])
    for name in (
        "scripts/evaluate_g1_cricket_impact_resolution.py",
        "docs/g1_cricket_impact_resolution.md",
    ):
        fine["source_sha256"][name] = sha256(ROOT / name)
    assert len(coarse["rows"]) == len(fine["rows"]) == 96
    comparisons = [compare(a, b) for a, b in zip(coarse["rows"], fine["rows"], strict=True)]
    result = {
        "scope": "complete_frozen_policy_resolution_extension_not_training",
        "parent_report_sha256": sha256(parent_path),
        "coarse_physics_dt_seconds": 0.00025,
        "fine_physics_dt_seconds": 0.000125,
        "report": fine,
        "comparisons": comparisons,
        "consistent_pairs": sum(not row["failed_checks"] for row in comparisons),
        "contact_bearing_pairs": sum(row["contact_bearing"] for row in comparisons),
        "consistent_contact_bearing_pairs": sum(
            row["contact_bearing"] and not row["failed_checks"] for row in comparisons
        ),
        "impact_resolution_consistent": all(not row["failed_checks"] for row in comparisons),
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (directory / "resolution_extension.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("report", "comparisons")}
        )
    )


if __name__ == "__main__":
    run()
