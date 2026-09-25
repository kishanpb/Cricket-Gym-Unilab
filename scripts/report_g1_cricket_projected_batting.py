"""Compare all frozen-controller outcomes against the unchanged bat trajectory."""

import argparse
import hashlib
import json
from pathlib import Path

import mjbatch.held_control
from report_g1_cricket_bimanual_contact import compare_resolution
from report_g1_cricket_bounced_delivery import summarize_bounced_row

ROOT = Path(__file__).resolve().parents[1]


def build_report(directory):
    rows, hashes = [], {}
    runtime_hash = hashlib.sha256(Path(mjbatch.held_control.__file__).read_bytes()).hexdigest()
    for variant in ("baseline", "projected"):
        for hand in ("right", "left"):
            checkpoint = (
                ROOT / f"g1_cricket_results/bimanual_batting_learning_v1/ppo_{hand}/model_255.pt"
            )
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
                path = directory / f"{variant}_{hand}_{resolution}/evaluation.json"
                evaluation = json.loads(path.read_text())
                expected = dict(
                    contact_dt=dt, soft_toss=False, bounced_delivery=True, compact_substeps=True
                )
                scope = "ball_contact_observed_ppo_nominal_bounced_delivery_not_held_out"
                if variant == "projected":
                    expected["reference_directory"] = str(directory.resolve().relative_to(ROOT))
                    scope = "frozen_ball_observed_ppo_projected_reference_not_retrained"
                if evaluation["evaluation_overrides"] != expected or evaluation["scope"] != scope:
                    raise ValueError("evaluation differs from the reference-projection protocol")
                if evaluation["checkpoint_sha256"] != checkpoint_hash:
                    raise ValueError("evaluation must use the frozen final checkpoint")
                for source, digest in evaluation["input_sha256"].items():
                    if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"evaluation input changed: {source}")
                if evaluation["runtime_source_sha256"] != {"mjbatch.held_control": runtime_hash}:
                    raise ValueError("recorder source changed")
                if [row["controller"] for row in evaluation["rows"]] != ["reference_only", "ppo"]:
                    raise ValueError("both reference and PPO controls are required")
                hashes[str(path.relative_to(directory))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for row in evaluation["rows"]:
                    if row["hand"] != hand:
                        raise ValueError("evaluation hand differs from declared case")
                    rows.append(
                        dict(variant=variant, resolution=resolution, **summarize_bounced_row(row))
                    )
    resolution_pairs, comparisons = [], []
    for hand in ("right", "left"):
        for controller in ("reference_only", "ppo"):
            matched = [
                row for row in rows if row["hand"] == hand and row["controller"] == controller
            ]
            for variant in ("baseline", "projected"):
                pair = [row for row in matched if row["variant"] == variant]
                resolution_pairs.append(
                    dict(
                        hand=hand,
                        controller=controller,
                        variant=variant,
                        **compare_resolution(*pair),
                    )
                )
            for resolution in ("fine", "finest"):
                baseline, candidate = [row for row in matched if row["resolution"] == resolution]
                comparisons.append(
                    dict(
                        hand=hand,
                        controller=controller,
                        resolution=resolution,
                        baseline_peak_bat_error_m=baseline["peak_bat_error_m"],
                        projected_peak_bat_error_m=candidate["peak_bat_error_m"],
                        peak_bat_error_change_m=candidate["peak_bat_error_m"]
                        - baseline["peak_bat_error_m"],
                    )
                )
    return {
        "scope": "frozen_actor_reference_projection_not_retrained_not_held_out",
        "rows": rows,
        "resolution_comparisons": resolution_pairs,
        "reference_comparisons": comparisons,
        "projected_qualification": all(
            row["all_checks_pass"] for row in rows if row["variant"] == "projected"
        )
        and all(pair["all_pass"] for pair in resolution_pairs if pair["variant"] == "projected"),
        "input_sha256": hashes,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = build_report(args.directory)
    (args.directory / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(report["rows"]),
                "projected_qualification": report["projected_qualification"],
            }
        )
    )
