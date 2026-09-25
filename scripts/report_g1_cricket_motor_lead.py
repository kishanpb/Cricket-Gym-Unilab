"""Paired motor-reference timing experiment on the complete bounced-delivery pool."""

import argparse
import hashlib
import json
from pathlib import Path

import mjbatch.held_control
from report_g1_cricket_bimanual_contact import compare_resolution
from report_g1_cricket_bounced_delivery import summarize_bounced_row

ROOT = Path(__file__).resolve().parents[1]


def build_report(directory, *, candidate_frames=(1, 3)):
    if (
        not candidate_frames
        or len(set(candidate_frames)) != len(candidate_frames)
        or any(type(lead) is not int or lead <= 0 for lead in candidate_frames)
    ):
        raise ValueError("candidate frames must be distinct positive integers")
    leads = (0, *candidate_frames)
    rows, hashes = [], {}
    for lead in leads:
        for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
            for hand in ("right", "left"):
                path = directory / f"lead{lead}_{hand}_{resolution}" / "evaluation.json"
                report = json.loads(path.read_text())
                expected = {
                    "waist_tracking_gain": 1,
                    "root_position_gain": 4,
                    "lookahead_frames": lead,
                    "contact_dt": dt,
                    "soft_toss": False,
                    "bounced_delivery": True,
                    "compact_substeps": True,
                }
                if report["evaluation_overrides"] != expected:
                    raise ValueError("evaluation differs from the motor-lead protocol")
                hashes[str(path.relative_to(directory))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for source, digest in report["input_sha256"].items():
                    if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"evaluation input changed: {source}")
                runtime_hash = hashlib.sha256(
                    Path(mjbatch.held_control.__file__).read_bytes()
                ).hexdigest()
                if report["runtime_source_sha256"] != {"mjbatch.held_control": runtime_hash}:
                    raise ValueError("recorder source changed")
                if [row["controller"] for row in report["rows"]] != ["reference_only", "ppo"]:
                    raise ValueError("expected both controller rows")
                for row in report["rows"]:
                    if row["hand"] != hand:
                        raise ValueError("hand differs from declared case")
                    rows.append(
                        {
                            "lookahead_frames": lead,
                            "resolution": resolution,
                            **summarize_bounced_row(row),
                        }
                    )
    resolution_pairs, timing_pairs = [], []
    for hand in ("right", "left"):
        for controller in ("reference_only", "ppo"):
            matched = [
                row for row in rows if row["hand"] == hand and row["controller"] == controller
            ]
            for lead in leads:
                fine, finest = [row for row in matched if row["lookahead_frames"] == lead]
                resolution_pairs.append(
                    {
                        "hand": hand,
                        "controller": controller,
                        "lookahead_frames": lead,
                        **compare_resolution(fine, finest),
                    }
                )
            for resolution in ("fine", "finest"):
                baseline, *candidates = [row for row in matched if row["resolution"] == resolution]
                for candidate in candidates:
                    timing_pairs.append(
                        {
                            "hand": hand,
                            "controller": controller,
                            "resolution": resolution,
                            "lookahead_frames": candidate["lookahead_frames"],
                            "baseline_peak_bat_error_m": baseline["peak_bat_error_m"],
                            "candidate_peak_bat_error_m": candidate["peak_bat_error_m"],
                            "peak_bat_error_change_m": candidate["peak_bat_error_m"]
                            - baseline["peak_bat_error_m"],
                            "candidate_all_checks_pass": candidate["all_checks_pass"],
                        }
                    )
    return {
        "scope": "frozen_actor_motor_timing_not_learned_interception",
        "rows": rows,
        "resolution_comparisons": resolution_pairs,
        "timing_comparisons": timing_pairs,
        "candidate_qualification": {
            str(lead): all(
                row["all_checks_pass"] for row in rows if row["lookahead_frames"] == lead
            )
            and all(
                pair["all_pass"] for pair in resolution_pairs if pair["lookahead_frames"] == lead
            )
            for lead in candidate_frames
        },
        "input_sha256": hashes,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--candidate-frames", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    report = build_report(args.directory, candidate_frames=tuple(args.candidate_frames))
    (args.directory / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(report["rows"]),
                "candidate_qualification": report["candidate_qualification"],
            }
        )
    )
