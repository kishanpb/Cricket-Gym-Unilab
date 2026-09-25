"""Complete fixed one-bounce delivery comparison, not learned interception."""

import argparse
import hashlib
import json
from pathlib import Path

import mjbatch.held_control
from report_g1_cricket_bimanual_contact import compare_resolution, create_media, summarize_row

ROOT = Path(__file__).resolve().parents[1]


def summarize_bounced_row(row):
    result = summarize_row(row)
    sequence = row["trace"][-1]["substep_audit"]["ball_contact_sequence"]
    first = sequence["first_blade_contact"]
    before = [
        event
        for event in sequence["pitch_events"]
        if first is not None and event["start_s"] < first["time_s"]
    ]
    completed = [
        event
        for event in before
        if "end_s" in event
        and event["end_s"] < first["time_s"]
        and event["incoming_velocity_m_s"][2] < 0
        and event["outgoing_velocity_m_s"][2] > 0
    ]
    checks = {
        "exactly_one_completed_bounce_before_blade": len(before) == len(completed) == 1,
        "bounce_in_front_on_pitch": len(completed) == 1
        and completed[0]["position_m"][0] > 0
        and abs(completed[0]["position_m"][1]) < 1.52,
        "incoming_at_blade": first is not None and first["incoming_velocity_m_s"][0] < 0,
    }
    return {
        **result,
        "ball_contact_sequence": sequence,
        "delivery_checks": checks,
        "all_checks_pass": result["all_checks_pass"] and all(checks.values()),
    }


def build_report(directory, *, include_finest=False):
    rows, hashes = [], {}
    resolutions = [("coarse", 0.0000625), ("fine", 0.00003125)]
    if include_finest:
        resolutions.append(("finest", 0.000015625))
    for resolution, dt in resolutions:
        for hand in ("right", "left"):
            path = directory / f"{hand}_{resolution}" / "evaluation.json"
            report = json.loads(path.read_text())
            expected = {
                "waist_tracking_gain": 1,
                "root_position_gain": 4,
                "contact_dt": dt,
                "soft_toss": False,
                "bounced_delivery": True,
                "compact_substeps": True,
            }
            if report["evaluation_overrides"] != expected:
                raise ValueError("evaluation differs from the fixed bounced-delivery protocol")
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for source, digest in report["input_sha256"].items():
                if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"evaluation input changed: {source}")
            recorder_hash = hashlib.sha256(
                Path(mjbatch.held_control.__file__).read_bytes()
            ).hexdigest()
            if report["runtime_source_sha256"] != {"mjbatch.held_control": recorder_hash}:
                raise ValueError("recorder source changed")
            if [r["controller"] for r in report["rows"]] != ["reference_only", "ppo"]:
                raise ValueError("expected both controller rows")
            for row in report["rows"]:
                if row["hand"] != hand:
                    raise ValueError("hand differs from declared case")
                rows.append({"resolution": resolution, **summarize_bounced_row(row)})
    comparisons = []
    for hand in ("right", "left"):
        for controller in ("reference_only", "ppo"):
            matched = [r for r in rows if r["hand"] == hand and r["controller"] == controller]
            for coarse, fine in zip(matched, matched[1:]):
                comparisons.append(
                    {
                        "hand": hand,
                        "controller": controller,
                        "resolutions": [coarse["resolution"], fine["resolution"]],
                        **compare_resolution(coarse, fine),
                    }
                )
    return {
        "scope": "complete_frozen_actor_one_bounce_diagnostic_not_learned_interception",
        "rows": rows,
        "resolution_comparisons": comparisons,
        "all_checks_pass": all(row["all_checks_pass"] for row in rows)
        and all(pair["all_pass"] for pair in comparisons),
        "input_sha256": hashes,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--include-finest", action="store_true")
    args = parser.parse_args()
    result = build_report(args.directory, include_finest=args.include_finest)
    if args.media:
        result["media"] = create_media(
            args.directory, suffix="_fine", name="bounced_delivery", indices=(0, 52, 69, 89, 149)
        )
    filename = "resolution_refinement.json" if args.include_finest else "summary.json"
    (args.directory / filename).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(result["rows"]), "all_checks_pass": result["all_checks_pass"]}))
