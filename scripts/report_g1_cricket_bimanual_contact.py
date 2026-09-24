"""Summarize every predeclared soft-toss row without selecting successful clips."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize_row(row):
    trace = row["trace"]
    audits = [frame["substep_audit"] for frame in trace]
    peaks = {key: max(audit["peaks"][key] for audit in audits) for key in audits[0]["peaks"]}
    contacts = {}
    for audit in audits:
        for pair, values in audit["ball_contacts"].items():
            total = contacts.setdefault(pair, dict.fromkeys(values, 0.0))
            for key, value in values.items():
                total[key] = (
                    total[key] + value
                    if key in ("duration_s", "normal_impulse_ns")
                    else max(total[key], value)
                )
    exit_state = audits[-1]["first_force_free_blade_exit"]
    checks = {
        "complete": not row["terminated"]
        and row["steps"] == 150
        and "time_out" in row["terminal_terms"],
        "pelvis_height": min(a["minimum_pelvis_height_m"] for a in audits) > 0.65,
        "no_unexpected_contact": not any(a["unexpected_contacts"] for a in audits),
        "hard_joint_limits": peaks["hard_joint_limit_excess_rad"] <= 0.0001,
        "grip": peaks["grip_separation_m"] < 0.006,
        "motor_limits": peaks["motor_force_fraction"] <= 1,
        "root_tracking": max(f["root_translation_error_m"] for f in trace) < 0.15,
        "joint_tracking": max(f["reference_joint_rmse_rad"] for f in trace) < 0.2,
        "bat_tracking": max(f["bat_tracking_error_m"] for f in trace) < 0.08,
    }
    contact_checks = {
        "loaded_blade_contact": contacts.get("ball_geom/bat_blade", {}).get("normal_impulse_ns", 0)
        > 0,
        "outgoing_vx": exit_state is not None and exit_state["velocity_m_s"][0] > 1,
        "all_ball_penetration": max((v["penetration_m"] for v in contacts.values()), default=0)
        <= 0.006,
    }
    return {
        "hand": row["hand"],
        "controller": row["controller"],
        "return": row["return"],
        "duration_s": trace[-1]["time_s"],
        "terminal_terms": row["terminal_terms"],
        "original_checks": checks,
        "contact_checks": contact_checks,
        "all_checks_pass": all(checks.values()) and all(contact_checks.values()),
        "peak_bat_error_m": max(f["bat_tracking_error_m"] for f in trace),
        "peaks": peaks,
        "ball_contacts": contacts,
        "first_force_free_blade_exit": exit_state,
        "unexpected_contacts": sorted({p for a in audits for p in a["unexpected_contacts"]}),
    }


def compare_resolution(coarse, fine):
    checks = {}
    for pair in sorted(coarse["ball_contacts"].keys() | fine["ball_contacts"].keys()):
        a, b = coarse["ball_contacts"].get(pair), fine["ball_contacts"].get(pair)
        checks[pair + ":present_both"] = a is not None and b is not None
        if a is None or b is None:
            continue
        for key, tolerance in (("peak_force_n", 1.0), ("penetration_m", 0.0001)):
            checks[pair + ":" + key] = abs(a[key] - b[key]) <= max(0.05 * b[key], tolerance)
    a, b = coarse["first_force_free_blade_exit"], fine["first_force_free_blade_exit"]
    checks["blade_exit_both"] = a is not None and b is not None
    if a is not None and b is not None:
        av, bv = np.asarray(a["velocity_m_s"]), np.asarray(b["velocity_m_s"])
        checks["blade_exit_velocity"] = bool(
            np.linalg.norm(av - bv) <= max(0.05 * np.linalg.norm(bv), 0.05)
        )
    return {"checks": checks, "all_pass": all(checks.values())}


def build_report(directory):
    rows, hashes = [], {}
    for resolution in ("coarse", "fine"):
        for condition in ("dry", "toss"):
            for hand in ("right", "left"):
                path = directory / f"{hand}_{resolution}_{condition}" / "evaluation.json"
                report = json.loads(path.read_text())
                expected = {
                    "waist_tracking_gain": 1,
                    "root_position_gain": 4,
                    "contact_dt": 0.0000625 if resolution == "coarse" else 0.00003125,
                    "soft_toss": condition == "toss",
                }
                if report["evaluation_overrides"] != expected:
                    raise ValueError("evaluation differs from the fixed contact protocol")
                hashes[str(path.relative_to(directory))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for source, digest in report["input_sha256"].items():
                    if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"evaluation input changed: {source}")
                if [r["controller"] for r in report["rows"]] != ["reference_only", "ppo"]:
                    raise ValueError("expected both complete controller rows")
                for row in report["rows"]:
                    if row["hand"] != hand:
                        raise ValueError("hand differs from predeclared case")
                    rows.append(
                        {"resolution": resolution, "condition": condition, **summarize_row(row)}
                    )
    comparisons = []
    for hand in ("right", "left"):
        for controller in ("reference_only", "ppo"):
            pair = [
                r
                for r in rows
                if r["hand"] == hand and r["controller"] == controller and r["condition"] == "toss"
            ]
            comparisons.append(
                {"hand": hand, "controller": controller, **compare_resolution(*pair)}
            )
    return {
        "scope": "complete_frozen_actor_development_pool_not_learned_interception",
        "row_count": len(rows),
        "rows": rows,
        "resolution_comparisons": comparisons,
        "all_toss_checks_pass": all(r["all_checks_pass"] for r in rows if r["condition"] == "toss"),
        "all_resolution_comparisons_pass": all(r["all_pass"] for r in comparisons),
        "input_sha256": hashes,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def create_media(directory):
    import imageio.v2 as imageio
    from PIL import Image

    combined = directory / "two_hand_ppo_soft_toss.mp4"
    sheet = Image.new("RGB", (1920, 432))
    indices = (0, 56, 69, 89, 149)
    with imageio.get_writer(combined, fps=25, macro_block_size=1) as writer:
        for row, hand in enumerate(("right", "left")):
            path = directory / f"{hand}_fine_toss" / "ppo_diagnostic.mp4"
            count = 0
            with imageio.get_reader(path) as reader:
                for index, frame in enumerate(reader):
                    if np.std(frame) < 5:
                        raise ValueError(f"blank video frame: {path.name}")
                    writer.append_data(frame)
                    count += 1
                    if index in indices:
                        tile = Image.fromarray(frame).resize((384, 216))
                        sheet.paste(tile, (384 * indices.index(index), 216 * row))
            if count != 175:
                raise ValueError(f"incomplete video: {path.name}")
    count = 0
    with imageio.get_reader(combined) as reader:
        for frame in reader:
            if np.std(frame) < 5:
                raise ValueError("blank combined video frame")
            count += 1
    if count != 350:
        raise ValueError("combined video failed decode validation")
    sheet.save(directory / "soft_toss_contact_sheet.png")
    return {
        "combined_frames": count,
        "fps": 25,
        "physical_speed": 0.5,
        "contact_sheet_times_s": [0.02, 1.14, 1.40, 1.80, 3.00],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--media", action="store_true")
    args = parser.parse_args()
    result = build_report(args.directory)
    if args.media:
        result["media"] = create_media(args.directory)
    (args.directory / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
