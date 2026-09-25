"""Verify the complete approach pilot and retain full-outcome media diagnostics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from evaluate_g1_cricket_approach_learning import HANDS, qualification
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_media(directory, row):
    path = directory / f"{row['hand']}_ppo_approach.mp4"
    with np.load(directory / row["trace"]) as trace:
        count = len(trace["states"])
    indices = sorted({min(i, count - 1) for i in (0, 50, 100, 200, 400)})
    sheet = Image.new("RGB", (384 * len(indices), 216))
    decoded, minimum_std = 0, float("inf")
    with imageio.get_reader(path) as reader:
        fps = reader.get_meta_data()["fps"]
        for index, frame in enumerate(reader):
            assert frame.shape == (540, 960, 3)
            minimum_std = min(minimum_std, float(np.std(frame)))
            if index in indices:
                sheet.paste(
                    Image.fromarray(frame).resize((384, 216)), (384 * indices.index(index), 0)
                )
            decoded += 1
    assert decoded == count and fps == 25 and minimum_std > 5
    sheet.save(directory / f"{row['hand']}_ppo_approach_contact_sheet.png")
    return dict(
        frames=decoded,
        fps=fps,
        physical_speed=0.5,
        minimum_frame_std=minimum_std,
        sheet_physics_times_s=[i * 0.02 for i in indices],
        selection="0/1/2/4/8 seconds, clamped to the final frame on early termination",
    )


def report(directory):
    evaluation = directory / "evaluation"
    result = json.loads((evaluation / "summary.json").read_text())
    for name, expected in result["input_sha256"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in result["artifact_sha256"].items():
        assert digest(evaluation / name) == expected, name
    assert len(result["rows"]) == 8
    gates = qualification(result["rows"])
    assert all(result[key] == value for key, value in gates.items())
    training, media = {}, {}
    for hand in HANDS:
        run = directory / f"ppo_{hand}"
        config = json.loads((run / "run_config.json").read_text())
        summary = json.loads((run / "run_summary.json").read_text())
        assert summary["status"] == "completed" and summary["total_env_steps"] == 49152
        assert config["run"]["effective_seed"] == 1
        with (run / "training_scalars.csv").open() as stream:
            scalars = list(csv.DictReader(stream))
        assert [int(row["iteration"]) for row in scalars] == list(range(256))
        diagnostics = json.loads((run / "training_diagnostics.json").read_text())
        assert all(tag["all_finite"] for tag in diagnostics["scalar_tags"].values())
        training[hand] = dict(
            source=config["run"]["git"],
            seed=1,
            transitions=summary["total_env_steps"],
            iterations=len(scalars),
            scalar_tags=len(diagnostics["scalar_tags"]),
            checkpoint=summary["last_checkpoint"],
            wall_time_s=summary["training_wall_time_sec"],
        )
        row = next(
            r
            for r in result["rows"]
            if r["hand"] == hand and r["controller"] == "ppo" and r["resolution"] == "finest"
        )
        media[hand] = (
            dict(status=row["video_status"])
            if row.get("video_status") == "blocked_nonfinite_states"
            else validate_media(evaluation, row)
        )
    return dict(
        scope=result["scope"],
        training=training,
        **gates,
        rows=result["rows"],
        media=media,
        evaluator_summary="evaluation/summary.json",
        reporter_sha256=digest(Path(__file__)),
        artifact_sha256={
            str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "summary.json"
        },
        guard=result["guard"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = report(args.directory)
    (args.directory / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {key: result[key] for key in ("training", "ppo_approach_qualified", "media")}, indent=2
        )
    )
