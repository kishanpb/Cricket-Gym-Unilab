"""Retain every development failure and the fixed diagnostic selection."""

import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "g1_cricket_results/residual_v3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_complete_pool_and_unchanged_baseline():
    parent_path = ROOT / "g1_cricket_results/residual_v1/evaluation.json"
    parent = json.loads(parent_path.read_text())
    report = json.loads((DIRECTORY / "evaluation.json").read_text())
    assert report["parent_report_sha256"] == digest(parent_path)
    assert report["predecessor_v2_report_sha256"] == digest(
        ROOT / "g1_cricket_results/residual_v2/evaluation.json"
    )
    assert len(report["rows"]) == report["evaluation"]["expected_rows"] == 96
    assert report["evaluation"] == parent["evaluation"]
    for row, old in zip(report["rows"], parent["rows"], strict=True):
        assert all(row[k] == old[k] for k in ("hand", "controller", "seed", "offset_m"))
        assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
        if row["controller"] == "zero_residual":
            assert {k: v for k, v in row.items() if k != "return"} == {
                k: v for k, v in old.items() if k != "return"
            }
        else:
            assert not row["passed"] and row["seconds"] == 2
            assert "outgoing_velocity_not_above_1_m_s" in row["failures"]
            assert not any(row["guard_contact_physics_step_counts"].values())
    for hand, count in (("right", 16), ("left", 8)):
        rows = [r for r in report["rows"] if r["hand"] == hand and r["controller"] == "ppo"]
        assert sum(r["blade_contact_seen"] for r in rows) == count
    assert report["training"]["transitions"] == 199680
    assert report["training"]["seed"] == 1
    for name, expected in report["source_sha256"].items():
        assert digest(ROOT / name) == expected
    assert digest(ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert digest(DIRECTORY / "right" / f"{name}.json") == report[f"{name}_sha256"]
    with (DIRECTORY / "right/training_scalars.csv").open() as stream:
        assert [int(r["iteration"]) for r in csv.DictReader(stream)] == list(range(2080))


def test_diagnostic_contains_fixed_full_episodes_not_success_selection():
    report = json.loads((DIRECTORY / "evaluation.json").read_text())
    media = json.loads((DIRECTORY / "development_media.json").read_text())
    assert media["scope"] == "development_diagnostic_not_showcase_or_policy_promotion"
    assert media["evaluation_sha256"] == digest(DIRECTORY / "evaluation.json")
    assert media["checkpoint_sha256"] == report["checkpoint"]["sha256"]
    assert media["renderer_sha256"] == digest(ROOT / "scripts/render_g1_cricket_residual.py")
    assert media["video_sha256"] == digest(DIRECTORY / "development_diagnostic.mp4")
    assert media["sheet_sha256"] == digest(DIRECTORY / "development_contact_sheet.png")
    selected = [r for r in report["rows"] if r["controller"] == "ppo" and r["seed"] == 4301]
    assert len(selected) == len(media["clips"]) == 6
    for clip, row in zip(media["clips"], selected, strict=True):
        for key in ("hand", "offset_m", "seed", "passed", "failures"):
            assert clip[key] == row[key]
        assert clip["simulation_seconds"] == row["seconds"]
        assert clip["motion_frames"] == 200 and clip["hold_frames"] == 25
        assert clip["first_to_last_mean_pixel_change"] > 0
        assert len(clip["trajectory_sha256"]) == 64
    with Image.open(DIRECTORY / "development_contact_sheet.png") as sheet:
        assert sheet.size == (1440, 704)
    with imageio.get_reader(DIRECTORY / "development_diagnostic.mp4") as reader:
        assert reader.get_meta_data()["fps"] == 50
        count = 0
        for frame in reader:
            assert frame.shape == (704, 960, 3) and np.asarray(frame).std() > 10
            count += 1
    assert count == media["decoded_frames"] == 1350
