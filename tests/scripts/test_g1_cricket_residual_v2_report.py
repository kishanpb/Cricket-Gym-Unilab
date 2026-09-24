"""V2 remains a rejected reward experiment on the unchanged development pool."""

import csv
import hashlib
import json
from pathlib import Path

from scripts.g1_cricket_historical_sources import legacy_source_digest

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_report_preserves_comparator_and_every_failed_trial():
    directory = ROOT / "g1_cricket_results/residual_v2"
    parent_path = ROOT / "g1_cricket_results/residual_v1/evaluation.json"
    parent = json.loads(parent_path.read_text())
    report = json.loads((directory / "evaluation.json").read_text())
    assert report["parent_report_sha256"] == digest(parent_path)
    assert len(report["rows"]) == report["evaluation"]["expected_rows"] == 96
    for row, old in zip(report["rows"], parent["rows"], strict=True):
        assert all(row[k] == old[k] for k in ("hand", "controller", "seed", "offset_m"))
        assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
        if row["controller"] == "zero_residual":
            assert {k: v for k, v in row.items() if k != "return"} == {
                k: v for k, v in old.items() if k != "return"
            }
        else:
            assert not row["blade_contact_seen"] and not row["passed"]
            assert "no_blade_contact" in row["failures"]
            if row["hand"] == "left":
                assert row["seconds"] < 2
                assert row["guard_contact_physics_step_counts"]["bat_robot_left_hip_yaw_link"] > 0
            else:
                assert row["seconds"] == 2
    assert report["training"]["transitions"] == 199680
    assert report["training"]["seed"] == 1
    assert report["evaluation"] == parent["evaluation"]
    for name, expected in report["source_sha256"].items():
        assert legacy_source_digest(ROOT, name) == expected
    assert digest(ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"]
    assert report["checkpoint"]["sha256"] != parent["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert digest(directory / "right" / f"{name}.json") == report[f"{name}_sha256"]
    with (directory / "right/training_scalars.csv").open() as stream:
        scalars = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in scalars] == list(range(2080))
