"""Complete-pool checks prevent contact reward from being promoted as a shot."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

from scripts.g1_cricket_historical_sources import legacy_source_digest

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_residual_report_retains_full_pool_and_failed_shot_gate():
    directory = ROOT / "g1_cricket_results/residual_v1"
    report = json.loads((directory / "evaluation.json").read_text())
    rows = report["rows"]
    expected = set(
        itertools.product(
            ("right", "left"), ("zero_residual", "ppo"), (-0.12, -0.1, 0.0), range(4301, 4309)
        )
    )
    assert len(rows) == len(expected) == report["evaluation"]["expected_rows"] == 96
    assert {(r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in rows} == expected
    for row in rows:
        assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
        assert row["passed"] == (not row["failures"])
        if row["passed"]:
            assert row["first_ball_contact"]["geoms"] == ["bat_blade"]
            assert row["first_separation_ball_vx_m_s"] > 1
            assert row["seconds"] == 2
            assert not any(row["guard_contact_physics_step_counts"].values())
            assert row["minimum_pelvis_height_m"] >= 0.48
            assert row["minimum_pelvis_up_z"] >= 0.65
            assert row["maximum_joint_limit_excess_rad"] <= 1e-6
            assert row["maximum_actuator_limit_fraction"] <= 1 + 1e-6
        if row["controller"] == "ppo":
            assert not row["passed"]
            assert "outgoing_velocity_not_above_1_m_s" in row["failures"]
            assert row["blade_contact_seen"] == (row["hand"] == "right")
    for group in report["aggregates"]:
        selected = [
            r for r in rows if r["hand"] == group["hand"] and r["controller"] == group["controller"]
        ]
        assert len(selected) == group["total"] == 24
        assert sum(r["passed"] for r in selected) == group["passed"]
    assert report["evaluation"]["left_hand"] == "untrained_transfer"
    assert digest(ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"]
    for name, value in report["source_sha256"].items():
        assert legacy_source_digest(ROOT, name) == value
    for name in ("run_config", "run_summary"):
        assert digest(directory / "right" / f"{name}.json") == report[f"{name}_sha256"]
    summary = json.loads((directory / "right/run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert report["training"]["transitions"] == summary["run_env_steps"] == 199680
    assert report["training"]["seed"] == 1
    assert report["training"]["torch_threads"] == 2
    with (directory / "right/training_scalars.csv").open() as stream:
        scalars = list(csv.DictReader(stream))
    assert [int(r["iteration"]) for r in scalars] == list(range(2080))
    diagnostics = json.loads((directory / "right/training_diagnostics.json").read_text())
    assert all(r["all_finite"] for r in diagnostics["scalar_tags"].values())
