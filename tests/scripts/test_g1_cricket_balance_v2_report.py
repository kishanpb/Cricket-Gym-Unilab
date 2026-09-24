"""Guarded balance evidence keeps contact failures distinct from physical falls."""

import csv
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "version,source_commit",
    [
        ("balance_v2", "f326872e55a6551de17834a65b1ca07eea56d73d"),
        ("balance_v3", "00057f30"),
    ],
)
def test_guarded_balance_evidence(version, source_commit):
    directory = ROOT / "g1_cricket_results" / version / "right"
    report = json.loads((directory / "evaluation.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["run_env_steps"] == report["training"]["actual_transitions"] == 199680
    assert report["training"]["updates"] == 2080
    assert report["training"]["training_hand"] == "right"
    assert report["training"]["torch_threads"] == report["training"]["python_cpu_count"] == 2
    assert report["checkpoint"]["actor_dimensions"] == [130, 64, 64, 29]
    checkpoint = ROOT / report["checkpoint"]["path"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint"]["sha256"]
    assert (
        hashlib.sha256((directory / "run_config.json").read_bytes()).hexdigest()
        == report["run_config_sha256"]
    )
    for name, digest in report["source_hashes"].items():
        frozen_source = subprocess.check_output(
            ["git", "show", f"{source_commit}:{name}"], cwd=ROOT
        )
        assert hashlib.sha256(frozen_source).hexdigest() == digest
    evaluation = report["evaluation"]
    assert evaluation["horizon_seconds"] == 3.0
    assert "incidental bat-ground/wicket/robot contact" in evaluation["termination_contract"]
    if version == "balance_v3":
        assert evaluation["gravity_observation"].startswith("unit gravity in torso IMU frame")
    rows = evaluation["rows"]
    expected = set(itertools.product(("zero", "ppo"), ("right", "left"), (4101, 4102, 4103, 4104)))
    assert len(rows) == len(expected) == evaluation["expected_episodes"] == 16
    assert {(row["policy"], row["hand"], row["seed"]) for row in rows} == expected
    assert all(row["terminated"] and not row["fallen"] and not row["time_limit"] for row in rows)
    assert all(row["incidental_bat_contact_seen_at_control_snapshots"] for row in rows)
    assert all(row["incidental_bat_contact_channels"] for row in rows)
    diagnostics = json.loads((directory / "training_diagnostics.json").read_text())
    assert all(item["all_finite"] for item in diagnostics["scalar_tags"].values())
    with (directory / "training_scalars.csv").open() as stream:
        scalars = list(csv.DictReader(stream))
    assert [int(row["iteration"]) for row in scalars] == list(range(2080))
