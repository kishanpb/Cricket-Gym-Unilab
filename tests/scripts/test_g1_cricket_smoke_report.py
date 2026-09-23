"""Retained native PPO smoke evidence must include the complete declared pool."""

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_native_ppo_smoke_evidence():
    directory = ROOT / "g1_cricket_results/ppo_smoke"
    report = json.loads((directory / "evaluation.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    assert report["scope"] == "native_CPU_PPO_pipeline_smoke_not_trained_cricket"
    assert summary["status"] == "completed"
    assert summary["run_env_steps"] == report["training"]["actual_transitions"] == 960
    assert report["training"]["updates"] == 10
    assert report["training"]["last_zero_indexed_iteration"] == 9
    checkpoint = ROOT / report["checkpoint"]["path"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint"]["sha256"]
    evaluation = report["evaluation"]
    expected = set(
        itertools.product(evaluation["policies"], evaluation["hands"], evaluation["seeds"])
    )
    rows = evaluation["rows"]
    assert len(rows) == len(expected) == evaluation["expected_episodes"] == 16
    assert {(row["policy"], row["hand"], row["seed"]) for row in rows} == expected
    assert all(row["fallen"] or row["time_limit"] for row in rows)
    assert sum(row["fallen"] for row in rows) == 16
    assert not any(row["bat_contact_seen_at_control_snapshots"] for row in rows)
