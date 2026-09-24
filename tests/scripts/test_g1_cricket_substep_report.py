"""Physics-rate evidence covers the complete unchanged native development pool."""

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_complete_interval_replay_matches_parent_and_keeps_all_failures():
    report = json.loads(
        (ROOT / "g1_cricket_results/unitree_prior_v2/substep_audit.json").read_text()
    )
    parent_path = ROOT / report["parent_report"]
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == report["parent_report_sha256"]
    parent = json.loads(parent_path.read_text())
    rows = report["rows"]
    expected = set(
        itertools.product(
            ("none", "right", "left"), ("constant_target", "unitree_onnx"), range(4201, 4209)
        )
    )
    assert len(rows) == len(expected) == 48
    assert {(r["hand"], r["controller"], r["seed"]) for r in rows} == expected
    assert sum(r["physics_steps"] for r in rows) == 132790
    for row, old in zip(rows, parent["rows"], strict=True):
        assert (row["hand"], row["controller"], row["seed"]) == (
            old["hand"],
            old["controller"],
            old["seed"],
        )
        assert row["seconds"] == old["seconds"]
        assert row["final_root_pose"] == old["final_root_pose"]
        assert row["native_completed"] == old["completed"]
        assert row["physics_steps"] == 10 * row["compared_policy_ticks"]
        assert row["maximum_endpoint_state_error"] == 0
        assert row["minimum_pelvis_height_m"] <= old["minimum_pelvis_height_m"] + 1e-7
        assert (
            row["maximum_actuator_force_limit_fraction"]
            >= old["maximum_actuator_force_limit_fraction_at_control_snapshots"] - 1e-7
        )
        if row["controller"] == "unitree_onnx":
            assert row["substep_gate_passed"] and row["seconds"] == 10
            assert row["minimum_pelvis_up_z"] >= 0.65
            assert row["maximum_joint_limit_excess_rad"] == 0
            assert row["maximum_actuator_force_limit_fraction"] <= 1
            assert not any(row["contact_presence_physics_step_counts"].values())
            assert row["first_guarded_contact"] is None
        else:
            assert not row["substep_gate_passed"]
            if row["hand"] != "none":
                assert row["first_guarded_contact"]["channel"] == "bat_pitch"
                assert row["first_guarded_contact"]["seconds"] <= row["seconds"]
    for name, digest in report["source_sha256"].items():
        assert not Path(name).is_absolute()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert report["contract"]["endpoint_comparison"].startswith("exact")
    assert "no learned cricket" in report["scope"]
    assert "not continuous collision detection" in report["contract"]["contact_scope"]
