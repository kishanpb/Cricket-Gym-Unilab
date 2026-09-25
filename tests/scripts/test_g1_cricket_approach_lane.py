import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from unilab.tasks.manipulation.g1_cricket.approach import approach_command
from unilab.tasks.manipulation.g1_cricket.approach_lane import lane_command


def test_lane_sign_bound_forward_preservation_and_partial_reset():
    positions = np.array([[0, 0.8, 0.8], [0, -0.8, 0.8], [0, 2, 0.8]])
    defaults = np.array([[0, 0.7, 0.8], [0, -0.7, 0.8], [0, 0.7, 0.8]])
    env = SimpleNamespace(
        num_envs=3,
        episode_length_buf=np.array([75, 150, 250]),
        step_dt=0.02,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=positions, default_root_state=defaults)
            )
        },
    )
    baseline = approach_command(env)
    result = lane_command(env)
    np.testing.assert_array_equal(result[:, [0, 2]], baseline[:, [0, 2]])
    np.testing.assert_allclose(result[:, 1], [-0.1, 0.1, -0.25])
    positions[0] = defaults[0]
    env.episode_length_buf[0] = 0
    after = lane_command(env)
    np.testing.assert_array_equal(after[0], 0)
    np.testing.assert_array_equal(after[1:], result[1:])


def test_retained_complete_comparison_and_provenance():
    root = Path(__file__).resolve().parents[2]
    directory = root / "g1_cricket_results/approach_lane_v1"
    result = json.loads((directory / "summary.json").read_text())
    assert not result["qualified"]
    assert {(row["hand"], row["resolution"]) for row in result["rows"]} == {
        (hand, resolution) for hand in ("right", "left") for resolution in ("fine", "finest")
    }
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        baseline = next(
            old
            for old in result["baseline_rows"]
            if (old["hand"], old["resolution"]) == (row["hand"], row["resolution"])
        )
        assert row["compiled_model_sha256"] == baseline["compiled_model_sha256"]
        assert row["exact_endpoint_and_sensor_replay"] and row["checks"]["complete"]
        assert row["checks"]["no_release"] and not row["passed"]
        with np.load(directory / row["trace"]) as trace:
            assert len(trace["states"]) == 401 and len(trace["actions"]) == 400
            drift = np.abs(trace["steps"][:, 2] - trace["states"][0, 2]).max()
            assert drift == row["maximum_lateral_drift_m"]
    assert sum(row["substeps"] for row in result["rows"]) == 768000
    for name, digest in result["input_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    for name, digest in result["artifact_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
