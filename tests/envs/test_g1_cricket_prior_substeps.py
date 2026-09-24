"""Serial interval replay must match public native endpoints before auditing loads."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("onnxruntime")
import audit_g1_cricket_prior_substeps as audit_module
from audit_g1_cricket_prior_substeps import IntervalReplay
from evaluate_g1_cricket_prior import make_env


@pytest.mark.parametrize("hand", ["none", "right", "left"])
def test_serial_interval_matches_native_and_records_all_substeps(hand):
    env = make_env(hand, "v2")
    try:
        env.reset(seed=4201)
        replay = IntervalReplay(env)
        assert replay.model.opt.timestep == 0.002
        for index in range(8):
            action = np.full((1, 29), index * 0.001, dtype=np.float32)
            state, trajectory, forces, presence, peaks = replay.step(env, action)
            assert not state.terminated.any()
            assert trajectory.shape == (10, 85) and forces.shape == (10, 29)
            assert presence.shape == peaks.shape == (10, len(replay.names))
        assert replay.state_error == 0
    finally:
        env.close()


def test_replay_rejects_mismatched_physics():
    env = make_env("right", "v2")
    try:
        env.reset(seed=4201)
        replay = IntervalReplay(env)
        replay.model.opt.timestep *= 2
        with pytest.raises(AssertionError):
            replay.step(env, np.zeros((1, 29), dtype=np.float32))
    finally:
        env.close()


def test_replay_rejects_mismatched_native_sensor():
    env = make_env("right", "v2")
    try:
        env.reset(seed=4201)
        replay = IntervalReplay(env)
        native = replay.native_sensors
        replay.native_sensors = SimpleNamespace(read=lambda: native.read() + 0.001)
        with pytest.raises(AssertionError):
            replay.step(env, np.zeros((1, 29), dtype=np.float32))
    finally:
        env.close()


def test_parent_source_change_blocks_replay(tmp_path, monkeypatch):
    parent = tmp_path / "g1_cricket_results/unitree_prior_v2/evaluation.json"
    parent.parent.mkdir(parents=True)
    (tmp_path / "source.py").write_text("changed")
    parent.write_text(
        json.dumps({"source_sha256": {"source.py": hashlib.sha256(b"original").hexdigest()}})
    )
    monkeypatch.setattr(audit_module, "ROOT", tmp_path)
    monkeypatch.setattr(audit_module, "load_prior", lambda _: None)
    with pytest.raises(ValueError, match="parent source changed"):
        audit_module.audit(tmp_path)
