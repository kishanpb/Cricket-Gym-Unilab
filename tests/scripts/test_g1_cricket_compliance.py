"""Explicit-pair identity, force accounting and complete finite-grid evidence."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_g1_cricket_compliance import (
    DIRECTORY,
    PAIR,
    PLAN,
    TOLERANCES,
    compare,
    model_xml,
    probe,
    sha256,
)
from probe_g1_cricket_impact import probe as historical_probe

from unilab.tasks.manipulation.g1_cricket.impact import G1CricketImpactCfg
from unilab.tasks.manipulation.g1_cricket.task import G1CricketCfg


def test_explicit_pair_matches_current_task_owner(monkeypatch, tmp_path):
    def base_scene(self, source, destination):
        destination.write_text("<mujoco/>")
        return ()

    monkeypatch.setattr(G1CricketCfg, "build_scene", base_scene)
    path = tmp_path / "scene.xml"
    G1CricketImpactCfg().build_scene(path, path)
    assert ET.parse(path).getroot().find("contact/pair").attrib == PAIR


def test_compiled_candidate_changes_only_pair_time_constant():
    a = mujoco.MjModel.from_xml_string(model_xml(0.000125, 0.004))
    b = mujoco.MjModel.from_xml_string(model_xml(0.000125, 0.002))
    assert a.npair == b.npair == 1
    np.testing.assert_array_equal(a.pair_solref, [[0.004, 1]])
    np.testing.assert_array_equal(b.pair_solref, [[0.002, 1]])
    for field in (
        "pair_solimp",
        "pair_friction",
        "pair_dim",
        "pair_gap",
        "pair_margin",
        "pair_geom1",
        "pair_geom2",
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_pos",
        "qpos0",
    ):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    np.testing.assert_array_equal(a.body_mass, [0, 0.156])
    assert a.opt.disableflags == b.opt.disableflags == 0
    assert a.opt.integrator == b.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert a.opt.solver == b.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON


@pytest.mark.parametrize("side", [-1, 1])
def test_force_sign_momentum_and_energy(side):
    row = probe(0.00025, 0.004, 2.5, side)
    assert not row["validation_failures"]
    assert side * row["world_impulse_on_ball_ns"][0] > 0
    np.testing.assert_allclose(
        row["world_impulse_on_ball_ns"], row["momentum_change_kg_m_s"], atol=1e-10, rtol=0
    )
    assert row["work_energy_error_j"] <= 1e-10
    assert row["contact_translational_work_j"] < 0
    assert 0 < row["loaded_seconds"] <= row["contact_seconds"]
    assert 0 < row["rebound_ratio"] < 1


def test_comparator_rejects_missing_exits_and_each_metric():
    a = dict.fromkeys(TOLERANCES, 1.0)
    assert not compare(a, a)
    for key, absolute in TOLERANCES.items():
        assert key in compare({**a, key: 1 + 2 * max(absolute, 0.05)}, a)
    assert "first_exit_vx_m_s" in compare({**a, "first_exit_vx_m_s": None}, a)


def test_historical_default_pair_rows_stay_reproducible():
    path = ROOT / "g1_cricket_results/residual_v3/isolated_impact.json"
    report = json.loads(path.read_text())
    assert report["source_sha256"] == sha256(ROOT / "scripts/probe_g1_cricket_impact.py")
    for row in report["rows"]:
        assert historical_probe(row["sim_dt"], row["solref_time_constant_s"]) == row


def test_retained_matrix_reproduces_every_row():
    path = DIRECTORY / "evaluation.json"
    if not path.exists():
        pytest.skip("preflight source tests; retained matrix not yet run")
    report = json.loads(path.read_text())
    assert report["plan"] == PLAN
    assert not report["robot_task_changed"]
    assert not report["physical_calibration_validated"]
    assert not report["policy_promoted"]
    for name, digest in report["input_sha256"].items():
        assert sha256(ROOT / name) == digest
    rows = report["rows"]
    assert len(rows) == PLAN["expected_rows"]
    assert {(r["time_constant_s"], r["incident_speed_m_s"], r["sim_dt"]) for r in rows} == {
        (tc, speed, dt)
        for tc in PLAN["time_constants_s"]
        for speed in PLAN["incident_speeds_m_s"]
        for dt in PLAN["timesteps_s"]
    }
    for row in rows:
        assert row == probe(row["sim_dt"], row["time_constant_s"], row["incident_speed_m_s"])
        assert not row["validation_failures"]
    assert len(report["comparisons"]) == 12
    for comparison in report["comparisons"]:
        group = [
            r
            for r in rows
            if r["time_constant_s"] == comparison["time_constant_s"]
            and r["incident_speed_m_s"] == comparison["incident_speed_m_s"]
        ]
        coarse = next(r for r in group if r["sim_dt"] == comparison["coarse_dt"])
        fine = next(r for r in group if r["sim_dt"] == comparison["fine_dt"])
        assert comparison["failed_checks"] == compare(coarse, fine)
