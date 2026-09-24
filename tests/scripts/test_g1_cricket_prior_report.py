"""Retain full native prior pools, failures and provenance without promotion claims."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scripts.g1_cricket_historical_sources import legacy_source_digest

from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES, REVISION

ROOT = Path(__file__).resolve().parents[2]


def read_report(version):
    directory = (
        ROOT / "g1_cricket_results" / ("unitree_prior" if version == "v1" else "unitree_prior_v2")
    )
    return directory, json.loads((directory / "evaluation.json").read_text())


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_prior_complete_pool_and_provenance(version):
    directory, report = read_report(version)
    assert report["version"] == version
    assert report["external_asset_sha256"] == ASSET_HASHES
    assert report["source"].endswith(REVISION)
    assert "no local cricket learning" in report["scope"]
    contract = report["contract"]
    assert not contract["root_support"] and not contract["pose_write_after_reset"]
    assert contract["control_dt_s"] == 0.02 and contract["physics_dt_s"] == 0.002
    assert "may miss between-control impacts" in contract["contact_scope"]
    assert "not hardware tactile pressure" in contract["force_scope"]
    expected = set(
        itertools.product(
            ("none", "right", "left"), ("constant_target", "unitree_onnx"), range(4201, 4209)
        )
    )
    rows = report["rows"]
    assert len(rows) == len(expected) == 48
    assert {(r["hand"], r["controller"], r["seed"]) for r in rows} == expected
    for name, digest in report["source_sha256"].items():
        assert not Path(name).is_absolute()
        assert legacy_source_digest(ROOT, name) == digest
    for name, key in (
        ("g1.xml", "robot_xml_sha256"),
        ("scene_flat.xml", "robot_scene_flat_sha256"),
    ):
        assert (
            hashlib.sha256((ROOT / "src/unilab/assets/robots/g1" / name).read_bytes()).hexdigest()
            == report[key]
        )
    image = np.asarray(Image.open(directory / "stance_diagnostic.png"))
    assert image.shape[:2] == (1080, 1920) and image[..., :3].std() > 10


def test_mount_comparison_keeps_failed_parent_and_identical_no_bat_rows():
    _, parent = read_report("v1")
    _, candidate = read_report("v2")
    assert [r for r in parent["rows"] if r["hand"] == "none"] == [
        r for r in candidate["rows"] if r["hand"] == "none"
    ]
    failures = [
        r for r in parent["rows"] if r["controller"] == "unitree_onnx" and not r["completed"]
    ]
    assert [(r["hand"], r["seed"]) for r in failures] == [
        ("right", seed) for seed in (4205, 4206, 4207)
    ]
    assert all(r["terminated"] and not r["fell"] for r in failures)
    assert all(r["contact_presence_control_counts"]["bat_wicket_1"] > 0 for r in failures)
    prior = [r for r in candidate["rows"] if r["controller"] == "unitree_onnx"]
    assert len(prior) == 24
    for row in prior:
        assert row["completed"] and row["time_limit"] and not row["terminated"]
        assert row["seconds"] == 10 and not row["fell"]
        assert row["maximum_joint_limit_excess_rad"] == 0
        assert row["maximum_actuator_force_limit_fraction_at_control_snapshots"] <= 1
        assert not any(row["contact_presence_control_counts"].values())
    controls = [r for r in candidate["rows"] if r["controller"] == "constant_target"]
    assert all(not r["completed"] for r in controls)
    assert all(r["fell"] for r in controls if r["hand"] == "none")
    assert all(
        r["contact_presence_control_counts"]["bat_pitch"] > 0
        for r in controls
        if r["hand"] != "none"
    )
