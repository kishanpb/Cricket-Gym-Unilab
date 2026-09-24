"""No reward repair may hide a physical-policy change or relabel native returns."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_events import compare_row
from evaluate_g1_cricket_residual import sha256
from g1_cricket_archival_sources import check_retained_hashes


def test_comparison_rejects_physics_or_unexplained_reward_changes():
    audit = json.loads(
        (ROOT / "g1_cricket_results/impact_v1/impact_reward_audit.json").read_text()
    )["rows"][0]
    old = json.loads((ROOT / "g1_cricket_results/impact_v1/trained_evaluation.json").read_text())[
        "reports"
    ][1]["rows"][0]
    new = copy.deepcopy(old)
    assert compare_row(old, new, audit)["non_return_fields_exact"]
    new["minimum_pelvis_height_m"] -= 1e-10
    with pytest.raises(AssertionError, match="physical row changed"):
        compare_row(old, new, audit)
    new = copy.deepcopy(old)
    new["return"] += 0.001
    with pytest.raises(AssertionError):
        compare_row(old, new, audit)


def test_complete_preflight():
    result = json.loads((ROOT / "g1_cricket_results/impact_events_v1/preflight.json").read_text())
    parent_path = ROOT / "g1_cricket_results/impact_v1/trained_evaluation.json"
    audit_path = ROOT / "g1_cricket_results/impact_v1/impact_reward_audit.json"
    assert result["parent_report_sha256"] == sha256(parent_path)
    assert result["reward_audit_sha256"] == sha256(audit_path)
    report = result["report"]
    check_retained_hashes(ROOT, report["source_sha256"])
    parent = json.loads(parent_path.read_text())["reports"][1]
    audit = json.loads(audit_path.read_text())
    assert len(report["rows"]) == len(result["comparisons"]) == 96
    for old, new, evidence, comparison in zip(
        parent["rows"], report["rows"], audit["rows"], result["comparisons"], strict=True
    ):
        assert compare_row(old, new, evidence) == comparison
    assert sum(c["new_first_event"] for c in result["comparisons"]) == 9
    assert sum(c["new_separation_bonus"] > 0 for c in result["comparisons"]) == 40
    assert all(not r["passed"] for r in report["rows"])
    assert all(np.isfinite(c["new_return"]) for c in result["comparisons"])
    assert result["training_preflight_ready"] is True
    assert result["physical_calibration_validated"] is result["policy_promoted"] is False
