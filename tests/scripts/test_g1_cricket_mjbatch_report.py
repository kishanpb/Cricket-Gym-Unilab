"""Native executor evidence must retain every frozen-policy outcome."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_impact_events_learning import validate_pool
from evaluate_g1_cricket_residual import sha256
from evaluate_g1_cricket_swing import check_hashes


def test_complete_native_execution_report():
    directory = ROOT / "g1_cricket_results/native_mjbatch_v1"
    contract = json.loads((directory / "preflight.json").read_text())
    result = json.loads((directory / "evaluation.json").read_text())
    parent = json.loads((ROOT / "g1_cricket_results/tanh_v1/trained_evaluation.json").read_text())
    check_hashes(result["input_sha256"])
    assert result["input_sha256"] == {
        **contract["source_sha256"],
        **contract["input_sha256"],
        str((directory / "preflight.json").relative_to(ROOT)): sha256(directory / "preflight.json"),
    }
    assert result["executor"] == contract["executor"]
    assert set(result["executor"]) == {
        "version",
        "python_sha256",
        "batch_wrapper_sha256",
        "native_extension_sha256",
    }
    for key, digest in result["executor"].items():
        if key != "version":
            assert len(digest) == 64 and int(digest, 16) >= 0
    assert result["mjbatch_source_revision"] == contract["mjbatch_source_revision"]
    assert len(result["mjbatch_source_revision"]) == 40
    assert result["all_192_rows_exact"] is True
    assert result["new_training"] is result["policy_promoted"] is False
    assert result["physical_calibration_validated"] is False
    assert contract["new_training"] is contract["policy_promoted"] is False
    for index, (report, dt) in enumerate(zip(result["reports"], (0.00025, 0.000125), strict=True)):
        validate_pool(report["rows"])
        check_hashes(report["source_sha256"])
        assert report["evaluation"]["physics_dt_seconds"] == dt
        assert report["evaluation"]["left_hand"] == "untrained_transfer"
        assert report["evaluation"]["strict_outgoing_vx_threshold_m_s"] == 1
        assert report["evaluation"]["maximum_blade_penetration_m"] == 0.006
        for key in (
            "rows",
            "aggregates",
            "checkpoint",
            "versions",
            "external_asset_sha256",
            "contact_models",
        ):
            assert report[key] == parent["reports"][index][key]
        for row in report["rows"]:
            assert row["maximum_endpoint_state_error"] == row["maximum_endpoint_sensor_error"] == 0
            assert not row["passed"]
