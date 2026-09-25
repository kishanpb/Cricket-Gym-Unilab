"""A contact or a completed swing does not clear the original physical gates."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import mjbatch.held_control
import numpy as np
import pytest
from evaluate_g1_cricket_tracking import BallContactSequence
from report_g1_cricket_bimanual_contact import compare_resolution, summarize_row
from report_g1_cricket_bounced_delivery import build_report, summarize_bounced_row
from report_g1_cricket_motor_lead import build_report as build_motor_lead_report


@pytest.fixture
def row():
    frame = {
        "time_s": 3.0,
        "root_translation_error_m": 0.01,
        "reference_joint_rmse_rad": 0.01,
        "bat_tracking_error_m": 0.01,
        "substep_audit": {
            "peaks": {
                "hard_joint_limit_excess_rad": 0.0,
                "grip_separation_m": 0.001,
                "motor_force_fraction": 0.5,
            },
            "minimum_pelvis_height_m": 0.75,
            "unexpected_contacts": [],
            "ball_contacts": {
                "ball_geom/bat_blade": {
                    "duration_s": 0.01,
                    "peak_force_n": 100,
                    "normal_impulse_ns": 0.5,
                    "penetration_m": 0.001,
                }
            },
            "first_force_free_blade_exit": {"time_s": 1.4, "velocity_m_s": [2, 0, 1]},
        },
    }
    return {
        "hand": "right",
        "controller": "ppo",
        "return": 1,
        "steps": 150,
        "terminated": False,
        "terminal_terms": ["time_out"],
        "trace": [frame],
    }


def test_bat_accuracy_gate_not_removed_by_contact(row):
    assert summarize_row(row)["all_checks_pass"]
    row["trace"][0]["bat_tracking_error_m"] = 0.081
    result = summarize_row(row)
    assert all(result["contact_checks"].values())
    assert not result["all_checks_pass"]


def test_motor_lead_report_keeps_every_case_and_rejects_one_failed_candidate(row, tmp_path):
    runtime_hash = hashlib.sha256(Path(mjbatch.held_control.__file__).read_bytes()).hexdigest()
    sequence = BallContactSequence()
    position = np.array([0.8, 0, 0.036])
    down, up = np.array([-3, 0, -6]), np.array([-2.3, 0, 3])
    sequence.update(1.0, position, down, down, True, False)
    sequence.update(1.02, position, up, up, False, False)
    sequence.update(1.4, position, down, up, False, True)
    row["trace"][0]["substep_audit"]["ball_contact_sequence"] = sequence.snapshot()
    for lead in (0, 1, 3):
        for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
            for hand in ("right", "left"):
                directory = tmp_path / f"lead{lead}_{hand}_{resolution}"
                directory.mkdir()
                rows = []
                for controller in ("reference_only", "ppo"):
                    candidate = deepcopy(row)
                    candidate.update(hand=hand, controller=controller)
                    candidate["trace"][0]["bat_tracking_error_m"] = 0.09 if lead == 0 else 0.07
                    rows.append(candidate)
                report = {
                    "evaluation_overrides": {
                        "waist_tracking_gain": 1,
                        "root_position_gain": 4,
                        "lookahead_frames": lead,
                        "contact_dt": dt,
                        "soft_toss": False,
                        "bounced_delivery": True,
                        "compact_substeps": True,
                    },
                    "input_sha256": {},
                    "runtime_source_sha256": {"mjbatch.held_control": runtime_hash},
                    "rows": rows,
                }
                path = directory / "evaluation.json"
                path.write_text(json.dumps(report))
    result = build_motor_lead_report(tmp_path)
    assert len(result["rows"]) == 24
    assert len(result["timing_comparisons"]) == 16
    assert len(result["resolution_comparisons"]) == 12
    assert result["candidate_qualification"] == {"1": True, "3": True}
    assert all(
        pair["peak_bat_error_change_m"] == pytest.approx(-0.02)
        for pair in result["timing_comparisons"]
    )
    report["rows"][1]["trace"][0]["bat_tracking_error_m"] = 0.081
    path.write_text(json.dumps(report))
    assert build_motor_lead_report(tmp_path)["candidate_qualification"] == {"1": True, "3": False}
    report["evaluation_overrides"]["lookahead_frames"] = 0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="protocol"):
        build_motor_lead_report(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        build_motor_lead_report(tmp_path)


@pytest.mark.parametrize("frames", [(), (0,), (-1,), (1, 1), (1.5,), (True,)])
def test_motor_lead_report_rejects_invalid_candidate_pool(tmp_path, frames):
    with pytest.raises(ValueError, match="distinct positive integers"):
        build_motor_lead_report(tmp_path, candidate_frames=frames)


def test_completion_uses_all_150_controls_not_float32_clock(row):
    row["trace"][0]["time_s"] = 2.999997854
    assert summarize_row(row)["original_checks"]["complete"]
    row["steps"] = 149
    assert not summarize_row(row)["original_checks"]["complete"]


def test_miss_cannot_pass_contact_or_resolution(row):
    audit = row["trace"][0]["substep_audit"]
    audit["ball_contacts"] = {}
    audit["first_force_free_blade_exit"] = None
    result = summarize_row(row)
    assert not result["all_checks_pass"]
    assert not compare_resolution(result, result)["all_pass"]


def test_late_contact_failure_and_impulse_are_retained(row):
    later = deepcopy(row["trace"][0])
    later["substep_audit"]["ball_contacts"]["ball_geom/bat_blade"]["penetration_m"] = 0.007
    later["substep_audit"]["unexpected_contacts"] = ["ball_geom/left_hand_collision"]
    row["trace"].append(later)
    result = summarize_row(row)
    assert not result["contact_checks"]["all_ball_penetration"]
    assert not result["original_checks"]["no_unexpected_contact"]
    assert result["ball_contacts"]["ball_geom/bat_blade"]["normal_impulse_ns"] == 1


def test_resolution_uses_complete_pair_not_only_exit_velocity(row):
    fine = summarize_row(row)
    coarse = deepcopy(fine)
    assert compare_resolution(coarse, fine)["all_pass"]
    coarse["ball_contacts"]["ball_geom/bat_blade"]["peak_force_n"] = 106
    assert not compare_resolution(coarse, fine)["all_pass"]
    coarse = deepcopy(fine)
    coarse["ball_contacts"]["ball_geom/pitch"] = {"peak_force_n": 1, "penetration_m": 0.001}
    assert not compare_resolution(coarse, fine)["all_pass"]


@pytest.mark.parametrize("case", ["one", "none", "two", "incomplete", "after", "upward"])
def test_one_bounce_requires_completed_incoming_pitch_before_blade(row, case):
    sequence = BallContactSequence()
    position = np.array([0.8, 0, 0.036])
    down, up = np.array([-3, 0, -6]), np.array([-2.3, 0, 3])
    if case == "after":
        sequence.update(0.9, position, down, down, False, True)
    if case != "none":
        sequence.update(1.0, position, up if case == "upward" else down, down, True, False)
        early = sequence.snapshot()
        if case != "incomplete":
            sequence.update(1.02, position, up, up, False, False)
            assert "end_s" not in early["pitch_events"][0]
    if case == "two":
        sequence.update(1.2, position, down, down, True, False)
        sequence.update(1.22, position, up, up, False, False)
    sequence.update(1.4, position, down, up, case == "incomplete", True)
    row["trace"][-1]["substep_audit"]["ball_contact_sequence"] = sequence.snapshot()
    result = summarize_bounced_row(row)
    assert result["all_checks_pass"] == (case == "one")
    assert result["delivery_checks"]["exactly_one_completed_bounce_before_blade"] == (case == "one")


def test_third_resolution_retains_prior_rows_and_both_adjacent_pairs(row, tmp_path):
    runtime_hash = hashlib.sha256(Path(mjbatch.held_control.__file__).read_bytes()).hexdigest()
    row["trace"][-1]["substep_audit"]["ball_contact_sequence"] = {
        "pitch_events": [
            {
                "start_s": 1.0,
                "end_s": 1.01,
                "position_m": [0.8, 0, 0.036],
                "incoming_velocity_m_s": [-3, 0, -6],
                "outgoing_velocity_m_s": [-2.3, 0, 3],
            }
        ],
        "first_blade_contact": {"time_s": 1.4, "incoming_velocity_m_s": [-2.3, 0, -0.5]},
    }
    for resolution, dt in (("coarse", 0.0000625), ("fine", 0.00003125), ("finest", 0.000015625)):
        if resolution == "finest":
            parent = build_report(tmp_path)
            assert len(parent["rows"]) == 8 and len(parent["resolution_comparisons"]) == 4
            with pytest.raises(FileNotFoundError):
                build_report(tmp_path, include_finest=True)
        for hand in ("right", "left"):
            directory = tmp_path / f"{hand}_{resolution}"
            directory.mkdir()
            rows = []
            for controller in ("reference_only", "ppo"):
                candidate = deepcopy(row)
                candidate.update(hand=hand, controller=controller)
                if resolution == "finest":
                    candidate["trace"][0]["substep_audit"]["ball_contacts"]["ball_geom/bat_blade"][
                        "peak_force_n"
                    ] = 120
                rows.append(candidate)
            report = {
                "evaluation_overrides": {
                    "waist_tracking_gain": 1,
                    "root_position_gain": 4,
                    "contact_dt": dt,
                    "soft_toss": False,
                    "bounced_delivery": True,
                    "compact_substeps": True,
                },
                "input_sha256": {},
                "runtime_source_sha256": {"mjbatch.held_control": runtime_hash},
                "rows": rows,
            }
            (directory / "evaluation.json").write_text(json.dumps(report))
    result = build_report(tmp_path, include_finest=True)
    assert result["rows"][:8] == parent["rows"]
    assert len(result["rows"]) == 12 and len(result["resolution_comparisons"]) == 8
    for comparison in result["resolution_comparisons"]:
        assert comparison["all_pass"] == (comparison["resolutions"] == ["coarse", "fine"])
    assert not result["all_checks_pass"]
