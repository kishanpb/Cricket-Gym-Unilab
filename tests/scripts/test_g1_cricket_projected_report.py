import hashlib
import json
from pathlib import Path

import mjbatch.held_control
import pytest
import report_g1_cricket_projected_batting as reporter


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "ROOT", tmp_path)
    monkeypatch.setattr(reporter, "summarize_bounced_row", lambda row: row)
    directory = tmp_path / "projected"
    runtime_hash = hashlib.sha256(Path(mjbatch.held_control.__file__).read_bytes()).hexdigest()
    for hand in ("right", "left"):
        run = tmp_path / f"g1_cricket_results/bimanual_batting_learning_v1/ppo_{hand}"
        run.mkdir(parents=True)
        (run / "model_255.pt").write_bytes(hand.encode())
        for variant in ("baseline", "projected"):
            for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
                case = directory / f"{variant}_{hand}_{resolution}"
                case.mkdir(parents=True)
                overrides = dict(
                    contact_dt=dt, soft_toss=False, bounced_delivery=True, compact_substeps=True
                )
                scope = "ball_contact_observed_ppo_nominal_bounced_delivery_not_held_out"
                if variant == "projected":
                    overrides["reference_directory"] = "projected"
                    scope = "frozen_ball_observed_ppo_projected_reference_not_retrained"
                report = {
                    "scope": scope,
                    "evaluation_overrides": overrides,
                    "checkpoint_sha256": hashlib.sha256(hand.encode()).hexdigest(),
                    "input_sha256": {},
                    "runtime_source_sha256": {"mjbatch.held_control": runtime_hash},
                    "rows": [
                        dict(
                            hand=hand,
                            controller=control,
                            all_checks_pass=True,
                            peak_bat_error_m=0.05,
                            ball_contacts={
                                "ball_geom/bat_blade": {"peak_force_n": 100, "penetration_m": 0.001}
                            },
                            first_force_free_blade_exit={"velocity_m_s": [2, 0, 1]},
                        )
                        for control in ("reference_only", "ppo")
                    ],
                }
                (case / "evaluation.json").write_text(json.dumps(report))
    return directory


def test_projection_report_preserves_failed_outcome_and_full_pool(pool):
    result = reporter.build_report(pool)
    assert len(result["rows"]) == 16
    assert len(result["resolution_comparisons"]) == len(result["reference_comparisons"]) == 8
    assert result["projected_qualification"]
    path = pool / "projected_left_finest/evaluation.json"
    report = json.loads(path.read_text())
    report["rows"][1]["all_checks_pass"] = False
    path.write_text(json.dumps(report))
    assert not reporter.build_report(pool)["projected_qualification"]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reporter.build_report(pool)


@pytest.mark.parametrize(
    "failure", ["checkpoint", "scope", "override", "controller", "hand", "source", "runtime"]
)
def test_projection_report_rejects_changed_evidence(pool, failure):
    path = pool / "projected_left_finest/evaluation.json"
    report = json.loads(path.read_text())
    if failure == "checkpoint":
        report["checkpoint_sha256"] = "changed"
    elif failure == "scope":
        report["scope"] = "different"
    elif failure == "override":
        report["evaluation_overrides"]["lookahead_frames"] = 1
    elif failure == "controller":
        report["rows"].pop()
    elif failure == "hand":
        report["rows"][1]["hand"] = "right"
    elif failure == "source":
        report["input_sha256"] = {str(path.relative_to(pool.parent)): "changed"}
    elif failure == "runtime":
        report["runtime_source_sha256"] = {}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        reporter.build_report(pool)


def test_compensation_report_uses_fixed_projected_reference_and_full_pool(pool):
    for path in list(pool.glob("*/evaluation.json")):
        report = json.loads(path.read_text())
        report["evaluation_overrides"]["reference_directory"] = (
            "g1_cricket_results/bimanual_projected_v1"
        )
        report["scope"] = "frozen_ball_observed_ppo_projected_reference_not_retrained"
        for row in report["rows"]:
            row["trace"] = [{"substep_audit": {"peaks": {"motor_force_fraction": 1.0}}}] * 150
        if path.parent.name.startswith("projected_"):
            report["scope"] = "frozen_ball_observed_ppo_inertial_compensation_not_retrained"
            report["evaluation_overrides"]["inertial_compensation"] = True
            report["inertial_feedforward_audit"] = [{}] * 151
            old = path.parent
            target = old.with_name(old.name.replace("projected_", "compensated_"))
            old.rename(target)
            path = target / path.name
        path.write_text(json.dumps(report))
    result = reporter.build_report(pool, compensation=True)
    assert len(result["rows"]) == 16 and result["compensated_qualification"]
    assert all(row["control_intervals_reaching_motor_limit"] == 150 for row in result["rows"])
    path = pool / "compensated_left_finest/evaluation.json"
    report = json.loads(path.read_text())
    report["rows"][1]["all_checks_pass"] = False
    path.write_text(json.dumps(report))
    assert not reporter.build_report(pool, compensation=True)["compensated_qualification"]
    report["inertial_feedforward_audit"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="all reference feedforward"):
        reporter.build_report(pool, compensation=True)
