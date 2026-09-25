"""A training budget or selected successful row cannot qualify the full pilot."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import mjbatch.held_control
import pytest
import report_g1_cricket_batting_learning as reporter


@pytest.fixture
def training():
    summary = {
        "status": "completed",
        "run_env_steps": 49152,
        "total_env_steps": 49152,
        "completed_iterations": 255,
        "global_num_envs": 8,
        "effective_seed": 1,
        "task": "G1CricketBimanualLearning",
        "algo": "ppo",
        "sim_backend": "mujoco",
    }
    config = {
        "algo": {"resume": False, "algorithm": {"disable_finite_checks": False}},
        "env": {
            "sim_dt": 0.00003125,
            "actions": {
                "reference": {
                    "scale": 0.05,
                    "lookahead_frames": 0,
                    "balance_gain": 4,
                    "waist_tracking_gain": 1,
                    "root_position_gain": 4,
                }
            },
            "observations": {
                group: {"terms": {name: {} for name in ("ball", "contact", "bat_error")}}
                for group in ("actor", "critic")
            },
        },
        "reward": {"bat_tracking": {"weight": 2, "params": {"std": 0.08}}},
    }
    diagnostics = {"scalar_tags": {"reward": {"all_finite": True}}}
    return summary, config, diagnostics


@pytest.mark.parametrize(
    "case", ["partial", "resume", "finite", "lead", "dt", "observation", "reward", "scalar"]
)
def test_learning_report_rejects_changed_protocol(training, case):
    summary, config, diagnostics = training
    reporter.validate_training(summary, config, diagnostics)
    if case == "partial":
        summary["run_env_steps"] -= 192
    elif case == "resume":
        config["algo"]["resume"] = True
    elif case == "finite":
        config["algo"]["algorithm"]["disable_finite_checks"] = True
    elif case == "lead":
        config["env"]["actions"]["reference"]["lookahead_frames"] = 1
    elif case == "dt":
        config["env"]["sim_dt"] = 0.001
    elif case == "observation":
        del config["env"]["observations"]["actor"]["terms"]["ball"]
    elif case == "reward":
        config["reward"]["bat_tracking"]["params"]["std"] = 0.2
    elif case == "scalar":
        diagnostics["scalar_tags"]["reward"]["all_finite"] = False
    with pytest.raises(ValueError):
        reporter.validate_training(summary, config, diagnostics)


def test_learning_report_requires_both_hands_controls_and_resolutions(
    training, tmp_path, monkeypatch
):
    monkeypatch.setattr(reporter, "summarize_bounced_row", lambda row: row)
    runtime_hash = hashlib.sha256(Path(mjbatch.held_control.__file__).read_bytes()).hexdigest()
    for hand in ("right", "left"):
        summary, config, diagnostics = deepcopy(training)
        run = tmp_path / f"ppo_{hand}"
        run.mkdir()
        checkpoint = run / "model_255.pt"
        checkpoint.write_bytes(hand.encode())
        summary["last_checkpoint"] = str(checkpoint)
        config["env"]["handedness"] = hand
        for name, value in (
            ("run_summary", summary),
            ("run_config", {"config": config}),
            ("training_diagnostics", diagnostics),
        ):
            (run / f"{name}.json").write_text(json.dumps(value))
        (run / "training_scalars.csv").write_text(
            "iteration,reward\n" + "".join(f"{i},1\n" for i in range(256))
        )
        for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
            case = tmp_path / f"{hand}_{resolution}"
            case.mkdir()
            report = {
                "scope": reporter.SCOPE,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "evaluation_overrides": {
                    "contact_dt": dt,
                    "soft_toss": False,
                    "bounced_delivery": True,
                    "compact_substeps": True,
                },
                "input_sha256": {},
                "runtime_source_sha256": {"mjbatch.held_control": runtime_hash},
                "rows": [
                    {
                        "hand": hand,
                        "controller": control,
                        "all_checks_pass": True,
                        "ball_contacts": {
                            "ball_geom/bat_blade": {"peak_force_n": 100, "penetration_m": 0.001}
                        },
                        "first_force_free_blade_exit": {"velocity_m_s": [2, 0, 1]},
                    }
                    for control in ("reference_only", "ppo")
                ],
            }
            (case / "evaluation.json").write_text(json.dumps(report))
    result = reporter.build_report(tmp_path)
    assert len(result["rows"]) == 8 and len(result["resolution_comparisons"]) == 4
    assert result["trained_policy_qualification"] == {"right": True, "left": True}
    path = tmp_path / "left_finest/evaluation.json"
    report = json.loads(path.read_text())
    report["rows"][1]["all_checks_pass"] = False
    path.write_text(json.dumps(report))
    result = reporter.build_report(tmp_path)
    assert result["trained_policy_qualification"] == {"right": True, "left": False}
    assert not result["all_trained_policies_qualify"]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reporter.build_report(tmp_path)
