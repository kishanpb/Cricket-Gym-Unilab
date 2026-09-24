"""Predeclare and evaluate the reward-only forward-swing learning experiment."""

import argparse
import copy
import json

from evaluate_g1_cricket_impact_events_learning import apply_penetration_gate, validate_baseline
from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_residual import ROOT, evaluate, sha256
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

DIRECTORY = ROOT / "g1_cricket_results/swing_v1"
RUN_DIR = DIRECTORY / "right"
PARENT = ROOT / "g1_cricket_results/impact_events_v1"


def validate_config(saved, reference):
    expected = copy.deepcopy(reference)
    expected["reward"]["batting"]["func"] = (
        "unilab.tasks.manipulation.g1_cricket.swing_reward.ForwardSwingReward"
    )
    expected["training"]["log_dir"] = "g1_cricket_results/swing_v1/right"
    if saved != expected:
        raise ValueError("swing config differs beyond approach reward and log directory")
    assert saved["algo"]["resume"] is False and saved["algo"]["resume_path"] is None


def check_hashes(pinned):
    for name, expected in pinned.items():
        assert sha256(ROOT / name) == expected, name


def preflight():
    if (DIRECTORY / "preflight.json").exists() or RUN_DIR.exists():
        raise FileExistsError("swing experiment already has retained inputs or a run")
    parent = json.loads((PARENT / "trained_evaluation.json").read_text())
    headroom = json.loads((PARENT / "impact_headroom.json").read_text())
    check_hashes(headroom["input_sha256"])
    assert headroom["parent_report_sha256"] == sha256(PARENT / "trained_evaluation.json")
    sources = dict(parent["reports"][0]["source_sha256"])
    for name in (
        "src/unilab/tasks/manipulation/g1_cricket/swing_reward.py",
        "src/unilab/conf/ppo/task/g1_cricket_swing_v1/mujoco.yaml",
        "docs/g1_cricket_swing_v1.md",
        "scripts/evaluate_g1_cricket_swing.py",
    ):
        sources[name] = sha256(ROOT / name)
    check_hashes(sources)
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        config = OmegaConf.to_container(
            compose(
                "config",
                overrides=[
                    "task=g1_cricket_swing_v1/mujoco",
                    "training.log_dir=g1_cricket_results/swing_v1/right",
                ],
            ),
            resolve=True,
        )
    reference = json.loads((PARENT / "right/run_config.json").read_text())["config"]
    validate_config(config, reference)
    pinned = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            PARENT / "trained_evaluation.json",
            PARENT / "impact_headroom.json",
            PARENT / "right/run_config.json",
        )
    }
    result = {
        "scope": "reward_only_forward_swing_pretraining_contract",
        "config": config,
        "source_sha256": sources,
        "input_sha256": pinned,
        "policy_promoted": False,
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    (DIRECTORY / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Reward-only composed config and parent/source hashes verified before training.")


def run():
    preflight_path = DIRECTORY / "preflight.json"
    inputs = json.loads(preflight_path.read_text())
    pinned = {**inputs["input_sha256"], **inputs["source_sha256"]}
    check_hashes(pinned)
    parent = json.loads((PARENT / "trained_evaluation.json").read_text())
    saved = json.loads((RUN_DIR / "run_config.json").read_text())["config"]
    assert saved == inputs["config"], "trained config differs from predeclared config"
    validate_config(saved, json.loads((PARENT / "right/run_config.json").read_text())["config"])
    summary = json.loads((RUN_DIR / "run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == 199680
    assert summary["configured_seed"] == summary["effective_seed"] == 1
    assert summary["completed_iterations"] == 2079
    assert summary["global_num_envs"] == 4 and summary["samples_per_iteration"] == 96
    assert summary["last_checkpoint"] == "g1_cricket_results/swing_v1/right/model_2079.pt"
    pinned.update(
        {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                preflight_path,
                RUN_DIR / "run_config.json",
                RUN_DIR / "run_summary.json",
                ROOT / summary["last_checkpoint"],
            )
        }
    )
    reports = []
    for index, dt in enumerate((0.00025, 0.000125)):
        check_hashes(pinned)
        report = evaluate(RUN_DIR, {"env": {"sim_dt": dt}})
        assert report["checkpoint"]["sha256"] == pinned[summary["last_checkpoint"]]
        for name in ("run_config", "run_summary"):
            assert (
                report[f"{name}_sha256"] == pinned[f"g1_cricket_results/swing_v1/right/{name}.json"]
            )
        for key in ("versions", "contact_models", "external_asset_sha256"):
            assert report[key] == parent["reports"][index][key]
        apply_penetration_gate(report)
        validate_baseline(report, parent["reports"][index], same_reward=False)
        report["source_sha256"].update(inputs["source_sha256"])
        report["scope"] = "fresh_right_ppo_forward_swing_reward_development_not_promoted"
        reports.append(report)
        print(json.dumps({"dt": dt, "aggregates": report["aggregates"]}), flush=True)
    check_hashes(pinned)
    comparisons = [
        compare(a, b) for a, b in zip(reports[0]["rows"], reports[1]["rows"], strict=True)
    ]
    result = {
        "scope": "fresh_ppo_forward_swing_reward_complete_development_evaluation",
        "input_sha256": pinned,
        "reports": reports,
        "comparisons": comparisons,
        "impact_resolution_consistent": all(not row["failed_checks"] for row in comparisons),
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (DIRECTORY / "trained_evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else run()
