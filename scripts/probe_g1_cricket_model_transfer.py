"""Paired humanoid contact-model transfer with frozen scripted/zero controls."""

import argparse
import json
from importlib.metadata import version

import numpy as np
import torch
from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import ROOT, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_trial import ImpactReplay, trial
from omegaconf import OmegaConf
from probe_g1_cricket_reachability import CONFIG, action_at
from probe_g1_cricket_terminal_residual import CANDIDATE

from unilab.base.config_adapter import BackendAdapter, create_env

DIRECTORY = ROOT / "g1_cricket_results/model_transfer_v1"
PARENT = ROOT / "g1_cricket_results/elbow_v1/evaluation.json"
ISOLATED = ROOT / "g1_cricket_results/compliance_v1/evaluation.json"
TASKS = {"control_4ms": "G1CricketImpact", "candidate_2ms": "G1CricketImpactV2"}
PLAN = {
    "changed_axis": "bat_ball_positive_solref_time_constant",
    "models": list(TASKS),
    "time_constants_s": [0.004, 0.002],
    "timesteps_s": [0.0000625, 0.00003125],
    "executors": ["mujoco", "mjbatch"],
    "hands": ["right", "left"],
    "controllers": ["zero_residual", CANDIDATE["id"]],
    "seed": 4301,
    "offset_m": 0.0,
    "ticks": 100,
    "full_trial_count": 32,
    "left_control_scope": "same_normalized_commands_untrained_transfer_not_mirrored_skill",
}
NEW_INPUTS = (
    "src/unilab/tasks/__init__.py",
    "src/unilab/tasks/manipulation/g1_cricket/compliance_v2.py",
    "src/unilab/conf/ppo/task/g1_cricket_compliance_v2/mujoco.yaml",
    "src/unilab/conf/ppo/task/g1_cricket_compliance_v2/mjbatch.yaml",
    "scripts/probe_g1_cricket_model_transfer.py",
    "tests/scripts/test_g1_cricket_model_transfer.py",
    "docs/g1_cricket_model_transfer_v1.md",
)


def make_env(model_name, hand, dt, engine):
    owner = OmegaConf.create(json.loads(CONFIG.read_text())["config"])
    owner.training.task_name = TASKS[model_name]
    owner.env.sim_dt = dt
    owner.env.mujoco_substep_engine = "mjbatch" if engine == "mjbatch" else "rollout"
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    assert env.step_dt == 0.02 and env.max_episode_length == PLAN["ticks"]
    return env


def action(controller, tick):
    return (
        np.zeros((1, 7), dtype=np.float32)
        if controller == "zero_residual"
        else action_at(CANDIDATE, tick)
    )


def compare_group(rows, model_name, hand, controller):
    selected = [
        r
        for r in rows
        if r["model"] == model_name
        and r["outcome"]["hand"] == hand
        and r["outcome"]["controller"] == controller
    ]
    expected = {(engine, dt) for engine in PLAN["executors"] for dt in PLAN["timesteps_s"]}
    if len(selected) != 4 or {(r["engine"], r["sim_dt"]) for r in selected} != expected:
        raise ValueError("each comparison needs four unique model/hand/control trials")
    by_engine = {
        engine: [
            next(r for r in selected if r["engine"] == engine and r["sim_dt"] == dt)
            for dt in PLAN["timesteps_s"]
        ]
        for engine in PLAN["executors"]
    }
    keys = (
        "outcome",
        "terminated",
        "truncated",
        "blade_episode_count",
        "first_impact",
        "first_loaded_impact",
    )
    exact = all(
        all(a[key] == b[key] for key in keys)
        for a, b in zip(by_engine["mujoco"], by_engine["mjbatch"], strict=True)
    )
    comparison = compare(*(r["outcome"] for r in by_engine["mujoco"]))
    fixture_checks = []
    for key, absolute in (("fixture_peak_force_norm_n", 1.0), ("fixture_peak_torque_norm_nm", 0.1)):
        a, b = (r["outcome"][key] for r in by_engine["mujoco"])
        if abs(a - b) > max(absolute, abs(b) * 0.05):
            fixture_checks.append(key)
    passed = all(
        r["outcome"]["passed"]
        and r["outcome"]["seconds"] == 2
        and r["truncated"]
        and not r["terminated"]
        for r in selected
    )
    return {
        "model": model_name,
        "hand": hand,
        "controller": controller,
        "executor_evidence_exact": exact,
        "resolution_comparison": comparison,
        "fixture_resolution_failures": fixture_checks,
        "all_full_gates_pass": passed,
        "scripted_witness": passed
        and exact
        and not comparison["failed_checks"]
        and not fixture_checks,
    }


def model_contrasts(rows):
    contrasts = []
    for control in (r for r in rows if r["model"] == "control_4ms"):
        candidate = next(
            r
            for r in rows
            if r["model"] == "candidate_2ms"
            and r["engine"] == control["engine"]
            and r["sim_dt"] == control["sim_dt"]
            and all(
                r["outcome"][k] == control["outcome"][k]
                for k in ("hand", "controller", "offset_m", "seed")
            )
        )
        metrics = {}
        for key in (
            "maximum_blade_penetration_m",
            "first_separation_ball_vx_m_s",
            "fixture_peak_force_norm_n",
            "fixture_peak_torque_norm_nm",
            "blade_peak_force_n",
        ):
            a, b = (
                r["outcome"][key]
                if key != "blade_peak_force_n"
                else r["outcome"]["ball_contact_peak_force_norm_n"].get("bat_blade", 0)
                for r in (control, candidate)
            )
            metrics[key] = {
                "control": a,
                "candidate": b,
                "candidate_minus_control": None if a is None or b is None else b - a,
            }
        contrasts.append(
            {
                "engine": control["engine"],
                "sim_dt": control["sim_dt"],
                "hand": control["outcome"]["hand"],
                "controller": control["outcome"]["controller"],
                "metrics": metrics,
            }
        )
    return contrasts


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("model-transfer evidence already retained")
    parent = json.loads(PARENT.read_text())
    isolated = json.loads(ISOLATED.read_text())
    pins = {**parent["input_sha256"], **isolated["input_sha256"]}
    check_hashes(pins)
    assert not any(r["validation_failures"] for r in isolated["rows"])
    finest = [r for r in isolated["comparisons"] if r["coarse_dt"] == PLAN["timesteps_s"][0]]
    assert len(finest) == 4 and all(not r["failed_checks"] for r in finest)
    assert parent["executor"] == executor_manifest()
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    for name in (*NEW_INPUTS, str(PARENT.relative_to(ROOT)), str(ISOLATED.relative_to(ROOT))):
        pins[name] = sha256(ROOT / name)
    contract = {
        "scope": "paired_scripted_model_transfer_not_training_or_material_calibration",
        "plan": PLAN,
        "parent_candidate": CANDIDATE,
        "input_sha256": pins,
        "executor": parent["executor"],
        "versions": parent["versions"],
        "new_training": False,
        "policy_promoted": False,
        "physical_calibration_validated": False,
    }
    DIRECTORY.mkdir()
    (DIRECTORY / "preflight.json").write_text(json.dumps(contract, indent=2) + "\n")


def run():
    path, output = DIRECTORY / "preflight.json", DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError("model-transfer results already retained")
    contract = json.loads(path.read_text())
    pins = {**contract["input_sha256"], str(path.relative_to(ROOT)): sha256(path)}
    check_hashes(pins)
    assert contract["plan"] == PLAN and contract["parent_candidate"] == CANDIDATE
    assert contract["executor"] == executor_manifest()
    assert {name: version(name) for name in contract["versions"]} == contract["versions"]
    torch.set_num_threads(2)
    rows, models = [], []
    for engine in PLAN["executors"]:
        for dt in PLAN["timesteps_s"]:
            for hand in PLAN["hands"]:
                for model_name, tc in zip(PLAN["models"], PLAN["time_constants_s"], strict=True):
                    env = make_env(model_name, hand, dt, engine)
                    try:
                        replay = ImpactReplay(env)
                        assert replay.steps == round(0.02 / dt)
                        model = replay.model
                        np.testing.assert_array_equal(model.pair_solref, [[tc, 1]])
                        models.append(
                            {
                                "model": model_name,
                                "engine": engine,
                                "hand": hand,
                                "sim_dt": dt,
                                "pair_solref": model.pair_solref.tolist(),
                                "pair_solimp": model.pair_solimp.tolist(),
                                "pair_friction": model.pair_friction.tolist(),
                                "control_dt_s": env.step_dt,
                                "physics_substeps": replay.steps,
                            }
                        )
                        for controller in PLAN["controllers"]:
                            identity = dict(
                                hand=hand,
                                controller=controller,
                                offset_m=PLAN["offset_m"],
                                seed=PLAN["seed"],
                            )
                            result = trial(
                                env,
                                replay,
                                identity,
                                lambda tick: action(controller, tick),
                                PLAN["ticks"],
                            )
                            rows.append(
                                {"model": model_name, "engine": engine, "sim_dt": dt, **result}
                            )
                            print(
                                json.dumps(
                                    {
                                        "row": len(rows),
                                        "model": model_name,
                                        "engine": engine,
                                        "sim_dt": dt,
                                        "hand": hand,
                                        "controller": controller,
                                        "vx": result["outcome"]["first_separation_ball_vx_m_s"],
                                        "depth": result["outcome"]["maximum_blade_penetration_m"],
                                        "failures": result["outcome"]["failures"],
                                    }
                                ),
                                flush=True,
                            )
                    finally:
                        env.close()
    comparisons = [
        compare_group(rows, model, hand, controller)
        for model in PLAN["models"]
        for hand in PLAN["hands"]
        for controller in PLAN["controllers"]
    ]
    assert len(rows) == PLAN["full_trial_count"]
    check_hashes(pins)
    assert executor_manifest() == contract["executor"]
    output.write_text(
        json.dumps(
            {
                **contract,
                "input_sha256": pins,
                "models": models,
                "rows": rows,
                "comparisons": comparisons,
                "matched_timestep_model_contrasts": model_contrasts(rows),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else run()
