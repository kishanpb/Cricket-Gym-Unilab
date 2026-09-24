"""Bounded scripted residual reversal diagnostic; no policy training or promotion."""

import argparse
import itertools
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

from unilab.base.config_adapter import BackendAdapter, create_env

DIRECTORY = ROOT / "g1_cricket_results/reachability_v1"
PARENT = ROOT / "g1_cricket_results/native_mjbatch_v1/evaluation.json"
CONFIG = ROOT / "g1_cricket_results/tanh_v1/right/run_config.json"
PLAN = {
    "hand": "right",
    "seed": 4301,
    "offset_m": 0.0,
    "normalized_amplitude": 0.95,
    "first_switch_tick": 10,
    "refined_switch_ticks": [6, 8, 12, 14],
    "stop_tick": 20,
    "prefix_ticks": 25,
    "full_ticks": 100,
    "refinement_parents": 8,
    "full_candidates": 8,
    "maximum_prefixes": 161,
    "physics_dt_seconds": [0.00025, 0.000125],
    "executors": ["mujoco", "mjbatch"],
}


def schedule(signs, switch=10):
    bits = "".join("p" if s > 0 else "m" for s in signs)
    return {"id": f"{bits}_s{switch:02d}", "signs": list(signs), "switch_tick": switch}


def action_at(candidate, tick):
    direction = 1 if tick < candidate["switch_tick"] else -1
    normalized = np.asarray(candidate["signs"], dtype=np.float32) * np.float32(0.95)
    if tick >= PLAN["stop_tick"]:
        normalized[:] = 0
    return np.arctanh(direction * normalized).reshape(1, 7)


def prefix_safe(result):
    row = result["outcome"]
    allowed = {"native_episode_incomplete", "outgoing_velocity_not_above_1_m_s"}
    return (
        row["seconds"] == 0.5
        and not result["terminated"]
        and not result["truncated"]
        and row["first_separation_ball_vx_m_s"] is not None
        and not (set(row["failures"]) - allowed)
    )


def rank(results, *, require_forward=False):
    eligible = [
        r
        for r in results
        if r["candidate"]["id"] != "zero_residual"
        and prefix_safe(r)
        and (not require_forward or r["outcome"]["first_separation_ball_vx_m_s"] > 1)
    ]
    return sorted(
        eligible,
        key=lambda r: (-r["outcome"]["first_separation_ball_vx_m_s"], r["candidate"]["id"]),
    )


def make_env(hand, dt, engine):
    owner = OmegaConf.create(json.loads(CONFIG.read_text())["config"])
    if engine == "mjbatch":
        owner.env.mujoco_substep_engine = "mjbatch"
    owner.env.sim_dt = dt
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    assert env.max_episode_length == 100 and env.step_dt == 0.02
    return env


def execute(env, replay, candidate, ticks, hand="right"):
    return {
        "candidate": candidate,
        **trial(
            env,
            replay,
            dict(hand=hand, controller=candidate["id"], offset_m=0.0, seed=4301),
            lambda tick: action_at(candidate, tick),
            ticks,
        ),
    }


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("reachability evidence already exists")
    parent = json.loads(PARENT.read_text())
    pins = dict(parent["input_sha256"])
    check_hashes(pins)
    assert executor_manifest() == parent["executor"]
    for name in (
        "scripts/probe_g1_cricket_reachability.py",
        "scripts/g1_cricket_trial.py",
        "scripts/audit_g1_cricket_impact_reward.py",
        "scripts/evaluate_g1_cricket_impact_resolution.py",
        "tests/scripts/test_g1_cricket_reachability.py",
        "docs/g1_cricket_reachability_v1.md",
        str(PARENT.relative_to(ROOT)),
    ):
        pins[name] = sha256(ROOT / name)
    versions = {name: version(name) for name in parent["reports"][0]["versions"]}
    assert versions == parent["reports"][0]["versions"]
    result = {
        "scope": "scripted_local_controllability_probe_not_learned_policy",
        "plan": PLAN,
        "input_sha256": pins,
        "executor": executor_manifest(),
        "versions": versions,
        "new_training": False,
        "policy_promoted": False,
    }
    DIRECTORY.mkdir()
    (DIRECTORY / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")


def run():
    path, output = DIRECTORY / "preflight.json", DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError("reachability evaluation already retained")
    contract = json.loads(path.read_text())
    pins = {**contract["input_sha256"], str(path.relative_to(ROOT)): sha256(path)}
    check_hashes(pins)
    assert contract["plan"] == PLAN and executor_manifest() == contract["executor"]
    assert {name: version(name) for name in contract["versions"]} == contract["versions"]
    torch.set_num_threads(2)
    zero = dict(id="zero_residual", signs=[0] * 7, switch_tick=10)
    parent = json.loads(PARENT.read_text())
    baselines, prefixes, full, comparisons, witnesses = [], [], [], [], []
    # Validate the complete collector before selecting any new candidate.
    for engine in PLAN["executors"]:
        for index, dt in enumerate(PLAN["physics_dt_seconds"]):
            for hand in ("right", "left"):
                env = make_env(hand, dt, engine)
                try:
                    result = execute(env, ImpactReplay(env), zero, 100, hand)
                    reference = next(
                        r
                        for r in parent["reports"][index]["rows"]
                        if all(
                            result["outcome"][k] == r[k]
                            for k in ("hand", "controller", "offset_m", "seed")
                        )
                    )
                    assert result["outcome"] == reference, "zero baseline differs"
                    baselines.append({"engine": engine, "sim_dt": dt, **result})
                finally:
                    env.close()
    print(json.dumps({"exact_full_baselines": len(baselines)}), flush=True)
    env = make_env("right", 0.00025, "mujoco")
    try:
        replay = ImpactReplay(env)
        initial = [zero] + [schedule(s) for s in itertools.product((-1, 1), repeat=7)]
        for candidate in initial:
            prefixes.append(execute(env, replay, candidate, 25))
            print(
                json.dumps(
                    {
                        "prefix": len(prefixes),
                        "id": candidate["id"],
                        "vx": prefixes[-1]["outcome"]["first_separation_ball_vx_m_s"],
                        "failures": prefixes[-1]["outcome"]["failures"],
                    }
                ),
                flush=True,
            )
        refinement_parents = rank(prefixes)[: PLAN["refinement_parents"]]
        for source in refinement_parents:
            for switch in PLAN["refined_switch_ticks"]:
                candidate = schedule(source["candidate"]["signs"], switch)
                prefixes.append(execute(env, replay, candidate, 25))
        assert len(prefixes) <= PLAN["maximum_prefixes"]
    finally:
        env.close()
    selected = rank(prefixes, require_forward=True)[: PLAN["full_candidates"]]
    for result in selected:
        candidate = result["candidate"]
        candidate_rows = {}
        for engine in PLAN["executors"]:
            candidate_rows[engine] = []
            for dt in PLAN["physics_dt_seconds"]:
                env = make_env("right", dt, engine)
                try:
                    measured = execute(env, ImpactReplay(env), candidate, 100)
                    candidate_rows[engine].append(measured["outcome"])
                    full.append({"engine": engine, "sim_dt": dt, **measured})
                finally:
                    env.close()
        exact = candidate_rows["mujoco"] == candidate_rows["mjbatch"]
        comparison = compare(*candidate_rows["mujoco"])
        comparisons.append(
            {"candidate_id": candidate["id"], "executor_rows_exact": exact, **comparison}
        )
        if (
            exact
            and not comparison["failed_checks"]
            and all(r["passed"] for rows in candidate_rows.values() for r in rows)
        ):
            witnesses.append(candidate["id"])
    check_hashes(pins)
    assert executor_manifest() == contract["executor"]
    result = {
        "scope": contract["scope"],
        "plan": PLAN,
        "input_sha256": pins,
        "executor": contract["executor"],
        "versions": contract["versions"],
        "baselines": baselines,
        "prefixes": prefixes,
        "refinement_parent_ids": [r["candidate"]["id"] for r in refinement_parents],
        "selected_full_candidate_ids": [r["candidate"]["id"] for r in selected],
        "full_trials": full,
        "comparisons": comparisons,
        "scripted_witness_ids": witnesses,
        "new_training": False,
        "physical_calibration_validated": False,
        "policy_promoted": False,
        "negative_result_scope": "only this finite residual reversal family, right-hand seed4301 center toss; not a physical impossibility claim",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"prefixes": len(prefixes), "full_trials": len(full), "scripted_witness_ids": witnesses}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else run()
