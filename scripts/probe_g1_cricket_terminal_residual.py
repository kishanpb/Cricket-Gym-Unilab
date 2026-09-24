"""Full-trial attenuation of one terminal arm command, without changing physics."""

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
from probe_g1_cricket_reachability import action_at as parent_action
from probe_g1_cricket_reachability import make_env

DIRECTORY = ROOT / "g1_cricket_results/terminal_residual_v1"
PARENT = ROOT / "g1_cricket_results/reachability_v1/evaluation.json"
CANDIDATE = dict(id="pmppppp_s10", signs=[1, -1, 1, 1, 1, 1, 1], switch_tick=10)
PLAN = {
    "changed_axis": "tick_19_normalized_residual_scale",
    "scales": [0.0, 0.25, 0.5, 0.75, 0.875, 1.0],
    "changed_tick": 19,
    "hand": "right",
    "offset_m": 0.0,
    "seed": 4301,
    "ticks": 100,
    "physics_dt_seconds": [0.00025, 0.000125],
    "executors": ["mujoco", "mjbatch"],
    "full_trial_count": 24,
}


def action_at(scale, tick):
    action = parent_action(CANDIDATE, tick)
    if tick == PLAN["changed_tick"] and scale != 1:
        normalized = np.asarray(CANDIDATE["signs"], dtype=np.float32) * np.float32(-0.95)
        action = np.arctanh(normalized * np.float32(scale)).reshape(1, 7)
    return action


def controller_id(scale):
    return CANDIDATE["id"] if scale == 1 else f"{CANDIDATE['id']}_t19_{scale:.3f}"


def compare_scale(rows, scale):
    selected = [r for r in rows if r["scale"] == scale]
    expected = {(engine, dt) for engine in PLAN["executors"] for dt in PLAN["physics_dt_seconds"]}
    if len(selected) != 4 or {(r["engine"], r["sim_dt"]) for r in selected} != expected:
        raise ValueError("each scale requires four unique complete executor/timestep trials")
    by_engine = {
        engine: [
            next(r for r in selected if r["engine"] == engine and r["sim_dt"] == dt)
            for dt in PLAN["physics_dt_seconds"]
        ]
        for engine in PLAN["executors"]
    }
    exact = all(
        all(
            a[key] == b[key]
            for key in (
                "outcome",
                "terminated",
                "truncated",
                "blade_episode_count",
                "first_impact",
                "first_loaded_impact",
            )
        )
        for a, b in zip(by_engine["mujoco"], by_engine["mjbatch"], strict=True)
    )
    comparison = compare(*(r["outcome"] for r in by_engine["mujoco"]))
    passes = all(
        r["outcome"]["passed"]
        and r["outcome"]["seconds"] == 2
        and r["truncated"]
        and not r["terminated"]
        for r in selected
    )
    return {
        "scale": scale,
        "executor_evidence_exact": exact,
        "resolution_comparison": comparison,
        "all_full_gates_pass": passes,
        "scripted_witness": passes and exact and not comparison["failed_checks"],
    }


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("terminal residual experiment already retained")
    parent = json.loads(PARENT.read_text())
    pins = dict(parent["input_sha256"])
    check_hashes(pins)
    assert parent["selected_full_candidate_ids"] == [CANDIDATE["id"]]
    assert executor_manifest() == parent["executor"]
    for name in (
        str(PARENT.relative_to(ROOT)),
        "scripts/probe_g1_cricket_terminal_residual.py",
        "tests/scripts/test_g1_cricket_terminal_residual.py",
        "docs/g1_cricket_terminal_residual_v1.md",
    ):
        pins[name] = sha256(ROOT / name)
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    result = {
        "scope": "scripted_terminal_command_attenuation_not_learned_control",
        "plan": PLAN,
        "parent_candidate": CANDIDATE,
        "input_sha256": pins,
        "executor": parent["executor"],
        "versions": parent["versions"],
        "new_training": False,
        "policy_promoted": False,
    }
    DIRECTORY.mkdir()
    (DIRECTORY / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")


def run():
    path, output = DIRECTORY / "preflight.json", DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError("terminal residual results already retained")
    contract = json.loads(path.read_text())
    pins = {**contract["input_sha256"], str(path.relative_to(ROOT)): sha256(path)}
    check_hashes(pins)
    assert contract["plan"] == PLAN and contract["parent_candidate"] == CANDIDATE
    assert contract["executor"] == executor_manifest()
    assert {name: version(name) for name in contract["versions"]} == contract["versions"]
    torch.set_num_threads(2)
    parent = json.loads(PARENT.read_text())
    rows = []
    for engine in PLAN["executors"]:
        for dt in PLAN["physics_dt_seconds"]:
            env = make_env(PLAN["hand"], dt, engine)
            try:
                replay = ImpactReplay(env)
                # The unchanged command is replayed first, before each set of alternatives.
                for scale in reversed(PLAN["scales"]):
                    identity = dict(
                        hand=PLAN["hand"],
                        controller=controller_id(scale),
                        offset_m=PLAN["offset_m"],
                        seed=PLAN["seed"],
                    )
                    result = trial(
                        env, replay, identity, lambda tick: action_at(scale, tick), PLAN["ticks"]
                    )
                    if scale == 1:
                        reference = next(
                            r
                            for r in parent["full_trials"]
                            if r["engine"] == engine and r["sim_dt"] == dt
                        )
                        assert result == {k: reference[k] for k in result}, "parent trial changed"
                    rows.append({"scale": scale, "engine": engine, "sim_dt": dt, **result})
                    print(
                        json.dumps(
                            {
                                "scale": scale,
                                "engine": engine,
                                "sim_dt": dt,
                                "vx": result["outcome"]["first_separation_ball_vx_m_s"],
                                "penetration_m": result["outcome"]["maximum_blade_penetration_m"],
                                "failures": result["outcome"]["failures"],
                            }
                        ),
                        flush=True,
                    )
            finally:
                env.close()
    assert len(rows) == PLAN["full_trial_count"]
    comparisons = [compare_scale(rows, scale) for scale in PLAN["scales"]]
    check_hashes(pins)
    assert contract["executor"] == executor_manifest()
    result = {
        **contract,
        "input_sha256": pins,
        "rows": rows,
        "comparisons": comparisons,
        "scripted_witness_scales": [c["scale"] for c in comparisons if c["scripted_witness"]],
        "physical_calibration_validated": False,
        "generalization_validated": False,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"full_trials": len(rows), "scripted_witness_scales": result["scripted_witness_scales"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight() if args.preflight else run()
