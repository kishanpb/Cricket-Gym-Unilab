"""Bounded wrist-pitch shaping with unchanged physics and full shot gates."""

import argparse
import json
from importlib.metadata import version

import numpy as np
import torch
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import ROOT, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_trial import ImpactReplay, trial
from probe_g1_cricket_reachability import action_at as parent_action
from probe_g1_cricket_reachability import make_env
from probe_g1_cricket_terminal_residual import CANDIDATE, compare_scale

DIRECTORY = ROOT / "g1_cricket_results/wrist_pitch_v1"
PARENT = ROOT / "g1_cricket_results/terminal_residual_v1/evaluation.json"
PLAN = {
    "changed_axis": "forward_phase_wrist_pitch_normalized_residual_scale",
    "scales": [1.0, 0.75, 0.5, 0.0, -0.5, -0.75, -1.0],
    "changed_ticks": list(range(10, 20)),
    "arm_channel": 5,
    "joint": "right_wrist_pitch_joint",
    "hand": "right",
    "offset_m": 0.0,
    "seed": 4301,
    "ticks": 100,
    "physics_dt_seconds": [0.00025, 0.000125],
    "executors": ["mujoco", "mjbatch"],
    "full_trial_count": 28,
}


def action_at(scale, tick):
    action = parent_action(CANDIDATE, tick)
    if tick in PLAN["changed_ticks"] and scale != 1:
        action[0, PLAN["arm_channel"]] = np.arctanh(np.float32(-0.95) * np.float32(scale))
    return action


def impact_direction(result):
    sample = result["first_loaded_impact"]
    if sample is None:
        return None
    contacts = []
    for contact in sample["contacts"]:
        normal = np.array(contact["normal_on_ball_world"])
        relative = np.array(contact["ball_point_velocity_world_m_s"]) - np.array(
            contact["bat_point_velocity_world_m_s"]
        )
        components = relative * normal
        np.testing.assert_allclose(
            components.sum(), contact["relative_normal_velocity_m_s"], rtol=1e-12, atol=1e-12
        )
        contacts.append(
            {
                "normal_on_ball_world": normal.tolist(),
                "relative_normal_components_m_s": components.tolist(),
                "relative_normal_velocity_m_s": float(components.sum()),
            }
        )
    return {"solve_seconds": sample["solve_seconds"], "contacts": contacts}


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("wrist-pitch evidence already retained")
    parent = json.loads(PARENT.read_text())
    pins = dict(parent["input_sha256"])
    check_hashes(pins)
    assert executor_manifest() == parent["executor"]
    for name in (
        str(PARENT.relative_to(ROOT)),
        "scripts/probe_g1_cricket_wrist_pitch.py",
        "tests/scripts/test_g1_cricket_wrist_pitch.py",
        "docs/g1_cricket_wrist_pitch_v1.md",
    ):
        pins[name] = sha256(ROOT / name)
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    result = {
        "scope": "scripted_wrist_pitch_probe_not_learned_control",
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
        raise FileExistsError("wrist-pitch results already retained")
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
                term = env.action_manager.get_term("residual")
                actuator = int(term.arm_ids[PLAN["arm_channel"]])
                joint = replay.model.actuator_trnid[actuator, 0]
                assert replay.model.joint(joint).name == PLAN["joint"]
                for scale in PLAN["scales"]:
                    controller = (
                        CANDIDATE["id"] if scale == 1 else f"{CANDIDATE['id']}_pitch_{scale:+.2f}"
                    )
                    identity = dict(
                        hand=PLAN["hand"],
                        controller=controller,
                        offset_m=PLAN["offset_m"],
                        seed=PLAN["seed"],
                    )
                    result = trial(
                        env, replay, identity, lambda tick: action_at(scale, tick), PLAN["ticks"]
                    )
                    if scale == 1:
                        reference = next(
                            r
                            for r in parent["rows"]
                            if r["engine"] == engine and r["sim_dt"] == dt and r["scale"] == 1
                        )
                        assert result == {k: reference[k] for k in result}, "parent trial changed"
                    rows.append(
                        {
                            "scale": scale,
                            "engine": engine,
                            "sim_dt": dt,
                            **result,
                            "impact_direction": impact_direction(result),
                        }
                    )
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
