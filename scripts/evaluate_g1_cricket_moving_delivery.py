"""Unfiltered bilateral full-sequence prototype evaluation, not trained cricket."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from evaluate_g1_cricket_running import render_rollout
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.moving_delivery import arm_weight, moving_command
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_case(owner, hand, dt, output, render):
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, sim_dt=dt, auto_reset=False)
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    try:
        env.reset(seed=1)
        term = env.action_manager.get_term("residual")
        replay, events = DeliveryReplay(env), DeliveryEvents(hand)
        physical = [env.get_physics_state_snapshot()[0].copy()]
        trace, controls, prior = [], [], []
        for tick in range(env.max_episode_length):
            time = float(physical[-1][0])
            command = moving_command(env)[0].copy()
            state = replay.step(env, np.zeros((1, 29), np.float32), events)
            snapshot = env.get_physics_state_snapshot()[0].copy()
            physical.append(snapshot)
            controls.append(term.processed_action[0].copy())
            prior.append(term.baseline_action[0].copy())
            trace.append(
                dict(
                    time_s=float(snapshot[0]),
                    pelvis_position_m=snapshot[1:4].tolist(),
                    command_m_s=command.tolist(),
                    arm_reference_weight=float(arm_weight(time)),
                    holder_peak_force_n=float(term.peak_load[0]),
                    holder_impulse_world_ns=term.impulse_world[0].tolist(),
                    hand_touch_fraction=float(term.touch_fraction[0]),
                    loaded_foot_peak_slip_m_s=term.slip_peak[0].tolist(),
                    released=bool(term.released[0]),
                )
            )
            if state.terminated[0] or state.truncated[0]:
                break
        result = events.finish(bool(state.truncated[0] and not state.terminated[0]))
        position = np.array([step["pelvis_position_m"] for step in trace])
        result.update(
            hand=hand,
            controller="live_prior_reference_arms",
            simulation_dt=dt,
            seed=1,
            steps=tick + 1,
            trace=trace,
            exact_endpoint_and_sensor_replay=True,
            approach_distance_m=float(position[-1, 0] - physical[0][1]),
            maximum_lateral_drift_m=float(np.abs(position[:, 1] - physical[0][2]).max()),
            maximum_loaded_foot_slip_m_s=np.max(
                [step["loaded_foot_peak_slip_m_s"] for step in trace], axis=0
            ).tolist(),
            termination_flags={
                name: bool(env.termination_manager.get_term(name)[0])
                for name in env.termination_manager.active_terms
            },
        )
        np.savez_compressed(output / "trace.npz", states=physical, controls=controls, prior=prior)
        if render:
            render_rollout(env, physical, output, result)
        (output / "evaluation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        return {key: value for key, value in result.items() if key != "trace"}
    finally:
        env.close()


def main(output, render):
    output.mkdir(exist_ok=False)
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=["task=g1_cricket_moving_delivery/mjbatch"])
    inputs = sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [
        Path(__file__),
        ROOT / "scripts/evaluate_g1_cricket_running.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
    ]
    inputs += list(
        (ROOT / "g1_cricket_results/running_fore_aft_support_v1").glob("*_dense_reference.npz")
    )
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    rows = []
    for hand in ("right", "left"):
        for resolution, dt in (("fine", 0.0000625), ("finest", 0.00003125)):
            directory = output / f"{hand}_{resolution}"
            directory.mkdir()
            print("START", hand, resolution, flush=True)
            row = evaluate_case(owner, hand, dt, directory, render)
            row["directory"] = directory.name
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    result = dict(
        scope="Continuous physics from rest through attempted delivery and recovery; no trained cricket policy or qualified showcase claim",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        config=OmegaConf.to_container(owner, resolve=True),
        input_sha256=hashes,
        external_asset_sha256=ASSET_HASHES,
        rows=rows,
        artifact_sha256={
            str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
        qualified_showcase=False,
        guard="Scheduled arm motion and release over an external learned locomotion prior, not locally learned running bowling. Every failed and early-terminated rollout is retained.",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(args.output, args.render)
