"""Full start-to-terminal running PPO/reference evaluation with native substep replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]


def evaluate(directory):
    saved = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    owner = OmegaConf.create(saved["config"])
    checkpoint = Path(summary["last_checkpoint"])
    output = directory / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    inputs = [
        directory / "run_config.json",
        checkpoint,
        Path(__file__),
        ROOT / "scripts/g1_cricket_delivery_trial.py",
        ROOT / owner.env.commands.motion.params.motion_file,
        ROOT / owner.env.actions.reference.reference_file,
        ROOT / "src/unilab/assets/robots/g1/g1.xml",
        ROOT / "src/unilab/assets/robots/g1/scene_flat.xml",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    registry.ensure_registries()
    env = registry.make(
        owner.training.task_name, num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    rows = []
    try:
        env.reset(seed=1)
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        cfg = normalize_ppo_train_cfg(algo_config_dict(owner))
        cfg["logger"] = "none"
        runner = OnPolicyRunner(wrapped, cfg, log_dir=None, device="cpu")
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
        policy = runner.get_inference_policy(device="cpu")
        for controller in ("reference_only", "ppo"):
            env.reset(seed=1)
            replay = DeliveryReplay(env, action_name="reference")
            events = DeliveryEvents(env.cfg.handedness)
            physical = [env.get_physics_state_snapshot()[0].copy()]
            trace, total = [], 0.0
            for tick in range(env.max_episode_length):
                with torch.inference_mode():
                    action = (
                        policy(wrapped.get_observations())
                        if controller == "ppo"
                        else torch.zeros((1, 29))
                    )
                if not torch.isfinite(action).all():
                    raise RuntimeError("non-finite running policy action")
                state = replay.step(env, action.numpy(), events)
                term = env.action_manager.get_term("reference")
                total += float(state.reward[0])
                snapshot = env.get_physics_state_snapshot()[0].copy()
                physical.append(snapshot)
                trace.append(
                    {
                        "time_s": float(snapshot[0]),
                        "pelvis_position_m": snapshot[1:4].tolist(),
                        "holder_peak_force_n": float(term.peak_load[0]),
                        "holder_impulse_world_ns": term.impulse_world[0].tolist(),
                        "hand_touch_fraction": float(term.touch_fraction[0]),
                        "released": bool(term.released[0]),
                        "reward": float(state.reward[0]),
                    }
                )
                if state.terminated[0] or state.truncated[0]:
                    break
            result = events.finish(bool(state.truncated[0] and not state.terminated[0]))
            result.update(
                hand=env.cfg.handedness,
                controller=controller,
                seed=1,
                steps=tick + 1,
                episode_return=total,
                exact_endpoint_and_sensor_replay=True,
                trace=trace,
                termination_flags={
                    name: bool(env.termination_manager.get_term(name)[0])
                    for name in env.termination_manager.active_terms
                },
            )
            rows.append(result)
            np.savez_compressed(directory / f"{controller}_physical.npz", state=physical)
    finally:
        env.close()
    for path, digest in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("running evaluation input changed")
    result = {
        "scope": "whole_body_PPO_tracking_with_scheduled_release_not_qualified_bowling",
        "input_sha256": hashes,
        "rows": rows,
        "evaluation_pool": "both controls, deterministic start, seed 1; no heldout/generalization claim",
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print([{k: v for k, v in row.items() if k != "trace"} for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    evaluate(parser.parse_args().directory)
