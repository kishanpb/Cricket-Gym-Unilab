"""Evaluate a native PPO smoke checkpoint without training or selected episodes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.training import algo_config_dict
from unilab.utils.sim2sim import policy_load_dim_guard
from unilab.visualization.interactive_playback import infer_checkpoint_actor_input_dim

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SEEDS = (4101, 4102, 4103, 4104)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(run_dir: Path) -> dict:
    saved = json.loads((run_dir / "run_config.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    checkpoint = ROOT / summary["last_checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    owner = OmegaConf.create(saved["config"])
    registry.ensure_registries()
    rows = []
    for hand in ("right", "left"):
        override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
        override.update(handedness=hand, auto_reset=False)
        env = registry.make(
            "G1CricketBatting", num_envs=1, sim_backend="mujoco", env_cfg_override=override
        )
        try:
            rl_cfg = algo_config_dict(owner)
            runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
            wrapped = runtime.wrapper_cls(env, device="cpu")
            train_cfg = normalize_ppo_train_cfg(rl_cfg)
            train_cfg["logger"] = "none"
            runner = (runtime.runner_cls or OnPolicyRunner)(
                wrapped, train_cfg, log_dir=None, device="cpu"
            )
            assert infer_checkpoint_actor_input_dim(str(checkpoint)) == wrapped.num_obs
            with policy_load_dim_guard(
                env_obs_dim=wrapped.num_obs, env_action_dim=29, algo_name="ppo"
            ):
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
            bat_contact = env.scene.bind_sensor_data(("ball_bat",))
            for policy_name in ("zero", "ppo"):
                for seed in EVALUATION_SEEDS:
                    env.reset(seed=seed)
                    episode_return = 0.0
                    contact_seen = False
                    max_joint_excess = 0.0
                    max_action = 0.0
                    for step in range(env.max_episode_length):
                        with torch.inference_mode():
                            action = (
                                policy(wrapped.get_observations())
                                if policy_name == "ppo"
                                else torch.zeros((1, 29))
                            )
                        if not torch.isfinite(action).all():
                            raise RuntimeError("nonfinite evaluation action")
                        max_action = max(max_action, float(action.abs().max()))
                        _, reward, done, _ = wrapped.step(action)
                        state = env.state
                        assert state is not None
                        episode_return += float(reward[0])
                        contact_seen |= bool(
                            (bat_contact.read().reshape(1, 4, 17)[..., 0] > 0).any()
                        )
                        joints = env.scene["robot"].data.joint_pos
                        limits = env.scene["robot"].data.soft_joint_pos_limits
                        max_joint_excess = max(
                            max_joint_excess,
                            float(
                                np.maximum(limits[..., 0] - joints, joints - limits[..., 1]).max()
                            ),
                        )
                        if bool(done[0]):
                            break
                    else:
                        raise RuntimeError(
                            "evaluation episode did not terminate within its declared horizon"
                        )
                    rows.append(
                        {
                            "policy": policy_name,
                            "hand": hand,
                            "seed": seed,
                            "steps": step + 1,
                            "seconds": (step + 1) * env.step_dt,
                            "return": episode_return,
                            "fallen": bool(state.terminated[0]),
                            "time_limit": bool(state.truncated[0]),
                            "bat_contact_seen_at_control_snapshots": contact_seen,
                            "max_joint_limit_excess_rad": max_joint_excess,
                            "max_raw_action_abs": max_action,
                        }
                    )
        finally:
            env.close()
    sources = [
        ROOT / "src/unilab/tasks/manipulation/g1_cricket" / name for name in ("task.py", "scene.py")
    ]
    sources += [ROOT / "src/unilab/conf/ppo/task/g1_cricket_batting/mujoco.yaml", Path(__file__)]
    return {
        "scope": "native_CPU_PPO_pipeline_smoke_not_trained_cricket",
        "training": {
            "actual_transitions": summary["run_env_steps"],
            "updates": summary["run_env_steps"] // summary["samples_per_iteration"],
            "last_zero_indexed_iteration": summary["completed_iterations"],
            "training_seed": summary["effective_seed"],
            "training_hand": "right",
            "learner_device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "python_cpu_count": os.cpu_count(),
            "reward": "upright minus action-rate penalty; no cricket reward",
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "actor_dimensions": [127, 64, 64, 29],
        },
        "task_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "rsl-rl-lib", "unilab-rl", "unisim-core", "mujoco")
        },
        "evaluation": {
            "seeds": EVALUATION_SEEDS,
            "hands": ["right", "left"],
            "policies": ["zero", "ppo"],
            "expected_episodes": 16,
            "action_selection": "deterministic actor mean; zero baseline",
            "reset_scope": "deterministic reset; seeds do not provide independent initial-state variation",
            "left_hand_scope": "untrained handedness transfer diagnostic",
            "contact_scope": "end-of-control sensor snapshots; not complete impact detection",
            "rows": rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run_dir)
    output = args.run_dir / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Retained {len(report['evaluation']['rows'])} episodes: {output}")


if __name__ == "__main__":
    main()
