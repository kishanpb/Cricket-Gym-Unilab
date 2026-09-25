"""Paired frozen-PPO lane feedback comparison; retain every physical outcome."""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import mjbatch.held_control
import numpy as np
import torch
from evaluate_g1_cricket_approach import resolution_comparison
from evaluate_g1_cricket_approach_learning import HANDS, RESOLUTIONS, evaluate_case
from omegaconf import OmegaConf
from report_g1_cricket_approach_learning import validate_media
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "g1_cricket_results/approach_feedback_v1"
COMMAND = "unilab.tasks.manipulation.g1_cricket.approach_lane.lane_command"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output):
    output.mkdir(exist_ok=False)
    parent_path = PARENT / "evaluation/summary.json"
    parent = json.loads(parent_path.read_text())
    inputs = dict(parent["input_sha256"])
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    runtime_hash = digest(Path(mjbatch.held_control.__file__))
    assert runtime_hash == parent["runtime_source_sha256"]["mjbatch.held_control"]
    inputs.update(
        {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                ROOT / "src/unilab/tasks/manipulation/g1_cricket/approach_lane.py",
                parent_path,
            )
        }
    )
    rows, media, pairs = [], {}, []
    for hand in HANDS:
        run = PARENT / f"ppo_{hand}"
        owner = OmegaConf.create(json.loads((run / "run_config.json").read_text())["config"])
        owner.env.observations.policy.terms.command.func = COMMAND
        checkpoint = ROOT / json.loads((run / "run_summary.json").read_text())["last_checkpoint"]
        for resolution, dt in RESOLUTIONS:
            owner.env.sim_dt = dt
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override["auto_reset"] = False
            env = create_env(owner, num_envs=1, env_cfg_override=override)
            try:
                env.reset(seed=1)
                wrapped = RslRlVecEnvWrapper(env, device="cpu")
                config = normalize_ppo_train_cfg(algo_config_dict(owner))
                config["logger"] = "none"
                runner = OnPolicyRunner(wrapped, config, log_dir=None, device="cpu")
                runner.load(
                    str(checkpoint),
                    load_cfg=dict(
                        actor=True, critic=False, optimizer=False, iteration=False, rnd=False
                    ),
                )
                policy = runner.get_inference_policy(device="cpu")

                def action_at():
                    with torch.inference_mode():
                        return policy(wrapped.get_observations()).numpy()

                print("START", hand, resolution, flush=True)
                row = evaluate_case(
                    env, action_at, output, "ppo", resolution, True, action_name="residual"
                )
                baseline = next(
                    r
                    for r in parent["rows"]
                    if r["hand"] == hand
                    and r["resolution"] == resolution
                    and r["controller"] == "ppo"
                )
                assert row["compiled_model_sha256"] == baseline["compiled_model_sha256"]
                with np.load(output / row["trace"]) as trace:
                    drift = np.abs(trace["steps"][:, 2] - trace["states"][0, 2]).max()
                    row["maximum_lateral_drift_m"] = float(drift) if np.isfinite(drift) else None
                rows.append(row)
                (output / "progress.json").write_text(
                    json.dumps(rows, indent=2, allow_nan=False) + "\n"
                )
                print(hand, resolution, row["failures"], flush=True)
            finally:
                env.close()
        pairs.append(
            resolution_comparison(*rows[-2:])
            if not any(row["evaluation_error"] for row in rows[-2:])
            else dict(hand=hand, passed=False, checks={"valid_pair": False})
        )
        media[hand] = (
            dict(status="blocked_nonfinite_states")
            if rows[-1].get("video_status") == "blocked_nonfinite_states"
            else validate_media(output, rows[-1])
        )
    for name, expected in inputs.items():
        assert digest(ROOT / name) == expected, name
    (output / "progress.json").unlink()
    result = dict(
        scope="frozen_local_PPO_and_external_locomotion_with_lateral_command_feedback_not_bowling",
        changed_axis="lateral velocity command = clip(-lane error / 1 second, -0.25, 0.25) m/s",
        command_function=COMMAND,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        training="No training; both approach_feedback_v1 final actors unchanged",
        source_parent=str(parent_path.relative_to(ROOT)),
        baseline_rows=[row for row in parent["rows"] if row["controller"] == "ppo"],
        rows=rows,
        resolution_comparisons=pairs,
        qualified=all(row["passed"] for row in rows + pairs) and len(rows) == 4,
        input_sha256=inputs,
        external_asset_sha256=parent["external_asset_sha256"],
        runtime_source_sha256={"mjbatch.held_control": runtime_hash},
        versions={name: version(name) for name in parent["versions"]},
        timing=parent["timing"],
        media=media,
        artifact_sha256={p.name: digest(p) for p in sorted(output.iterdir())},
        guard="Unchanged physical gates; no gather, release, running-delivery or independent Menagerie training claim.",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
