"""Complete deterministic development-cohort evaluation of a saved PPO policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import zipfile

import mujoco
import numpy as np
import skrl
import torch

from integrations.g1_dynamics.groot_batting_env import EPISODE_STEPS, GrootBattingEnv
from integrations.g1_dynamics.groot_articulated_probe import CONFIG
from integrations.g1_dynamics.groot_batting_ppo import Policy, verify_inputs
from integrations.g1_dynamics.twist2_cpu_probe import fingerprint
from unilab.tasks.manipulation.g1_cricket.articulated_learning import (
    ARTICULATED_DELIVERY_POOL,
)


def case_keys():
    return [
        (hand, feed)
        for hand in ("right", "left")
        for feed in range(len(ARTICULATED_DELIVERY_POOL))
    ]


def cohort_metrics(rows):
    keys = [(r["hand"], r["feed_index"]) for r in rows]
    if len(keys) != len(case_keys()) or set(keys) != set(case_keys()):
        raise ValueError("A complete, duplicate-free two-hand feed cohort is required")
    failures = {}
    for row in rows:
        for name, passed in row["physical"]["checks"].items():
            if not passed:
                failures[name] = failures.get(name, 0) + 1
    return {
        "cases": len(rows),
        "physical_passes": sum(r["physical"]["passed"] for r in rows),
        "qualified_hits": sum(r["qualified_hit"] for r in rows),
        "one_pitch_hit_flags": sum(r["valid_one_pitch_hit"] for r in rows),
        "raw_boundary_cases": sum(r["boundary_runs"] > 0 for r in rows),
        "qualified_boundary_cases": sum(
            r["qualified_hit"] and r["boundary_runs"] > 0 for r in rows
        ),
        "physical_failures": failures,
    }


def mean_action(policy, observation):
    with torch.no_grad():
        action = policy.compute({"observations": torch.from_numpy(observation)[None]})[
            0
        ]
    return action[0].numpy()


def evaluate_case(job, *, scene_file=None, env_class=GrootBattingEnv):
    (
        upstream,
        checkout,
        baseline,
        checkpoint,
        output,
        hand,
        feed,
        recovery_fade,
        env_options,
    ) = job
    started = time.monotonic()
    torch.set_num_threads(1)
    env = env_class(
        upstream,
        checkout,
        baseline / "scene.xml" if scene_file is None else scene_file,
        recovery_fade=recovery_fade,
        **env_options,
    )
    policy = Policy(env.observation_space, env.action_space)
    policy.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["policy"]
    )
    policy.eval()
    observation, _ = env.reset(seed=0, options={"hand": hand, "feed_index": feed})
    observations, actions, rewards = [observation.copy()], [], []
    for step in range(EPISODE_STEPS):
        action = mean_action(policy, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        assert terminated == (step == EPISODE_STEPS - 1) and not truncated
        assert env.observation_space.contains(observation) and np.isfinite(reward)
        observations.append(observation.copy())
        actions.append(action.copy())
        rewards.append(reward)
    np.testing.assert_array_equal(np.asarray(actions), np.asarray(env.actions))
    case = f"{hand}_feed{feed}"
    trace = output / f"{case}.npz"
    np.savez_compressed(
        trace,
        observations=observations,
        residual_actions=actions,
        rewards=rewards,
        states=env.states,
        controls=env.controls,
        policy=env.controller.policy_rows,
        ball_metrics=env.audit.metrics,
        invalid_ball_contacts=env.audit.invalid_metrics,
        ball_time_s=env.audit.sample_times,
        integrator_code=np.asarray(env.audit.integrator_codes, dtype=np.uint8),
    )
    contacts = output / f"{case}_contacts.json"
    contacts.write_text(json.dumps(env.tactile, allow_nan=False) + "\n")
    row = {
        "case": case,
        "hand": hand,
        "feed_index": feed,
        **env.result,
        "training_reward_return": float(sum(rewards)),
        "max_abs_residual_action": float(np.abs(actions).max()),
        "integration": env.schedule.report(),
        "trace": {"path": str(trace), **fingerprint(trace)},
        "contacts": {"path": str(contacts), **fingerprint(contacts)},
        "wall_seconds": time.monotonic() - started,
    }
    env.close()
    return row


def verified_training(training, baseline):
    report = json.loads((training / "training.json").read_text())
    checkpoint = training / "final.pt"
    if fingerprint(checkpoint) != report["checkpoint"]:
        raise ValueError("Checkpoint differs from the recorded training result")
    parity = Path(report["parity"]["path"])
    if fingerprint(parity) != {
        k: report["parity"][k] for k in ("size_bytes", "sha256")
    }:
        raise ValueError("Training parity evidence changed")
    proof = json.loads(parity.read_text())
    if report.get("recovery_fade", False) != proof.get("recovery_fade", False):
        raise ValueError("Training controller differs from its parity configuration")
    if report.get("num_envs", 1) != proof.get("workers", 1):
        raise ValueError("Training worker count differs from its parity configuration")
    if report.get("environment_options", {}) != proof.get(
        "environment_options", {}
    ) or report.get("action_dim", 17) != proof.get("action_dim", 17):
        raise ValueError("Training action/reference configuration differs from parity")
    verify_inputs(proof["inputs"])
    for name in ("summary.json", "scene.xml"):
        if not any(
            Path(p).resolve() == (baseline / name).resolve() for p in proof["inputs"]
        ):
            raise ValueError("Baseline must match the source-bound training inputs")
    parent = json.loads((baseline / "summary.json").read_text())
    if (
        parent["mujoco_version"] != mujoco.__version__
        or report["torch_version"] != torch.__version__
        or report["skrl_version"] != skrl.__version__
    ):
        raise ValueError(
            "Evaluation requires the recorded simulator and policy runtime versions"
        )
    cohort_metrics(parent["rows"])
    return checkpoint, parity, proof, parent


def evaluation_options(training_options, swing_power=None):
    options = dict(training_options)
    if swing_power is not None:
        if not np.isfinite(swing_power) or not 0 <= swing_power <= 1:
            raise ValueError("Evaluation swing power must be finite and in [0, 1]")
        options["swing_power"] = swing_power
    return options


def evaluate(upstream, checkout, baseline, training, output, workers=2, *, swing_power=None):
    checkpoint, parity, proof, parent = verified_training(training, baseline)
    training_options = proof.get("environment_options", {})
    env_options = evaluation_options(training_options, swing_power)
    selected = [upstream / CONFIG] + [
        checkout / f"g1_cricket_results/articulated_swing_compact/{hand}_reference.npz"
        for hand in ("right", "left")
    ]
    recorded = {Path(name).resolve() for name in proof["inputs"]}
    if any(path.resolve() not in recorded for path in selected):
        raise ValueError(
            "Controller and reference paths must match the training inputs"
        )
    output.mkdir(parents=True, exist_ok=False)
    extra = [
        Path(__file__),
        Path("tests/test_groot_batting_evaluate.py"),
        training / "training.json",
        checkpoint,
        parity,
    ]
    inputs = {**proof["inputs"], **{str(p): fingerprint(p) for p in extra}}
    with zipfile.ZipFile(parity.parent / "source_bundle.zip") as old:
        sources = {name: old.read(name) for name in old.namelist()}
    for path in extra:
        if path != checkpoint:
            sources[str(path.resolve().relative_to(Path.cwd()))] = path.read_bytes()
    with zipfile.ZipFile(
        output / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name, content in sorted(sources.items()):
            archive.writestr(name, content)
    jobs = [
        (
            upstream,
            checkout,
            baseline,
            checkpoint,
            output,
            hand,
            feed,
            proof.get("recovery_fade", False),
            env_options,
        )
        for hand, feed in case_keys()
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate_case, jobs):
            rows.append(row)
            print(
                json.dumps(
                    {
                        "case": row["case"],
                        "qualified_hit": row["qualified_hit"],
                        "physical_pass": row["physical"]["passed"],
                        "boundary_runs": row["boundary_runs"],
                    }
                ),
                flush=True,
            )
    verify_inputs(inputs)
    report = {
        "scope": "All fourteen development feeds; deterministic PPO mean residuals over frozen GR00T balance/reference swing. Not held-out validation or a public-ready showcase.",
        "checkpoint": {"path": str(checkpoint), **fingerprint(checkpoint)},
        "recovery_fade": proof.get("recovery_fade", False),
        "environment_options": env_options,
        "training_environment_options": training_options,
        "evaluation_intervention": (
            {
                "axis": "reference_swing_power",
                "training_value": training_options.get("swing_power", 0.0),
                "evaluation_value": swing_power,
                "scope": "Fixed checkpoint with changed reference timing; not newly learned power or matched-training evaluation.",
            }
            if env_options != training_options else None
        ),
        "action_dim": proof.get("action_dim", 17),
        "training_num_envs": proof.get("workers", 1),
        "baseline_metrics": cohort_metrics(parent["rows"]),
        "ppo_metrics": cohort_metrics(rows),
        "inputs": inputs,
        "rows": rows,
        "promotion_allowed": False,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "checkout", "baseline", "training", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--swing-power", type=float)
    args = parser.parse_args()
    evaluate(
        args.upstream,
        args.checkout,
        args.baseline,
        args.training,
        args.output,
        args.workers,
        swing_power=args.swing_power,
    )
