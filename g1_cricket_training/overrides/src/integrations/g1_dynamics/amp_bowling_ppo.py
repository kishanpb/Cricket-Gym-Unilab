"""Bounded SKRL PPO torque-residual training and complete bowling evaluation."""

import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import skrl
import torch
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

from .amp_bowling_env import AmpBowlingEnv, CASES, FLIGHT_REWARD_COLUMNS
from .amp_running_probe import JOINTS
from .groot_batting_ppo import make_agent, verify_inputs
from .twist2_cpu_probe import fingerprint


def validate_parity(proof, parent, *, release_curriculum=False, learn_release=False,
                    flight_curriculum=False, post_saturation_residual=False,
                    physics_backend="mujoco"):
    if (
        proof["status"] != "zero_residual_full_cohort_parity_passed"
        or proof.get("release_curriculum", False) != release_curriculum
        or proof.get("learn_release", False) != learn_release
        or proof.get("flight_curriculum", False) != flight_curriculum
        or proof.get("post_saturation_residual", False) != post_saturation_residual
        or physics_backend not in ("mujoco", "mjbatch")
        or proof.get("physics_backend", "mujoco") != physics_backend
        or Path(proof["parent"]).resolve() != parent.resolve()
        or len(proof["rows"]) != len(CASES)
        or {(r["result"]["hand"], r["result"]["timestep"]) for r in proof["rows"]}
        != set(CASES)
        or not all(r["states_exact"] and r["native_caps_pass"] and r["no_external_support"] for r in proof["rows"])
    ):
        raise ValueError("Complete source-bound bowling parity is required")
    verify_inputs(proof["inputs"])


def optimize(raw, steps, seed, *, gae_lambda=0.95):
    set_seed(seed)
    env = wrap_env(raw, wrapper="gymnasium", verbose=False)
    agent, cfg = make_agent(env, steps, log_std=-2.0, gae_lambda=gae_lambda)
    before = {
        name: torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        for name, model in agent.models.items()
    }
    updates = []
    hook = agent.optimizer.register_step_post_hook(lambda *args: updates.append(1))
    agent.enable_training_mode(True)
    obs, _ = env.reset()
    probes = [obs.clone()]
    try:
        for step in range(steps):
            agent.pre_interaction(timestep=step, timesteps=steps)
            with torch.no_grad():
                action, _ = agent.act(obs, None, timestep=step, timesteps=steps)
                next_obs, reward, terminal, truncated, info = env.step(action)
                assert not truncated.any()
                agent.record_transition(
                    observations=obs, states=None, actions=action, rewards=reward,
                    next_observations=next_obs, next_states=None,
                    terminated=terminal, truncated=truncated, infos=info,
                    timestep=step, timesteps=steps,
                )
            agent.post_interaction(timestep=step, timesteps=steps)
            obs = next_obs
            if terminal.any():
                print(json.dumps(dict(
                    steps=step + 1, episodes=len(raw.episodes), optimizer_steps=len(updates),
                    last_passed=raw.episodes[-1]["passed"],
                )), flush=True)
                probes.append(obs.clone())
                if step + 1 < steps:
                    obs, _ = env.reset()
        changes = {}
        for name, model in agent.models.items():
            vector = torch.cat([p.detach().flatten() for p in model.parameters()])
            assert torch.isfinite(vector).all()
            changes[name] = float(torch.linalg.vector_norm(vector - before[name]))
            assert changes[name] > 0
        assert len(updates) == steps // 400 * 16
        return agent, dict(
            cfg=cfg, optimizer_steps=len(updates), parameter_change_l2=changes,
            episodes=raw.episodes.copy(),
            unfinished_training_episode_steps=0 if terminal.any() else raw.steps,
        ), torch.cat(probes)
    finally:
        hook.remove()
        env.close()


def leg_residual_mask(*, learn_release=False):
    mask = np.array([
        not any(part in name for part in ("hip", "knee", "ankle"))
        for name in JOINTS
    ], dtype=np.float32)
    return np.r_[mask, np.float32(1)] if learn_release else mask


def evaluate(agent, raw, output, *, action_mask=None):
    mask = np.ones(raw.action_space.shape) if action_mask is None else action_mask
    rows = []
    agent.enable_training_mode(False)
    for hand, dt in CASES:
        obs, _ = raw.reset(options={"hand": hand, "timestep": dt})
        observations = []
        policy_means = []
        terminal = False
        while not terminal:
            observations.append(obs.copy())
            with torch.no_grad():
                action = agent.policy.compute({"observations": torch.from_numpy(obs)[None]})[0][0].numpy()
            policy_means.append(action.copy())
            obs, _, terminal, truncated, _ = raw.step(action * mask)
            assert raw.observation_space.contains(obs) and not truncated
        trace = output / (raw.result["case"] + ".npz")
        record = dict(
            raw.records, residual_observations=np.asarray(observations),
            policy_mean_rows=np.asarray(policy_means),
        )
        for value in record.values():
            assert np.isfinite(value).all()
        assert record["metrics"][:, 8].max() == 0
        assert record["metrics"][:, 7].max() <= 1 + 1e-9
        applied_actions = record["residual_rows"]
        if "release_action_rows" in record:
            applied_actions = np.column_stack((applied_actions, record["release_action_rows"]))
        for observation, mean, expected in zip(
            observations, policy_means, applied_actions, strict=True
        ):
            with torch.no_grad():
                action = agent.policy.compute({"observations": torch.from_numpy(observation)[None]})[0][0].numpy()
            np.testing.assert_array_equal(action, mean)
            np.testing.assert_array_equal(action * mask, expected)
        np.savez_compressed(trace, **record)
        rows.append(dict(
            **raw.result, trace=fingerprint(trace), peak_eligible_held_speed_m_s=raw.peak_speed,
            residual_actions_exact=len(observations), physics_samples=len(record["metrics"]),
        ))
        print(json.dumps({key: rows[-1][key] for key in (
            "case", "terminal", "passed", "release", "peak_eligible_held_speed_m_s"
        )}), flush=True)
    return rows


def archive_inputs(inputs, output):
    with zipfile.ZipFile(output / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(inputs):
            path = Path(name)
            if path.suffix in {".py", ".xml", ".md"}:
                archive.write(path, arcname=path.relative_to(Path.cwd()) if path.is_absolute() else path)


def evaluate_checkpoint(upstream, unilab, parent, parity, checkpoint_dir, output, *, freeze_legs=False):
    proof = json.loads(parity.read_text())
    baseline = json.loads((checkpoint_dir / "summary.json").read_text())
    release_curriculum = baseline.get("release_curriculum", False)
    learn_release = baseline.get("learn_release", False)
    flight_curriculum = baseline.get("flight_curriculum", False)
    post_saturation_residual = baseline.get("post_saturation_residual", False)
    physics_backend = baseline.get("physics_backend", "mujoco")
    validate_parity(proof, parent, release_curriculum=release_curriculum, learn_release=learn_release,
                    flight_curriculum=flight_curriculum, post_saturation_residual=post_saturation_residual,
                    physics_backend=physics_backend)
    if Path(baseline["parent"]).resolve() != parent.resolve():
        raise ValueError("Checkpoint must use the same bowling parent")
    runner = str(Path(__file__).resolve().relative_to(Path.cwd()))
    changed_sources = {runner, "tests/test_amp_bowling_ppo.py"}
    verify_inputs({k: v for k, v in baseline["inputs"].items() if k not in changed_sources})
    for name, expected in baseline["artifacts"].items():
        assert fingerprint(checkpoint_dir / name) == expected, name
    inputs = {name: fingerprint(Path(name)) for name in baseline["inputs"]}
    for path in [checkpoint_dir / "summary.json", *(checkpoint_dir / n for n in baseline["artifacts"])]:
        inputs[str(path)] = fingerprint(path)
    output.mkdir(parents=True, exist_ok=False)
    archive_inputs(inputs, output)
    torch.set_num_threads(1)
    started = time.monotonic()
    raw = AmpBowlingEnv(
        upstream, unilab, parent, release_curriculum=release_curriculum,
        learn_release=learn_release,
        flight_curriculum=flight_curriculum,
        post_saturation_residual=post_saturation_residual,
        physics_backend=physics_backend,
    )
    agent, _ = make_agent(raw, baseline["steps"], log_std=-2.0)
    checkpoint = checkpoint_dir / "final.pt"
    agent.load(str(checkpoint))
    mask = leg_residual_mask(learn_release=learn_release) if freeze_legs else np.ones(raw.action_space.shape)
    try:
        rows = evaluate(agent, raw, output, action_mask=mask)
    finally:
        raw.close()
    verify_inputs(inputs)
    result = dict(
        scope="Evaluation-only closed-loop action-mask ablation of the same checkpoint; no retraining, native-leg-controller freeze, or promotion",
        parent=str(parent), checkpoint_dir=str(checkpoint_dir), inputs=inputs,
        checkpoint=fingerprint(checkpoint), action_mask=mask.tolist(),
        freeze_leg_residuals=freeze_legs, rows=rows,
        release_curriculum=release_curriculum,
        learn_release=learn_release,
        flight_curriculum=flight_curriculum,
        post_saturation_residual=post_saturation_residual,
        physics_backend=physics_backend,
        elapsed_s=time.monotonic() - started, promotion_allowed=False,
        reward_trace_columns=FLIGHT_REWARD_COLUMNS if flight_curriculum else None,
        artifacts={p.name: fingerprint(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def run(
    upstream, unilab, parent, parity, output, *, steps=1600, seed=0,
    release_curriculum=False, gae_lambda=0.95, learn_release=False, flight_curriculum=False,
    post_saturation_residual=False,
    physics_backend="mujoco",
):
    if steps <= 0 or steps % 400:
        raise ValueError("Budget must be a positive multiple of 400 PPO transitions")
    if not 0 <= gae_lambda <= 1:
        raise ValueError("GAE lambda must be in [0, 1]")
    proof = json.loads(parity.read_text())
    validate_parity(proof, parent, release_curriculum=release_curriculum, learn_release=learn_release,
                    flight_curriculum=flight_curriculum, post_saturation_residual=post_saturation_residual,
                    physics_backend=physics_backend)
    inputs = dict(proof["inputs"])
    for path in (Path(__file__).resolve().relative_to(Path.cwd()), parity,
                 Path("tests/test_amp_bowling_ppo.py"), Path("tests/test_amp_bowling_env.py")):
        inputs[str(path)] = fingerprint(path)
    output.mkdir(parents=True, exist_ok=False)
    archive_inputs(inputs, output)
    started = time.monotonic()
    torch.set_num_threads(1)
    raw = AmpBowlingEnv(
        upstream, unilab, parent, release_curriculum=release_curriculum,
        learn_release=learn_release,
        flight_curriculum=flight_curriculum,
        post_saturation_residual=post_saturation_residual,
        physics_backend=physics_backend,
    )
    agent, training, probes = optimize(raw, steps, seed, gae_lambda=gae_lambda)
    checkpoint = output / "final.pt"
    agent.save(str(checkpoint))
    restored, _ = make_agent(raw, steps, log_std=-2.0, gae_lambda=gae_lambda)
    restored.load(str(checkpoint))
    with torch.no_grad():
        torch.testing.assert_close(
            agent.policy.compute({"observations": probes})[0],
            restored.policy.compute({"observations": probes})[0], rtol=0, atol=0,
        )
    try:
        rows = evaluate(restored, raw, output)
    finally:
        raw.close()
    verify_inputs(inputs)
    result = dict(
        scope="CPU PPO torque residual over frozen AMP/GR00T and reference arms with a mechanical holder; not learned finger grasp",
        parent=str(parent), parity=str(parity), inputs=inputs, steps=steps, seed=seed,
        release_curriculum=release_curriculum,
        learn_release=learn_release,
        flight_curriculum=flight_curriculum,
        post_saturation_residual=post_saturation_residual,
        physics_backend=physics_backend,
        gae_lambda=gae_lambda,
        training=training, rows=rows, checkpoint=fingerprint(checkpoint),
        checkpoint_action_roundtrip_exact=True, skrl_version=skrl.__version__,
        reward_trace_columns=FLIGHT_REWARD_COLUMNS if flight_curriculum else None,
        torch_version=torch.__version__, elapsed_s=time.monotonic() - started,
        promotion_allowed=False,
        artifacts={p.name: fingerprint(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "unilab", "parent", "parity", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--release-curriculum", action="store_true")
    parser.add_argument("--learn-release", action="store_true")
    parser.add_argument("--flight-curriculum", action="store_true")
    parser.add_argument("--post-saturation-residual", action="store_true")
    parser.add_argument("--backend", choices=("mujoco", "mjbatch"), default="mujoco")
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--freeze-leg-residuals", action="store_true")
    args = parser.parse_args()
    if args.evaluate_checkpoint is not None:
        evaluate_checkpoint(
            args.upstream, args.unilab, args.parent, args.parity,
            args.evaluate_checkpoint, args.output, freeze_legs=args.freeze_leg_residuals,
        )
    elif args.freeze_leg_residuals:
        parser.error("--freeze-leg-residuals requires --evaluate-checkpoint")
    else:
        run(
            args.upstream, args.unilab, args.parent, args.parity, args.output,
            steps=args.steps, seed=args.seed, release_curriculum=args.release_curriculum,
            gae_lambda=args.gae_lambda,
            learn_release=args.learn_release,
            flight_curriculum=args.flight_curriculum,
            post_saturation_residual=args.post_saturation_residual,
            physics_backend=args.backend,
        )
