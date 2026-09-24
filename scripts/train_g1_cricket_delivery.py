"""Fixed-budget, separately trained right/left CPU PPO delivery pilot."""

import argparse
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import rsl_rl
import torch
import uni_rl
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryReplay, trial
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from retain_g1_training_diagnostics import retain
from rsl_rl.runners import OnPolicyRunner
from train_g1_cricket_bc import CountedWrapper, StrictNanGuard, validate_scalars, write_json
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES
from unilab.training import algo_config_dict
from unilab.training.run import resolve_nan_guard_cfg

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "g1_cricket_results/delivery_v1"
PLAN = dict(
    hands=["right", "left"],
    training_seed=1,
    iterations=256,
    num_envs=4,
    rollout_steps=24,
    transitions_per_hand=24576,
    engine="mjbatch",
    evaluation_seeds=[6301, 6302],
    evaluation_dt=[0.00025, 0.000125],
    evaluation_engines=["mujoco", "mjbatch"],
    controllers=["zero", "ppo"],
    evaluation_rows=32,
    heldout_claim=False,
    policy_promoted=False,
)
VERSIONS = (
    "mujoco",
    "mjbatch",
    "numpy",
    "onnxruntime",
    "torch",
    "rsl-rl-lib",
    "unilab-rl",
    "unisim-core",
)


def runtime_sources():
    return {
        package.__name__: {
            str(path.relative_to(Path(package.__file__).parent)): sha256(path)
            for path in sorted(Path(package.__file__).parent.rglob("*.py"))
        }
        for package in (uni_rl, rsl_rl)
    }


def owner_config():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=["task=g1_cricket_delivery_v1/mjbatch"])


def make_env(owner, hand, *, count=1, dt=0.00025, engine="mjbatch", training=False):
    cfg = OmegaConf.create(OmegaConf.to_container(owner, resolve=True))
    override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    override.update(
        handedness=hand,
        auto_reset=training,
        sim_dt=dt,
        mujoco_substep_engine=engine if engine == "mjbatch" else "rollout",
    )
    env = create_env(cfg, num_envs=count, env_cfg_override=override)
    guard = resolve_nan_guard_cfg(cfg.training)
    if guard is None:
        raise ValueError("delivery training requires finite-state guards")
    env.set_nan_guard(
        StrictNanGuard(guard, count, env.play_capabilities.supports_physics_state_playback)
    )
    return env


def make_runner(owner, env, directory=None):
    cfg = algo_config_dict(owner)
    runtime = resolve_rsl_rl_ppo_runtime(cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    if runtime.wrapper_cls is not RslRlVecEnvWrapper:
        raise ValueError("unexpected delivery wrapper")
    wrapped = CountedWrapper(env)
    train_cfg = normalize_ppo_train_cfg(cfg)
    train_cfg["logger"] = "tensorboard" if directory else "none"
    runner = (runtime.runner_cls or OnPolicyRunner)(
        wrapped, train_cfg, log_dir=str(directory) if directory else None, device="cpu"
    )
    assert wrapped.num_obs == 122 and wrapped.num_actions == 8
    return wrapped, runner


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("delivery experiment already exists")
    owner = owner_config()
    assert (owner.algo.num_envs, owner.algo.num_steps_per_env, owner.algo.max_iterations) == (
        4,
        24,
        256,
    )
    paths = set()
    for folder in ("base", "managers", "envs", "training", "tasks/manipulation/g1_cricket"):
        paths.update((ROOT / "src/unilab" / folder).rglob("*.py"))
    for name in (
        "scripts/train_g1_cricket_delivery.py",
        "scripts/g1_cricket_delivery_trial.py",
        "scripts/train_g1_cricket_bc.py",
        "scripts/retain_g1_training_diagnostics.py",
        "scripts/evaluate_g1_cricket_mjbatch.py",
        "scripts/evaluate_g1_cricket_residual.py",
        "src/unilab/tasks/__init__.py",
        "src/unilab/utils/rotation.py",
        "src/unilab/utils/nan_guard.py",
        "src/unilab/assets/robots/g1/g1.xml",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "src/unilab/conf/ppo/config.yaml",
        "docs/g1_cricket_delivery_v1.md",
        "tests/scripts/test_g1_cricket_delivery.py",
        "tests/scripts/test_g1_cricket_delivery_learning.py",
    ):
        paths.add(ROOT / name)
    for task in ("g1_cricket_prior_v1", "g1_cricket_bowling_v1", "g1_cricket_delivery_v1"):
        paths.update((ROOT / "src/unilab/conf/ppo/task" / task).glob("*.yaml"))
    record = dict(
        plan=PLAN,
        config=OmegaConf.to_container(owner, resolve=True),
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)},
        versions={name: version(name) for name in VERSIONS},
        executor=executor_manifest(),
        runtime_source_sha256=runtime_sources(),
        external_prior=ASSET_HASHES,
        scope="from_scratch_cricket_actor_over_frozen_locomotion_prior",
    )
    DIRECTORY.mkdir(parents=True)
    write_json(DIRECTORY / "preflight.json", record)


def contract():
    record = json.loads((DIRECTORY / "preflight.json").read_text())
    if (
        record["plan"] != PLAN
        or record["executor"] != executor_manifest()
        or record["versions"] != {name: version(name) for name in VERSIONS}
        or record["runtime_source_sha256"] != runtime_sources()
    ):
        raise ValueError("delivery plan/runtime changed")
    for name, expected in record["input_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"delivery input changed: {name}")
    return record


def checkpoint_info(hand):
    return dict(
        hand=hand,
        preflight_sha256=sha256(DIRECTORY / "preflight.json"),
        stage="final_ppo",
        observations=122,
        actions=8,
    )


def train(hand):
    record = contract()
    run = DIRECTORY / hand
    if run.exists():
        raise FileExistsError("inspect retained delivery run instead of restarting")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    owner = OmegaConf.create(record["config"])
    env = make_env(owner, hand, count=4, training=True)
    run.mkdir()
    try:
        env.reset(seed=1)
        wrapped, runner = make_runner(owner, env, run)
        initial = {k: v.detach().clone() for k, v in runner.alg.get_policy().state_dict().items()}
        started = time.monotonic()
        runner.learn(num_learning_iterations=PLAN["iterations"], init_at_random_ep_len=False)
        elapsed = time.monotonic() - started
        assert wrapped.transitions == PLAN["transitions_per_hand"]
        assert runner.current_learning_iteration == PLAN["iterations"] - 1
        final = runner.alg.get_policy().state_dict()
        assert all(torch.isfinite(v).all() for v in final.values())
        delta = sum(float((final[k] - initial[k]).square().sum()) for k in initial)
        assert delta > 0
        runner.save(str(run / "ppo.pt"), infos=checkpoint_info(hand))
        runner.logger.writer.flush()
        runner.logger.writer.close()
        retain(run)
        validate_scalars(run / "training_scalars.csv", PLAN["iterations"])
        contract()
        write_json(
            run / "run_summary.json",
            dict(
                status="completed",
                hand=hand,
                seed=1,
                updates=PLAN["iterations"],
                transitions=wrapped.transitions,
                elapsed_seconds=elapsed,
                actor_state_squared_change=delta,
                checkpoint_sha256=sha256(run / "ppo.pt"),
                final_action_std=runner.alg.get_policy().output_std.detach().cpu().tolist(),
                qualification="requires_complete_independent_evaluation",
                policy_promoted=False,
            ),
        )
        for path in (*run.glob("events.out.tfevents.*"), *run.glob("model_*.pt")):
            path.unlink()
        print(json.dumps(json.loads((run / "run_summary.json").read_text()), indent=2))
    finally:
        env.close()


def compare_rows(rows):
    expected = {
        (h, c, s, dt, e)
        for h in PLAN["hands"]
        for c in PLAN["controllers"]
        for s in PLAN["evaluation_seeds"]
        for dt in PLAN["evaluation_dt"]
        for e in PLAN["evaluation_engines"]
    }
    indexed = {
        (r["hand"], r["controller"], r["seed"], r["dt"], r["engine"]): r["outcome"] for r in rows
    }
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("incomplete or duplicate delivery evaluation pool")
    pairs = []
    for h in PLAN["hands"]:
        for c in PLAN["controllers"]:
            for seed in PLAN["evaluation_seeds"]:
                outcomes = [
                    indexed[h, c, seed, dt, e]
                    for dt in PLAN["evaluation_dt"]
                    for e in PLAN["evaluation_engines"]
                ]
                if outcomes[0] != outcomes[1] or outcomes[2] != outcomes[3]:
                    raise ValueError("delivery executor outcome mismatch")
                coarse, fine = outcomes[0], outcomes[2]
                stable = coarse["failures"] == fine["failures"]
                for key, absolute in (
                    ("peak_holder_force_n", 1.0),
                    ("maximum_ball_penetration_m", 0.0005),
                ):
                    stable &= abs(coarse[key] - fine[key]) <= max(
                        absolute, 0.05 * max(abs(coarse[key]), abs(fine[key]))
                    )
                for key, absolute in (
                    ("release", 0.1),
                    ("first_bounce", 0.1),
                    ("target_crossing", 0.1),
                ):
                    a, b = coarse[key], fine[key]
                    if (a is None) != (b is None):
                        stable = False
                    elif a is not None:
                        stable &= bool(
                            np.max(np.abs(np.array(a["position"]) - b["position"])) <= absolute
                        )
                        if key == "release":
                            stable &= bool(
                                np.max(np.abs(np.array(a["velocity"]) - b["velocity"])) <= 0.1
                            )
                pairs.append(
                    dict(
                        hand=h,
                        controller=c,
                        seed=seed,
                        exact_executors=True,
                        timestep_stable=bool(stable),
                        qualified=bool(stable and all(o["passed"] for o in outcomes)),
                    )
                )
    return pairs


def evaluate():
    record = contract()
    output = DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    owner = OmegaConf.create(record["config"])
    rows = []
    checkpoints = {h: sha256(DIRECTORY / h / "ppo.pt") for h in PLAN["hands"]}
    for hand in PLAN["hands"]:
        checkpoint = DIRECTORY / hand / "ppo.pt"
        summary = json.loads((DIRECTORY / hand / "run_summary.json").read_text())
        if sha256(checkpoint) != summary["checkpoint_sha256"]:
            raise ValueError("delivery checkpoint hash changed")
        if torch.load(checkpoint, weights_only=True)["infos"] != checkpoint_info(hand):
            raise ValueError("delivery checkpoint provenance mismatch")
        for dt in PLAN["evaluation_dt"]:
            for engine in PLAN["evaluation_engines"]:
                env = make_env(owner, hand, dt=dt, engine=engine)
                try:
                    wrapped, runner = make_runner(owner, env)
                    runner.load(
                        str(checkpoint),
                        load_cfg=dict(
                            actor=True, critic=False, optimizer=False, iteration=False, rnd=False
                        ),
                    )
                    policy = runner.get_inference_policy(device="cpu")
                    replay = DeliveryReplay(env)
                    for controller in PLAN["controllers"]:
                        for seed in PLAN["evaluation_seeds"]:

                            def action_at(tick):
                                del tick
                                with torch.inference_mode():
                                    return (
                                        policy(wrapped.get_observations()).numpy()
                                        if controller == "ppo"
                                        else np.zeros((1, 8), np.float32)
                                    )

                            outcome = trial(env, replay, action_at, seed)
                            row = dict(
                                hand=hand,
                                controller=controller,
                                seed=seed,
                                dt=dt,
                                engine=engine,
                                outcome=outcome,
                            )
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                finally:
                    env.close()
    comparisons = compare_rows(rows)
    contract()
    if checkpoints != {h: sha256(DIRECTORY / h / "ppo.pt") for h in PLAN["hands"]}:
        raise ValueError("delivery checkpoint changed during evaluation")
    write_json(
        output,
        dict(
            scope="complete_development_pilot_not_hardware_certification",
            plan=PLAN,
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            checkpoints=checkpoints,
            rows=rows,
            comparisons=comparisons,
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--hand", choices=PLAN["hands"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "preflight":
        preflight()
    elif args.stage == "train":
        if args.hand is None:
            parser.error("train requires --hand")
        train(args.hand)
    else:
        evaluate()
