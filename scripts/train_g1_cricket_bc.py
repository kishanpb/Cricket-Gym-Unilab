"""Retain imperfect demonstrations, fit only the actor, then run bounded PPO."""

import argparse
import copy
import csv
import json
import random
import time
from importlib.metadata import version

import numpy as np
import torch
from evaluate_g1_cricket_impact_resolution import compare
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import ROOT, SEEDS, load_policy, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_trial import ImpactReplay, trial
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from probe_g1_cricket_reachability import action_at
from probe_g1_cricket_terminal_residual import CANDIDATE
from retain_g1_training_diagnostics import retain
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.residual import TOSS_OFFSETS
from unilab.training import algo_config_dict
from unilab.training.run import resolve_nan_guard_cfg
from unilab.utils.nan_guard import NanGuard

DIRECTORY = ROOT / "g1_cricket_results/bc_v1"
RUN = DIRECTORY / "right"
PARENT = ROOT / "g1_cricket_results/model_transfer_v1/evaluation.json"
PLAN = {
    "training_seeds": list(range(5301, 5309)),
    "training_offsets_m": list(TOSS_OFFSETS),
    "teacher": CANDIDATE,
    "teacher_hand": "right",
    "teacher_episodes": 24,
    "bc_updates": 2000,
    "bc_samples_per_phase": 32,
    "bc_learning_rate": 0.001,
    "bc_seed": 1,
    "phase_tick_ranges": [[0, 10], [10, 20], [20, 100]],
    "ppo_iterations": 256,
    "ppo_transitions": 24576,
    "evaluation_seeds": list(SEEDS),
    "evaluation_hands": ["right", "left"],
    "evaluation_timesteps_s": [0.0000625, 0.00003125],
    "evaluation_executors": ["mujoco", "mjbatch"],
    "evaluation_controllers": ["zero_residual", "bc", "ppo"],
    "evaluation_rows": 576,
    "evaluation_scope": "reused_development_seeds_left_untrained_transfer_not_holdout",
}
NEW_INPUTS = (
    "scripts/train_g1_cricket_bc.py",
    "scripts/retain_g1_training_diagnostics.py",
    "src/unilab/training/run.py",
    "src/unilab/utils/nan_guard.py",
    "src/unilab/conf/ppo/task/g1_cricket_bc_v1/mujoco.yaml",
    "tests/scripts/test_g1_cricket_bc.py",
    "docs/g1_cricket_bc_v1.md",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def owner_config():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=["task=g1_cricket_bc_v1/mujoco"])


def validate_config(config, reference):
    expected = copy.deepcopy(reference)
    expected["training"].update(
        task_name="G1CricketImpactV2", log_dir="g1_cricket_results/bc_v1/right"
    )
    expected["env"]["sim_dt"] = 0.0000625
    expected["algo"].update(max_iterations=256, save_interval=256)
    if config != expected:
        raise ValueError("BC owner changed beyond contact model, timestep, budget and log path")


class StrictNanGuard(NanGuard):
    def check(self, obs, reward, step=0):
        if super().check(obs, reward, step) is not None:
            raise RuntimeError("nonfinite observation or reward before sanitization")

    def check_ctrl(self, ctrl, step=0):
        if super().check_ctrl(ctrl, step) is not None:
            raise RuntimeError("nonfinite control before physics")

    def capture(self, physics_state):
        if physics_state is not None and not np.isfinite(physics_state).all():
            raise RuntimeError("nonfinite physics state")
        super().capture(physics_state)


def make_env(owner, *, count=1, hand="right", dt=0.0000625, engine="mujoco", training=False):
    config = OmegaConf.create(OmegaConf.to_container(owner, resolve=True))
    config.env.sim_dt = dt
    config.env.mujoco_substep_engine = "mjbatch" if engine == "mjbatch" else "rollout"
    override = BackendAdapter(config, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=training)
    env = create_env(config, num_envs=count, env_cfg_override=override)
    guard_cfg = resolve_nan_guard_cfg(owner.training)
    assert guard_cfg is not None
    env.set_nan_guard(
        StrictNanGuard(guard_cfg, count, env.play_capabilities.supports_physics_state_playback)
    )
    assert env.max_episode_length == 100 and env.step_dt == 0.02
    return env


class CountedWrapper(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env, device="cpu")
        self.transitions = 0

    def step(self, actions):
        result = super().step(actions)
        self.transitions += self.num_envs
        return result


def make_runner(owner, env, log_dir=None):
    cfg = algo_config_dict(owner)
    runtime = resolve_rsl_rl_ppo_runtime(cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    assert runtime.wrapper_cls is RslRlVecEnvWrapper
    wrapped = CountedWrapper(env)
    train_cfg = normalize_ppo_train_cfg(cfg)
    train_cfg["logger"] = "tensorboard"
    runner = (runtime.runner_cls or OnPolicyRunner)(
        wrapped, train_cfg, log_dir=str(log_dir) if log_dir else None, device="cpu"
    )
    assert wrapped.num_obs == 115 and wrapped.num_actions == 7
    return wrapped, runner


def phases(ticks):
    indices = [
        torch.where((ticks >= low) & (ticks < high))[0] for low, high in PLAN["phase_tick_ranges"]
    ]
    if any(len(index) == 0 for index in indices):
        raise ValueError("demonstrations must contain every declared swing phase")
    return indices


def fit_actor(runner, samples, *, updates=2000):
    actor = runner.alg.get_policy()
    critic_before = copy.deepcopy(runner.alg.critic.state_dict())
    distribution_before = copy.deepcopy(actor.distribution.state_dict())
    optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
    assert not optimizer_before["state"] and runner.current_learning_iteration == 0
    groups = phases(samples["ticks"])
    generator = torch.Generator().manual_seed(PLAN["bc_seed"])
    optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=PLAN["bc_learning_rate"])
    losses = []
    for _ in range(updates):
        indices = torch.cat(
            [
                group[
                    torch.randint(len(group), (PLAN["bc_samples_per_phase"],), generator=generator)
                ]
                for group in groups
            ]
        )
        obs = TensorDict({"policy": samples["observations"][indices]}, batch_size=[len(indices)])
        prediction = actor(obs, stochastic_output=False)
        loss = torch.nn.functional.mse_loss(prediction, samples["raw_actions"][indices])
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite imitation loss")
        optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in actor.mlp.parameters()):
            raise RuntimeError("nonfinite imitation gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    for before, after in (
        (critic_before, runner.alg.critic.state_dict()),
        (distribution_before, actor.distribution.state_dict()),
    ):
        assert before.keys() == after.keys()
        assert all(torch.equal(before[key], after[key]) for key in before)
    assert runner.alg.optimizer.state_dict() == optimizer_before
    assert runner.current_learning_iteration == 0
    with torch.no_grad():
        prediction = actor(
            TensorDict({"policy": samples["observations"]}, batch_size=[len(samples["ticks"])]),
            stochastic_output=False,
        )
        per_sample = (prediction - samples["raw_actions"]).square().mean(dim=1)
    return {
        "losses": losses,
        "phase_sample_counts": [len(group) for group in groups],
        "phase_mse": [float(per_sample[group].mean()) for group in groups],
        "critic_unchanged": True,
        "distribution_unchanged": True,
        "ppo_optimizer_empty_unchanged": True,
        "ppo_iteration_before_learning": 0,
        "closed_loop_success_implied": False,
    }


def preflight():
    if DIRECTORY.exists():
        raise FileExistsError("BC experiment already retained")
    parent = json.loads(PARENT.read_text())
    pins = dict(parent["input_sha256"])
    check_hashes(pins)
    assert executor_manifest() == parent["executor"]
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    config = OmegaConf.to_container(owner_config(), resolve=True)
    reference = json.loads((ROOT / "g1_cricket_results/tanh_v1/right/run_config.json").read_text())
    validate_config(config, reference["config"])
    for name in (*NEW_INPUTS, str(PARENT.relative_to(ROOT))):
        pins[name] = sha256(ROOT / name)
    DIRECTORY.mkdir()
    write_json(
        DIRECTORY / "preflight.json",
        {
            "plan": PLAN,
            "config": config,
            "input_sha256": pins,
            "executor": parent["executor"],
            "versions": parent["versions"],
            "policy_promoted": False,
        },
    )


def load_contract():
    contract = json.loads((DIRECTORY / "preflight.json").read_text())
    check_hashes(contract["input_sha256"])
    assert contract["plan"] == PLAN
    assert executor_manifest() == contract["executor"]
    assert {name: version(name) for name in contract["versions"]} == contract["versions"]
    return contract


def collect():
    contract = load_contract()
    output = DIRECTORY / "demonstrations.pt"
    if output.exists() or (DIRECTORY / "teacher_evaluation.json").exists():
        raise FileExistsError("demonstrations already retained")
    env = make_env(OmegaConf.create(contract["config"]))
    wrapped = RslRlVecEnvWrapper(env, device="cpu")
    observations, actions, ticks, episodes, rows = [], [], [], [], []
    try:
        replay = ImpactReplay(env)
        for offset in PLAN["training_offsets_m"]:
            for seed in PLAN["training_seeds"]:
                episode = len(rows)

                def teacher(tick):
                    raw = action_at(CANDIDATE, tick)
                    observations.append(wrapped.get_observations()["policy"].clone())
                    actions.append(torch.from_numpy(raw.copy()))
                    ticks.append(tick)
                    episodes.append(episode)
                    return raw

                result = trial(
                    env,
                    replay,
                    dict(hand="right", controller="scripted_teacher", offset_m=offset, seed=seed),
                    teacher,
                    100,
                )
                rows.append(result)
                print(
                    json.dumps(
                        {
                            "teacher_episode": episode,
                            "outcome": {
                                key: result["outcome"][key]
                                for key in ("failures", "seconds", "return")
                            },
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()
    samples = {
        "observations": torch.cat(observations),
        "raw_actions": torch.cat(actions),
        "ticks": torch.tensor(ticks),
        "episode_ids": torch.tensor(episodes),
    }
    assert len(rows) == PLAN["teacher_episodes"]
    assert samples["observations"].shape == (len(ticks), 115)
    assert all(torch.isfinite(value).all() for value in samples.values())
    phases(samples["ticks"])
    torch.save(samples, output)
    write_json(
        DIRECTORY / "teacher_evaluation.json",
        {
            "scope": "all_imperfect_scripted_demonstrations_no_success_filter",
            "preflight_sha256": sha256(DIRECTORY / "preflight.json"),
            "dataset_sha256": sha256(output),
            "rows": rows,
            "samples": len(ticks),
        },
    )


def checkpoint_info(stage):
    return {
        "stage": stage,
        "preflight_sha256": sha256(DIRECTORY / "preflight.json"),
        "physical_model": "G1CricketImpactV2_2ms",
        "teacher_dataset_sha256": sha256(DIRECTORY / "demonstrations.pt"),
        "training_hand": "right",
    }


def validate_checkpoint(payload, expected):
    if payload.get("infos") != expected:
        raise ValueError("checkpoint physical-model or training provenance mismatch")


def validate_scalars(path, iterations):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["iteration"]) for row in rows] != list(range(iterations)):
        raise ValueError("training scalars do not cover every PPO iteration")
    for row in rows:
        for field in ("Loss/value", "Loss/surrogate", "Policy/mean_std"):
            if not row.get(field) or not np.isfinite(float(row[field])):
                raise ValueError(f"missing or nonfinite per-iteration scalar: {field}")


def train():
    contract = load_contract()
    if RUN.exists():
        raise FileExistsError("BC/PPO run already exists; inspect it instead of restarting")
    teacher = json.loads((DIRECTORY / "teacher_evaluation.json").read_text())
    assert teacher["dataset_sha256"] == sha256(DIRECTORY / "demonstrations.pt")
    assert teacher["preflight_sha256"] == sha256(DIRECTORY / "preflight.json")
    samples = torch.load(DIRECTORY / "demonstrations.pt", weights_only=True)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    owner = OmegaConf.create(contract["config"])
    env = make_env(owner, count=4, training=True)
    RUN.mkdir()
    write_json(RUN / "run_config.json", {"config": contract["config"]})
    try:
        env.reset(seed=1)
        wrapped, runner = make_runner(owner, env, RUN)
        bc = fit_actor(runner, samples, updates=PLAN["bc_updates"])
        write_json(RUN / "imitation.json", bc)
        # Use the runner's native payload without opening a second logging writer.
        torch.save({**runner.alg.save(), "iter": 0, "infos": checkpoint_info("bc")}, RUN / "bc.pt")
        started = time.monotonic()
        runner.learn(num_learning_iterations=PLAN["ppo_iterations"], init_at_random_ep_len=True)
        elapsed = time.monotonic() - started
        assert wrapped.transitions == PLAN["ppo_transitions"]
        assert runner.current_learning_iteration == PLAN["ppo_iterations"] - 1
        runner.save(str(RUN / "ppo.pt"), infos=checkpoint_info("ppo"))
        runner.logger.writer.flush()
        runner.logger.writer.close()
        retain(RUN)
        validate_scalars(RUN / "training_scalars.csv", PLAN["ppo_iterations"])
        diagnostics = json.loads((RUN / "training_diagnostics.json").read_text())
        assert diagnostics["scalar_tags"]
        assert all(row["all_finite"] for row in diagnostics["scalar_tags"].values())
        write_json(
            RUN / "run_summary.json",
            {
                "status": "completed",
                "configured_seed": 1,
                "effective_seed": 1,
                "run_env_steps": wrapped.transitions,
                "completed_updates": PLAN["ppo_iterations"],
                "last_iteration_index": runner.current_learning_iteration,
                "ppo_elapsed_seconds": elapsed,
                "policy_promoted": False,
                "checkpoints": {stage: sha256(RUN / f"{stage}.pt") for stage in ("bc", "ppo")},
                "final_action_std": runner.alg.get_policy().output_std.detach().cpu().tolist(),
            },
        )
        for path in (*RUN.glob("events.out.tfevents.*"), *RUN.glob("model_*.pt")):
            path.unlink()
    finally:
        env.close()


def evaluate():
    contract = load_contract()
    output = DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError("BC/PPO evaluation already retained")
    summary = json.loads((RUN / "run_summary.json").read_text())
    assert summary["status"] == "completed" and summary["run_env_steps"] == PLAN["ppo_transitions"]
    pins = dict(contract["input_sha256"])
    for path in (
        DIRECTORY / "preflight.json",
        DIRECTORY / "demonstrations.pt",
        DIRECTORY / "teacher_evaluation.json",
        RUN / "run_config.json",
        RUN / "run_summary.json",
        RUN / "imitation.json",
        RUN / "training_scalars.csv",
        RUN / "training_diagnostics.json",
        RUN / "bc.pt",
        RUN / "ppo.pt",
    ):
        pins[str(path.relative_to(ROOT))] = sha256(path)
    for stage in ("bc", "ppo"):
        assert sha256(RUN / f"{stage}.pt") == summary["checkpoints"][stage]
        validate_checkpoint(
            torch.load(RUN / f"{stage}.pt", weights_only=True), checkpoint_info(stage)
        )
    owner = OmegaConf.create(contract["config"])
    rows = []
    for engine in PLAN["evaluation_executors"]:
        for dt in PLAN["evaluation_timesteps_s"]:
            for hand in PLAN["evaluation_hands"]:
                env = make_env(owner, hand=hand, dt=dt, engine=engine)
                try:
                    replay = ImpactReplay(env)
                    pair = replay.model.pair("cricket_blade_impact_v1").id
                    np.testing.assert_array_equal(replay.model.pair_solref[pair], [0.002, 1])
                    for controller in PLAN["evaluation_controllers"]:
                        if controller != "zero_residual":
                            wrapped, policy = load_policy(owner, env, RUN / f"{controller}.pt")

                        def action(_tick):
                            with torch.inference_mode():
                                return (
                                    np.zeros((1, 7), dtype=np.float32)
                                    if controller == "zero_residual"
                                    else policy(wrapped.get_observations()).numpy()
                                )

                        for offset in PLAN["training_offsets_m"]:
                            for seed in PLAN["evaluation_seeds"]:
                                result = trial(
                                    env,
                                    replay,
                                    dict(
                                        hand=hand,
                                        controller=controller,
                                        offset_m=offset,
                                        seed=seed,
                                    ),
                                    action,
                                    100,
                                )
                                rows.append(dict(engine=engine, sim_dt=dt, **result))
                                print(
                                    json.dumps(
                                        {
                                            "row": len(rows),
                                            "engine": engine,
                                            "dt": dt,
                                            "hand": hand,
                                            "controller": controller,
                                            "failures": result["outcome"]["failures"],
                                        }
                                    ),
                                    flush=True,
                                )
                finally:
                    env.close()
    check_hashes(pins)
    comparisons = summarize(rows)
    write_json(
        output,
        {
            "scope": PLAN["evaluation_scope"],
            "input_sha256": pins,
            "rows": rows,
            "comparisons": comparisons,
            "policy_promoted": False,
            "physical_calibration_validated": False,
        },
    )


def summarize(rows):
    expected = {
        (e, dt, h, c, o, s)
        for e in PLAN["evaluation_executors"]
        for dt in PLAN["evaluation_timesteps_s"]
        for h in PLAN["evaluation_hands"]
        for c in PLAN["evaluation_controllers"]
        for o in PLAN["training_offsets_m"]
        for s in PLAN["evaluation_seeds"]
    }
    keyed = {
        (
            r["engine"],
            r["sim_dt"],
            *(r["outcome"][k] for k in ("hand", "controller", "offset_m", "seed")),
        ): r
        for r in rows
    }
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError("evaluation requires every unique declared context")
    comparisons = []
    for key in sorted(expected):
        engine, dt, hand, controller, offset, seed = key
        if engine != "mujoco" or dt != PLAN["evaluation_timesteps_s"][0]:
            continue
        four = [
            keyed[(e, t, hand, controller, offset, seed)]
            for e in PLAN["evaluation_executors"]
            for t in PLAN["evaluation_timesteps_s"]
        ]
        exact = all(
            {k: v for k, v in a.items() if k != "engine"}
            == {k: v for k, v in b.items() if k != "engine"}
            for a, b in zip(four[:2], four[2:], strict=True)
        )
        resolution = compare(*(r["outcome"] for r in four[:2]))
        fixture_failures = []
        for field, floor in (
            ("fixture_peak_force_norm_n", 1),
            ("fixture_peak_torque_norm_nm", 0.1),
        ):
            a, b = (r["outcome"][field] for r in four[:2])
            if abs(a - b) > max(floor, 0.05 * abs(b)):
                fixture_failures.append(field)
        comparisons.append(
            {
                "hand": hand,
                "controller": controller,
                "offset_m": offset,
                "seed": seed,
                "executor_evidence_exact": exact,
                "resolution_comparison": resolution,
                "fixture_resolution_failures": fixture_failures,
                "qualified": exact
                and not resolution["failed_checks"]
                and not fixture_failures
                and all(
                    r["outcome"]["passed"]
                    and r["outcome"]["seconds"] == 2
                    and r["truncated"]
                    and not r["terminated"]
                    for r in four
                ),
            }
        )
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "collect", "train", "evaluate"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"preflight": preflight, "collect": collect, "train": train, "evaluate": evaluate}[args.stage]()
