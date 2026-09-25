"""Full from-rest approach gates for both final PPO actors and zero residual."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import mjbatch.held_control
import mujoco
import numpy as np
import torch
from audit_g1_cricket_running_stance import foot_loads
from evaluate_g1_cricket_approach import (
    COLUMNS,
    ApproachEvents,
    LoadedFootSlip,
    resolution_comparison,
    summarize,
)
from evaluate_g1_cricket_tracking import visual_model
from g1_cricket_delivery_trial import DeliveryReplay
from omegaconf import OmegaConf
from retarget_g1_cricket_running import render_poses
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]
HANDS = ("right", "left")
CONTROLLERS = ("reference", "ppo")
RESOLUTIONS = (("fine", 0.0000625), ("finest", 0.00003125))


def qualification(rows):
    pairs, qualified = [], {}
    for controller in CONTROLLERS:
        candidate = [row for row in rows if row["controller"] == controller]
        coverage = sorted((r["hand"], r["resolution"]) for r in candidate) == sorted(
            (hand, resolution) for hand in HANDS for resolution, _ in RESOLUTIONS
        )
        comparisons = []
        for hand in HANDS:
            group = [r for r in candidate if r["hand"] == hand]
            group.sort(key=lambda r: r["resolution"])
            if len(group) == 2 and not any(r.get("evaluation_error") for r in group):
                pair = resolution_comparison(*group)
            else:
                pair = dict(hand=hand, seed=1, passed=False, checks={"valid_pair": False})
            pair["controller"] = controller
            comparisons.append(pair)
        qualified[controller] = coverage and all(r["passed"] for r in candidate + comparisons)
        pairs.extend(comparisons)
    return dict(
        resolution_comparisons=pairs,
        ppo_approach_qualified=qualified["ppo"],
        reference_approach_qualified=qualified["reference"],
        all_controller_checks_pass=all(qualified.values()),
    )


def evaluate_case(
    env, action_at, output, controller, resolution, render, *, action_name="reference"
):
    env.reset(seed=1)
    replay = DeliveryReplay(env, action_name=action_name)
    model = replay.model
    hand = env.cfg.handedness
    slip, events = LoadedFootSlip(model), ApproachEvents(hand)
    wrist = model.body(f"{hand}_wrist_yaw_link").id
    ball = model.body("cricket_ball").id
    offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
    dofs = model.jnt_dofadr[replay.joints]
    metrics, controls, actions, loads = [], [], [], []
    states = [env.get_physics_state_snapshot()[0].copy()]
    episode_return, sensing_error = 0.0, 0.0

    def observe(model, data):
        held = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset
        metrics.append(
            np.r_[
                data.time,
                data.qpos[:3],
                data.qvel[:3],
                np.linalg.norm(data.qvel[3:6]),
                np.max(np.abs(data.qvel[dofs])),
                foot_loads(model, data),
                np.linalg.norm(data.xpos[ball] - held),
                np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1]),
                slip.measure(model, data),
            ]
        )

    error = None
    try:
        for _ in range(400):
            begin = len(metrics)
            action = action_at()
            actions.append(action[0].copy())
            try:
                state = replay.step(env, action, events, observer=observe)
            finally:
                term = env.action_manager.get_term(action_name)
                controls.append(term.processed_action[0].copy())
                loads.append(float(term.peak_load[0]))
                states.append(env.get_physics_state_snapshot()[0].copy())
            assert env.equality_constraints.get_equality_active().all() and not term.released.any()
            speeds = np.asarray(metrics[begin:])[:, 13:15]
            measured = np.square(speeds).mean(axis=0)
            sensing_error = max(
                sensing_error, float(np.max(np.abs(term.slip_squared[0] - measured)))
            )
            np.testing.assert_allclose(term.slip_squared[0], measured, atol=1e-5, rtol=1e-5)
            episode_return += float(state.reward[0])
            if state.terminated[0] or state.truncated[0]:
                break
        assert all(np.isfinite(x).all() for x in (states, metrics, controls, actions, loads))
        assert np.isfinite(episode_return)
    except (AssertionError, FloatingPointError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and str(exc) != "invalid delivery replay":
            raise
        error = dict(type=type(exc).__name__, message=str(exc))
    states, metrics = np.asarray(states), np.asarray(metrics)
    if error is None:
        result = summarize(
            events,
            metrics,
            states,
            float(-model.body_mass.sum() * model.opt.gravity[2]),
            bool(len(controls) == 400 and state.truncated[0] and not state.terminated[0]),
        )
    else:
        result = dict(
            passed=False,
            checks={"complete": False, "evaluation_valid": False},
            failures=["evaluation_error"],
            duration_s=float(states[-1, 0]) if np.isfinite(states[-1, 0]) else None,
        )
    name = f"{hand}_{controller}_{resolution}"
    compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=compiled)
    result.update(
        hand=hand,
        controller=controller,
        seed=1,
        resolution=resolution,
        physics_dt_s=env.physics_dt,
        episode_return=episode_return if np.isfinite(episode_return) else None,
        trace=f"{name}.npz",
        substeps=len(metrics),
        maximum_slip_cost_sensor_error=sensing_error,
        evaluation_error=error,
        exact_endpoint_and_sensor_replay=error is None,
        compiled_model_sha256=hashlib.sha256(compiled.tobytes()).hexdigest(),
    )
    np.savez_compressed(
        output / result["trace"],
        states=states,
        steps=metrics,
        actions=actions,
        controls=controls,
        holder_peak_force_n=loads,
    )
    if render and controller == "ppo" and resolution == "finest":
        if not np.isfinite(states).all():
            result["video_status"] = "blocked_nonfinite_states"
            return result
        visual = visual_model(Path(env.scene_directory.name) / "cricket.xml", model)
        visual.vis.global_.offwidth, visual.vis.global_.offheight = 960, 540
        render_poses(
            visual,
            states[:, 1 : 1 + model.nq],
            hand,
            output / f"{hand}_ppo_approach.mp4",
            "WHOLE-BODY PPO APPROACH, NOT BOWLING",
            subtitle=(
                "Final actor | measured-command residual | held ball | full outcome | 0.5x"
                if action_name == "reference"
                else "Local whole-body residual + external locomotion prior | held ball | 0.5x"
            ),
        )
        result["video_status"] = "partial_error_outcome" if error else "full_episode_outcome"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--action-name", choices=("reference", "residual"), default="reference")
    args = parser.parse_args()
    output = args.directory / "evaluation"
    output.mkdir(exist_ok=False)
    inputs = [Path(__file__), ROOT / "src/unilab/tasks/__init__.py"]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    source = json.loads((ROOT / "g1_cricket_results/approach_teacher_v1/summary.json").read_text())
    inputs += [ROOT / p for p in source["input_sha256"]]
    for task in (
        "g1_cricket_measured_approach",
        "g1_cricket_approach_learning",
        "g1_cricket_approach_feedback",
    ):
        inputs.append(ROOT / f"src/unilab/conf/ppo/task/{task}/mjbatch.yaml")
    configurations = {}
    for hand in HANDS:
        directory = args.directory / f"ppo_{hand}"
        owner = OmegaConf.create(json.loads((directory / "run_config.json").read_text())["config"])
        run = json.loads((directory / "run_summary.json").read_text())
        checkpoint = Path(run["last_checkpoint"])
        configurations[hand] = (owner, checkpoint)
        inputs += [directory / "run_config.json", directory / "run_summary.json", checkpoint]
        if args.action_name == "reference":
            inputs += [
                ROOT / owner.env.commands.motion.reference_file,
                ROOT / owner.env.commands.motion.params.motion_file,
            ]
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    runtime = Path(mjbatch.held_control.__file__)
    runtime_hash = hashlib.sha256(runtime.read_bytes()).hexdigest()
    rows = []
    for hand, (owner, checkpoint) in configurations.items():
        for resolution, dt in RESOLUTIONS:
            owner.env.sim_dt = dt
            if args.action_name == "reference":
                owner.env.commands.motion.params.sampling_mode = "start"
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override["auto_reset"] = False
            env = create_env(owner, num_envs=1, env_cfg_override=override)
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
                for controller in CONTROLLERS:

                    def action_at():
                        with torch.inference_mode():
                            return (
                                policy(wrapped.get_observations()).numpy()
                                if controller == "ppo"
                                else np.zeros((1, 29), np.float32)
                            )

                    print("START", hand, controller, resolution, flush=True)
                    result = evaluate_case(
                        env,
                        action_at,
                        output,
                        controller,
                        resolution,
                        args.render,
                        action_name=args.action_name,
                    )
                    rows.append(result)
                    (output / "progress.json").write_text(
                        json.dumps(dict(status="incomplete", rows=rows), indent=2, allow_nan=False)
                        + "\n"
                    )
                    print(
                        hand,
                        controller,
                        resolution,
                        result["duration_s"],
                        result["failures"],
                        flush=True,
                    )
            finally:
                env.close()
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == runtime_hash
    (output / "progress.json").unlink()
    report = dict(
        scope=(
            "full_from_rest_measured_command_ppo_approach_not_running_delivery"
            if args.action_name == "reference"
            else "full_from_rest_local_whole_body_residual_over_external_locomotion_not_running_delivery"
        ),
        action_name=args.action_name,
        external_asset_sha256=ASSET_HASHES if args.action_name == "residual" else {},
        rows=rows,
        **qualification(rows),
        columns=COLUMNS,
        input_sha256=hashes,
        runtime_source_sha256={"mjbatch.held_control": runtime_hash},
        versions={
            name: version(name)
            for name in (
                "mujoco",
                "mjbatch",
                "unilab-rl",
                "rsl-rl-lib",
                "torch",
                "numpy",
                "onnxruntime",
            )
        },
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())
        },
        selection="Both final actors and zero residual, both hands and timesteps; both finest PPO videos regardless of outcome",
        guard="No gather, release, independent Menagerie training or full bowling claim; mechanical holder and unchanged approach gates.",
        timing=source["timing"],
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
