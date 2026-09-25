"""Final PPO versus zero residual, with unchanged first-step gates and exact replay."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_startup import STEP_COLUMNS, replay_startup
from evaluate_g1_cricket_tracking import visual_model
from omegaconf import OmegaConf
from retarget_g1_cricket_running import render_poses
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]


class FirstStepReplay:
    """Drive the live env through motors; reset only the independent replay state."""

    def __init__(self, env, action_at):
        self.env, self.action_at = env, action_at
        model = env.get_playback_model()
        self.sensors = env.scene.bind_sensor_data(
            tuple(model.sensor(i).name for i in range(model.nsensor))
        )
        self.done = False
        self.total_reward = 0.0
        self.actions, self.controls, self.states = [], [], []

    def initialize(self, model, data):
        initial = self.env.get_physics_state_snapshot()[0].copy()
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.states.append(initial)

    def begin(self, model, data):
        initial = self.env.get_physics_state_snapshot()[0].copy()
        action = self.action_at()
        if not np.isfinite(action).all():
            raise RuntimeError("non-finite first-step action")
        state = self.env.step(action)
        term = self.env.action_manager.get_term("reference")
        if term.released.any() or not self.env.equality_constraints.get_equality_active().all():
            raise RuntimeError("first-step curriculum unexpectedly released the ball")
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.done = bool(state.terminated[0] or state.truncated[0])
        self.total_reward += float(state.reward[0])
        self.actions.append(action[0].copy())
        self.controls.append(term.processed_action[0].copy())
        return self.controls[-1]

    def finish(self, model, data):
        actual = self.env.get_physics_state_snapshot()[0].copy()
        expected = np.empty(actual.shape, dtype=np.float64)
        mujoco.mj_getState(model, data, expected, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(expected.astype(actual.dtype), actual)
        sensors = self.sensors.read()[0]
        np.testing.assert_array_equal(data.sensordata.astype(sensors.dtype), sensors)
        self.states.append(actual)


def evaluation_override(owner, dt):
    evaluation = OmegaConf.create(OmegaConf.to_container(owner, resolve=True))
    evaluation.env.sim_dt = dt
    evaluation.env.commands.motion.params.sampling_mode = "start"
    override = BackendAdapter(evaluation, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    return override


def evaluate(directory, render=False):
    owner = OmegaConf.create(json.loads((directory / "run_config.json").read_text())["config"])
    summary = json.loads((directory / "run_summary.json").read_text())
    checkpoint = Path(summary["last_checkpoint"])
    output = directory / "evaluation"
    output.mkdir(exist_ok=False)
    inputs = [directory / "run_config.json", directory / "run_summary.json", checkpoint]
    inputs += [
        Path(__file__),
        ROOT / "scripts/evaluate_g1_cricket_startup.py",
        ROOT / "scripts/audit_g1_cricket_running_stance.py",
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
        ROOT / "scripts/evaluate_g1_cricket_tracking.py",
        ROOT / owner.env.actions.reference.reference_file,
        ROOT / owner.env.commands.motion.params.motion_file,
        ROOT / "src/unilab/assets/robots/g1/g1.xml",
        ROOT / "src/unilab/assets/robots/g1/scene_flat.xml",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    with np.load(ROOT / owner.env.actions.reference.reference_file) as saved:
        reference = {name: saved[name] for name in saved.files}
    registry.ensure_registries()
    rows = []
    for dt in (0.0000625, 0.00003125):
        override = evaluation_override(owner, dt)
        env = registry.make(
            owner.training.task_name, num_envs=1, sim_backend="mujoco", env_cfg_override=override
        )
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
            model = env.get_playback_model()
            for controller in ("reference_only", "ppo"):
                env.reset(seed=1)
                np.testing.assert_array_equal(env.command_manager.get_term("motion").time_steps, 0)
                np.testing.assert_array_equal(env.episode_length_buf, 0)

                def action_at():
                    with torch.inference_mode():
                        return (
                            policy(wrapped.get_observations()).numpy()
                            if controller == "ppo"
                            else np.zeros((1, 29), dtype=np.float32)
                        )

                driver = FirstStepReplay(env, action_at)
                result, trace = replay_startup(
                    model, reference, env.cfg.handedness, first_step=True, env_driver=driver
                )
                name = f"{controller}_{dt * 1e6:g}us"
                result.update(
                    controller=controller,
                    seed=1,
                    episode_return=driver.total_reward,
                    exact_endpoint_and_sensor_replay=True,
                    trace=f"{name}.npz",
                    termination_flags={
                        key: bool(env.termination_manager.get_term(key)[0])
                        for key in env.termination_manager.active_terms
                    },
                )
                rows.append(result)
                np.savez_compressed(
                    output / result["trace"],
                    **trace,
                    state=driver.states,
                    actions=driver.actions,
                    controls=driver.controls,
                )
                print(
                    env.cfg.handedness,
                    dt,
                    controller,
                    result["duration_s"],
                    result["failures"],
                    flush=True,
                )
                if render and dt == 0.00003125:
                    visual = visual_model(Path(env.scene_directory.name) / "cricket.xml", model)
                    visual.vis.global_.offwidth, visual.vis.global_.offheight = 960, 540
                    render_poses(
                        visual,
                        trace["qpos"],
                        env.cfg.handedness,
                        output / f"{controller}.mp4",
                        f"FIRST-STEP {controller}, NOT RUNNING OR DELIVERY",
                        subtitle=f"Complete {controller} rollout | mechanical holder | 0.5x | development only",
                    )
        finally:
            env.close()
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(
        scope="held_ball_first_step_curriculum_not_running_bowling",
        input_sha256=hashes,
        columns=STEP_COLUMNS,
        rows=rows,
        evaluation_pool="final checkpoint and zero residual; seed 1; both timesteps; no selection",
        training_sampling_mode=owner.env.commands.motion.params.sampling_mode,
        evaluation_sampling_mode="start",
        guard="No learned release, full run-up, delivery, generalization or independent Menagerie claim.",
    )
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.directory, args.render)
