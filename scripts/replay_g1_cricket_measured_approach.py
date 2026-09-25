"""Compare measured-target zero residual against every retained approach trial."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / "g1_cricket_results/approach_teacher_v1"


def replay_case(row, output):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                "task=g1_cricket_measured_approach/mjbatch",
                f"env.handedness={row['hand']}",
                f"env.sim_dt={row['physics_dt_s']}",
                "env.commands.motion.params.sampling_mode=start",
                f"env.commands.motion.reference_file={TEACHER.relative_to(ROOT) / row['trace']}",
                f"env.commands.motion.params.motion_file={TEACHER.relative_to(ROOT) / row['tracking']}",
            ],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    with np.load(TEACHER / row["trace"]) as saved:
        expected = {k: saved[k].copy() for k in ("states", "controls", "holder_peak_force_n")}
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    try:
        env.reset(seed=row["seed"])
        action = env.action_manager.get_term("reference")
        states = [env.get_physics_state_snapshot()[0].copy()]
        controls, loads, rewards = [], [], []
        for _ in range(len(expected["controls"])):
            state = env.step(np.zeros((1, 29), np.float32))
            states.append(env.get_physics_state_snapshot()[0].copy())
            controls.append(action.processed_action[0].copy())
            loads.append(float(action.peak_load[0]))
            rewards.append(float(state.reward[0]))
            assert env.equality_constraints.get_equality_active().all()
            if state.terminated[0] or state.truncated[0]:
                break
        states, controls, loads = np.asarray(states), np.asarray(controls), np.asarray(loads)
        for values in (states, controls, loads, rewards):
            assert np.isfinite(values).all()
        length = len(states)
        error = np.abs(states - expected["states"][:length])
        nq = env.get_playback_model().nq
        complete = len(controls) == 400 and bool(state.truncated[0] and not state.terminated[0])
        name = row["trace"]
        np.savez_compressed(
            output / name, states=states, controls=controls, holder_peak_force_n=loads
        )
        return dict(
            hand=row["hand"],
            seed=row["seed"],
            resolution=row["resolution"],
            physics_dt_s=row["physics_dt_s"],
            trace=name,
            intervals=len(controls),
            completed=complete,
            exact_states=bool(np.array_equal(states, expected["states"])),
            exact_controls=bool(np.array_equal(controls, expected["controls"])),
            exact_holder_loads=bool(np.array_equal(loads, expected["holder_peak_force_n"])),
            maximum_state_error=error.max(axis=0).tolist(),
            maximum_qpos_error=float(error[:, 1 : 1 + nq].max()),
            maximum_qvel_error=float(error[:, 1 + nq :].max()),
            episode_return=float(sum(rewards)),
            source_failures=row["failures"],
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    teacher = json.loads((TEACHER / "summary.json").read_text())
    inputs = [Path(__file__), ROOT / "src/unilab/tasks/__init__.py", TEACHER / "summary.json"]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += sorted(
        (ROOT / "src/unilab/conf/ppo/task/g1_cricket_measured_approach").glob("*.yaml")
    )
    inputs += sorted(TEACHER.glob("*.npz"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows = []
    for row in teacher["rows"]:
        result = replay_case(row, args.output)
        rows.append(result)
        print(
            result["hand"],
            result["seed"],
            result["resolution"],
            result["completed"],
            result["maximum_qpos_error"],
            result["maximum_qvel_error"],
            flush=True,
        )
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    report = dict(
        scope="zero_residual_measured_command_replay_not_local_learning_or_gait_qualification",
        rows=rows,
        exact_replay=all(
            r["completed"] and r["exact_states"] and r["exact_controls"] and r["exact_holder_loads"]
            for r in rows
        ),
        source_approach_qualified=teacher["approach_qualified"],
        input_sha256=hashes,
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.output.iterdir())
        },
        guard="Source foot-slip and drift failures remain; no full bowling, local PPO or promotion claim.",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
