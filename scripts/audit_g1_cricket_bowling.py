"""Complete, untrained carry/drop checks before a bowling learning experiment."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[1]


def run(engine, hand, dt):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=[f"task=g1_cricket_bowling_v1/{engine}"])
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, sim_dt=dt, auto_reset=False)
    env = create_env(owner, num_envs=4, env_cfg_override=override)
    try:
        env.reset(seed=5301)
        term = env.action_manager.get_term("residual")
        alive = np.ones(4, dtype=bool)
        rows = [
            dict(
                engine=engine,
                hand=hand,
                physics_dt_s=dt,
                seed=5301,
                row=i,
                controller="zero_arm_hold" if i < 2 else "zero_arm_drop_at_1s",
                steps=0,
                peak_holder_force_n=0.0,
                holder_impulse_world_ns=[0.0, 0.0, 0.0],
                touch_intervals=0,
                minimum_pelvis_height_m=float("inf"),
                minimum_up_component=float("inf"),
            )
            for i in range(4)
        ]
        for tick in range(200):
            action = np.zeros((4, 8), np.float32)
            if tick == 50:
                action[2:, 7] = 1
            state = env.step(action)
            physics = env.get_physics_state_snapshot()
            if not all(
                np.isfinite(value).all() for value in (physics, state.reward, *state.obs.values())
            ):
                raise RuntimeError("nonfinite bowling smoke state")
            robot = env.scene["robot"].data
            for i in np.flatnonzero(alive):
                row = rows[i]
                row["steps"] += 1
                row["peak_holder_force_n"] = max(
                    row["peak_holder_force_n"], float(term.peak_load[i])
                )
                row["holder_impulse_world_ns"] = (
                    np.array(row["holder_impulse_world_ns"]) + term.impulse_world[i]
                ).tolist()
                row["touch_intervals"] += int(term.touch_fraction[i] > 0)
                row["minimum_pelvis_height_m"] = min(
                    row["minimum_pelvis_height_m"], float(robot.root_link_pos_w[i, 2])
                )
                row["minimum_up_component"] = min(
                    row["minimum_up_component"], float(-robot.projected_gravity_b[i, 2])
                )
                if term.just_released[i]:
                    row["release_position_m"] = term.release_position[i].tolist()
                    row["release_velocity_m_s"] = term.release_velocity[i].tolist()
                if state.terminated[i] or state.truncated[i] or tick == 199:
                    row.update(
                        terminated=bool(state.terminated[i]),
                        truncated=bool(state.truncated[i]),
                        final_root_position_m=robot.root_link_pos_w[i].tolist(),
                        released=bool(term.released[i]),
                    )
                    alive[i] = False
            if not alive.any():
                break
            done = np.flatnonzero(state.terminated | state.truncated)
            if len(done):
                env.reset(env_ids=done)
        return rows
    finally:
        env.close()


def summarize(rows):
    if len(rows) != 32:
        raise ValueError("expected every hand/executor/timestep/four-row context")
    groups = {}
    for row in rows:
        key = (row["hand"], row["physics_dt_s"], row["row"])
        if row["engine"] in groups.setdefault(key, {}):
            raise ValueError("duplicate executor row")
        groups[key][row["engine"]] = {k: v for k, v in row.items() if k != "engine"}
    exact_pairs = sum(pair.get("mujoco") == pair.get("mjbatch") for pair in groups.values())
    if len(groups) != 16 or exact_pairs != 16:
        raise ValueError("executor outcome/telemetry mismatch")
    return dict(
        rows=len(rows),
        exact_executor_pairs=exact_pairs,
        completed=sum(r["steps"] == 200 and r["truncated"] and not r["terminated"] for r in rows),
        released=sum(r["released"] for r in rows),
        geometric_touch_intervals=sum(r["touch_intervals"] for r in rows),
        maximum_holder_force_n=max(r["peak_holder_force_n"] for r in rows),
        minimum_pelvis_height_m=min(r["minimum_pelvis_height_m"] for r in rows),
        minimum_up_component=min(r["minimum_up_component"] for r in rows),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for hand in ("right", "left"):
        for dt in (0.00025, 0.000125):
            for engine in ("mujoco", "mjbatch"):
                rows.extend(run(engine, hand, dt))
    inputs = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py")),
        *sorted((ROOT / "src/unilab/conf/ppo/task/g1_cricket_bowling_v1").glob("*.yaml")),
    ]
    result = dict(
        scope="untrained_carry_and_timed_drop_not_learned_bowling_or_legal_delivery",
        force_scope="simulated_fixture_force_not_finger_tactile_or_hardware_calibration",
        torque_scope="omitted_unvalidated_weld_site_torque",
        versions={name: version(name) for name in ("mujoco", "mjbatch", "numpy", "onnxruntime")},
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        summary=summarize(rows),
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
