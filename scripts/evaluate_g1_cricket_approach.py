"""Record a complete physically achieved approach teacher, never a bowling claim."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import mjbatch.held_control
import mujoco
import numpy as np
from audit_g1_cricket_running_stance import foot_loads
from evaluate_g1_cricket_tracking import visual_model
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from retarget_g1_cricket_running import render_poses

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.approach import approach_command
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES, SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.tracking import export_reference

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302)
RESOLUTIONS = (("fine", 0.0000625), ("finest", 0.00003125))
COLUMNS = (
    "time_s",
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "root_angular_speed_rad_s",
    "maximum_joint_speed_rad_s",
    "left_load_n",
    "right_load_n",
    "holder_error_m",
    "motor_force_fraction",
    "left_contact_slip_speed_m_s",
    "right_contact_slip_speed_m_s",
    "left_loaded_stance_slip_m",
    "right_loaded_stance_slip_m",
)


def owner_config():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=["task=g1_cricket_approach_v1/mujoco"])


def make_env(hand, dt):
    owner = owner_config()
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, sim_dt=dt)
    return create_env(owner, num_envs=1, env_cfg_override=override)


class ApproachEvents(DeliveryEvents):
    def __init__(self, hand):
        super().__init__(hand)
        self.minimum_foot_y, self.maximum_foot_y = np.inf, -np.inf
        self.minimum_side_clearance = np.inf

    def observe_support(self, time, side, loaded, bounds):
        super().observe_support(time, side, loaded, bounds)
        self.minimum_foot_y = min(self.minimum_foot_y, float(bounds[0, 1]))
        self.maximum_foot_y = max(self.maximum_foot_y, float(bounds[1, 1]))
        clearance = bounds[0, 1] if self.hand == "right" else -bounds[1, 1]
        self.minimum_side_clearance = min(self.minimum_side_clearance, float(clearance))


class LoadedFootSlip:
    def __init__(self, model):
        self.pitch = model.geom("pitch").id
        self.feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.distance = np.zeros(2)

    def measure(self, model, data):
        velocity = np.empty((2, 6))
        for index, body in enumerate(self.feet):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity[index], 0
            )
        speed = np.zeros(2)
        loaded = np.zeros(2, dtype=bool)
        wrench = np.empty(6)
        for index, contact in enumerate(data.contact):
            if self.pitch not in contact.geom:
                continue
            other = int(contact.geom[1] if contact.geom[0] == self.pitch else contact.geom[0])
            body = int(model.geom_bodyid[other])
            if body not in self.feet:
                continue
            mujoco.mj_contactForce(model, data, index, wrench)
            if wrench[0] <= 1:
                continue
            side = self.feet.index(body)
            loaded[side] = True
            point_velocity = velocity[side, 3:] + np.cross(
                velocity[side, :3], contact.pos - data.xpos[body]
            )
            speed[side] = max(
                speed[side], np.linalg.norm(contact.frame.reshape(3, 3)[1:] @ point_velocity)
            )
        self.distance[:] = np.where(loaded, self.distance + speed * model.opt.timestep, 0)
        return np.r_[speed, self.distance]


def summarize(events, steps, states, weight, complete):
    values = dict(zip(COLUMNS, steps.T, strict=True))
    initial = steps[(steps[:, 0] >= 0.5) & (steps[:, 0] < 1)]
    final = steps[steps[:, 0] >= 7.5]
    cruise = steps[(steps[:, 0] >= 3) & (steps[:, 0] < 4)]

    def settled(window):
        return bool(
            len(window)
            and np.linalg.norm(window[:, 4:7], axis=1).max() < 0.05
            and window[:, 7].max() < 0.2
            and window[:, 8].max() < 0.5
            and window[:, 9:11].min() > 1
            and abs(window[:, 9:11].sum(axis=1).mean() / weight - 1) < 0.02
        )

    distance = float(states[-1, 1] - states[0, 1])
    cruise_speed = float(cruise[:, 4].mean()) if len(cruise) else None
    checks = {
        "complete": complete,
        "pelvis_height": events.height > 0.65,
        "pelvis_orientation": events.up > 0.95,
        "joint_limits": events.limit_excess <= 1e-6,
        "motor_limits": events.force_fraction <= 1 + 1e-6,
        "no_unexpected_contact": not events.failures,
        "holder_error": values["holder_error_m"].max() < 0.001,
        "ball_penetration": events.peak_penetration <= 0.006,
        "no_release": events.release_record is None,
        "foot_corridor": events.minimum_foot_y > -1.32 and events.maximum_foot_y < 1.32,
        "declared_wicket_side": events.minimum_side_clearance > 0,
        "lateral_root_drift": np.abs(values["root_y_m"] - states[0, 2]).max() < 0.15,
        "initial_settle": settled(initial),
        "final_settle": settled(final),
        "forward_distance": 2.4 < distance < 3.6,
        "cruise_speed": cruise_speed is not None and 0.7 < cruise_speed < 1.3,
        "repeated_steps": all(len(events.landings[side]) >= 2 for side in ("left", "right")),
        "loaded_foot_slip_speed": steps[:, 13:15].max() < 0.2,
        "loaded_stance_slip_distance": steps[:, 15:17].max() < 0.03,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return dict(
        passed=all(checks.values()),
        checks=checks,
        failures=[key for key, value in checks.items() if not value],
        duration_s=float(states[-1, 0]),
        forward_distance_m=distance,
        cruise_mean_vx_m_s=cruise_speed,
        final_root_position_m=states[-1, 1:4].tolist(),
        minimum_pelvis_height_m=events.height,
        minimum_pelvis_up=events.up,
        maximum_joint_limit_excess_rad=events.limit_excess,
        maximum_motor_force_fraction=events.force_fraction,
        maximum_holder_error_m=float(values["holder_error_m"].max()),
        peak_holder_force_n=events.peak_holder_force,
        holder_impulse_world_ns=events.holder_impulse.tolist(),
        unexpected_contacts=sorted(events.failures),
        foot_landings=events.landings,
        foot_lateral_bounds_m=[events.minimum_foot_y, events.maximum_foot_y],
        minimum_foot_side_clearance_m=events.minimum_side_clearance,
        system_weight_n=weight,
        trace_minimum=steps.min(axis=0).tolist(),
        trace_maximum=steps.max(axis=0).tolist(),
        final_half_second_maximum=final.max(axis=0).tolist() if len(final) else None,
    )


def evaluate_case(output, hand, resolution, dt, seed, render):
    env = make_env(hand, dt)
    try:
        env.reset(seed=seed)
        replay = DeliveryReplay(env)
        model = replay.model
        slip = LoadedFootSlip(model)
        events = ApproachEvents(hand)
        wrist = model.body(f"{hand}_wrist_yaw_link").id
        ball = model.body("cricket_ball").id
        offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
        dofs = model.jnt_dofadr[replay.joints]
        metrics, controls, actions, commands = [], [], [], []
        holder_loads, holder_impulses, touch_fractions = [], [], []
        states = [env.get_physics_state_snapshot()[0].copy()]
        compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=compiled)

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

        for _ in range(env.max_episode_length):
            commands.append(approach_command(env)[0].copy())
            state = replay.step(env, np.zeros((1, 8), np.float32), events, observer=observe)
            term = env.action_manager.get_term("residual")
            if term.released.any() or not env.equality_constraints.get_equality_active().all():
                raise RuntimeError("approach must keep the ball held")
            controls.append(term.processed_action[0].copy())
            actions.append(term.baseline_action[0].copy())
            holder_loads.append(float(term.peak_load[0]))
            holder_impulses.append(term.impulse_world[0].copy())
            touch_fractions.append(float(term.touch_fraction[0]))
            states.append(env.get_physics_state_snapshot()[0].copy())
            if state.terminated[0] or state.truncated[0]:
                break
        states, metrics = np.asarray(states), np.asarray(metrics)
        if not all(
            np.isfinite(values).all() for values in (states, metrics, controls, actions, commands)
        ):
            raise RuntimeError("non-finite approach trace")
        qpos, qvel = states[:, 1 : 1 + model.nq], states[:, 1 + model.nq : 1 + model.nq + model.nv]
        result = summarize(
            events,
            metrics,
            states,
            float(-model.body_mass.sum() * model.opt.gravity[2]),
            bool(state.truncated[0] and not state.terminated[0] and len(controls) == 400),
        )
        name = f"{hand}_{seed}_{resolution}"
        result.update(
            hand=hand,
            seed=seed,
            physics_dt_s=dt,
            resolution=resolution,
            trace=f"{name}.npz",
            tracking=f"{name}_tracking.npz",
            substeps=len(metrics),
            exact_endpoint_and_sensor_replay=True,
            compiled_model_sha256=hashlib.sha256(compiled.tobytes()).hexdigest(),
            env_overrides=dict(handedness=hand, sim_dt=dt),
        )
        np.savez_compressed(
            output / result["trace"],
            states=states,
            times=states[:, 0],
            qpos=qpos,
            qvel=qvel,
            controls=controls,
            prior_actions=actions,
            commands=commands,
            holder_peak_force_n=holder_loads,
            holder_impulse_world_ns=holder_impulses,
            geometric_touch_fraction=touch_fractions,
            steps=metrics,
            equality_active=np.ones((len(states), model.neq), dtype=bool),
            joint_names=np.array(SDK_JOINTS),
            body_names=np.array([model.body(i).name for i in range(model.nbody)]),
        )
        export_reference(model, qpos, 50, output / result["tracking"], qvel=qvel)
        if render and seed == SEEDS[0] and resolution == "finest":
            visual = visual_model(Path(env.scene_directory.name) / "cricket.xml", model)
            visual.vis.global_.offwidth, visual.vis.global_.offheight = 960, 540
            render_poses(
                visual,
                qpos,
                hand,
                output / f"{hand}_approach.mp4",
                "MEASURED APPROACH, NOT BOWLING",
                subtitle="External locomotion prior | actual native states | held ball | 0.5x",
            )
        return result
    finally:
        env.close()


def resolution_comparison(a, b):
    checks = {
        "same_gate_outcomes": a["checks"] == b["checks"],
        "final_position": np.linalg.norm(
            np.array(a["final_root_position_m"]) - b["final_root_position_m"]
        )
        < 0.05,
        "holder_force": abs(a["peak_holder_force_n"] - b["peak_holder_force_n"])
        <= max(1, 0.05 * b["peak_holder_force_n"]),
        "holder_error": abs(a["maximum_holder_error_m"] - b["maximum_holder_error_m"]) < 0.0001,
        "landing_counts": all(
            len(a["foot_landings"][s]) == len(b["foot_landings"][s]) for s in ("left", "right")
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return dict(hand=a["hand"], seed=a["seed"], checks=checks, passed=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__),
        ROOT / "scripts/g1_cricket_delivery_trial.py",
        ROOT / "scripts/audit_g1_cricket_running_stance.py",
        ROOT / "scripts/evaluate_g1_cricket_tracking.py",
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROOT / "src/unilab/assets/robots/g1/g1.xml",
        ROOT / "src/unilab/assets/robots/g1/scene_flat.xml",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs.append(ROOT / "src/unilab/base/mujoco_substeps.py")
    robot = ROOT / "src/unilab/assets/robots/g1/g1.xml"
    xml = ET.parse(robot)
    mesh_dir = robot.parent / xml.find("compiler").get("meshdir", ".")
    inputs += [mesh_dir / mesh.get("file") for mesh in xml.findall("asset/mesh[@file]")]
    for task in (
        "g1_cricket_prior_v1",
        "g1_cricket_bowling_v1",
        "g1_cricket_delivery_v1",
        "g1_cricket_delivery_pitch_v2",
        "g1_cricket_approach_v1",
    ):
        inputs.append(ROOT / f"src/unilab/conf/ppo/task/{task}/mujoco.yaml")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    recorder = Path(mjbatch.held_control.__file__)
    recorder_hash = hashlib.sha256(recorder.read_bytes()).hexdigest()
    rows = []
    for hand in ("right", "left"):
        for seed in SEEDS:
            for resolution, dt in RESOLUTIONS:
                print("START", hand, seed, resolution, flush=True)
                row = evaluate_case(args.output, hand, resolution, dt, seed, args.render)
                rows.append(row)
                print(hand, seed, resolution, row["duration_s"], row["failures"], flush=True)
    pairs = [resolution_comparison(*rows[i : i + 2]) for i in range(0, len(rows), 2)]
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(recorder.read_bytes()).hexdigest() == recorder_hash
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())
    }
    report = dict(
        scope="measured_external_prior_approach_teacher_not_locally_learned_bowling",
        rows=rows,
        resolution_comparisons=pairs,
        approach_qualified=all(r["passed"] for r in rows + pairs),
        columns=COLUMNS,
        timing="State/qvel/time columns are integrated ends; contacts, loads, geometry and contact-point slip use the solved-start frame one physics step earlier.",
        base_owner_config=OmegaConf.to_container(owner_config(), resolve=True),
        input_sha256=hashes,
        artifact_sha256=artifacts,
        external_asset_sha256=ASSET_HASHES,
        runtime_source_sha256={"mjbatch.held_control": recorder_hash},
        versions={
            name: version(name)
            for name in ("mujoco", "mjbatch", "unilab-rl", "onnxruntime", "numpy")
        },
        selection="Both hands, seeds 5301/5302, both resolutions; videos are seed 5301 finest regardless of outcome",
        guard="Gather, legal delivery, recovery and local whole-body learning remain required. External weights are not redistributed.",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
