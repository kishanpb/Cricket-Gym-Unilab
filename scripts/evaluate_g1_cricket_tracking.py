"""Complete dry-swing episodes from the retained whole-body PPO checkpoint."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]


def reference_bat_positions(model, poses):
    data = mujoco.MjData(model)
    positions = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions.append(data.site("bat_center").xpos.copy())
    return np.asarray(positions)


class TrackingReplay:
    """Measure solve-phase loads while independently replaying each held control."""

    def __init__(self, env, *, ball_contact=False):
        self.model = env.get_playback_model()
        self.data = mujoco.MjData(self.model)
        self.hand = env.cfg.handedness
        self.ball_contact = ball_contact
        self.blade_loaded = False
        self.first_separation = None
        self.sensors = env.scene.bind_sensor_data(
            tuple(self.model.sensor(i).name for i in range(self.model.nsensor))
        )

    def measure(self, env, initial):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = env.action_manager.get_term("reference").target[0]
        joints = model.actuator_trnid[:, 0]
        peaks = dict.fromkeys(
            (
                "hard_joint_limit_excess_rad",
                "motor_force_fraction",
                "grip_separation_m",
                "fixture_force_n",
                "fixture_torque_nm",
                "unexpected_contact_force_n",
                "unexpected_penetration_m",
            ),
            0.0,
        )
        contacts = set()
        joint_excess = {}
        wrench = np.empty(6)
        minimum_height = float(data.qpos[2])
        ball_contacts = {}
        for _ in range(env.cfg.sim_substeps):
            mujoco.mj_step(model, data)
            minimum_height = min(minimum_height, float(data.qpos[2]))
            q = data.qpos[model.jnt_qposadr[joints]]
            excess = np.maximum(model.jnt_range[joints, 0] - q, q - model.jnt_range[joints, 1])
            for index in np.flatnonzero(excess > 0):
                name = model.joint(int(joints[index])).name
                joint_excess[name] = max(joint_excess.get(name, 0.0), float(excess[index]))
            values = {
                "hard_joint_limit_excess_rad": max(
                    0.0,
                    float(
                        np.maximum(
                            model.jnt_range[joints, 0] - q, q - model.jnt_range[joints, 1]
                        ).max()
                    ),
                ),
                "motor_force_fraction": float(
                    np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1])
                ),
                "grip_separation_m": float(
                    np.linalg.norm(
                        data.site("bat_lower_grip").xpos - data.site(f"{self.hand}_palm").xpos
                    )
                ),
                "fixture_force_n": float(np.linalg.norm(data.sensor("bat_fixture_force").data)),
                "fixture_torque_nm": float(np.linalg.norm(data.sensor("bat_fixture_torque").data)),
            }
            for key, value in values.items():
                peaks[key] = max(peaks[key], value)
            loaded_pairs = set()
            blade_loaded = False
            for i, contact in enumerate(data.contact):
                if contact.efc_address < 0:
                    continue
                names = {model.geom(int(g)).name for g in contact.geom}
                if self.ball_contact and "ball_geom" in names:
                    pair = "/".join(sorted(names))
                    stats = ball_contacts.setdefault(
                        pair,
                        dict.fromkeys(
                            ("duration_s", "peak_force_n", "normal_impulse_ns", "penetration_m"),
                            0.0,
                        ),
                    )
                    stats["penetration_m"] = max(stats["penetration_m"], float(-contact.dist))
                    mujoco.mj_contactForce(model, data, i, wrench)
                    stats["peak_force_n"] = max(
                        stats["peak_force_n"], float(np.linalg.norm(wrench[:3]))
                    )
                    stats["normal_impulse_ns"] += max(0.0, float(wrench[0])) * model.opt.timestep
                    if wrench[0] > 0:
                        loaded_pairs.add(pair)
                        blade_loaded |= "bat_blade" in names
                    if names == {"ball_geom", "bat_blade"}:
                        continue
                if "pitch" in names and ("ball_geom" in names or any("foot" in n for n in names)):
                    continue
                mujoco.mj_contactForce(model, data, i, wrench)
                if wrench[0] <= 0.1:
                    continue
                contacts.add("/".join(sorted(names)))
                peaks["unexpected_contact_force_n"] = max(
                    peaks["unexpected_contact_force_n"], float(np.linalg.norm(wrench[:3]))
                )
                peaks["unexpected_penetration_m"] = max(
                    peaks["unexpected_penetration_m"], float(-contact.dist)
                )
            for pair in loaded_pairs:
                ball_contacts[pair]["duration_s"] += model.opt.timestep
            if self.ball_contact:
                if self.blade_loaded and not blade_loaded and self.first_separation is None:
                    address = model.joint("ball_free").dofadr[0]
                    self.first_separation = {
                        "time_s": float(data.time),
                        "velocity_m_s": data.qvel[address : address + 3].tolist(),
                    }
                self.blade_loaded = blade_loaded
        expected = np.empty_like(initial, dtype=np.float64)
        mujoco.mj_getState(model, data, expected, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        actual = env.get_physics_state_snapshot()[0]
        np.testing.assert_array_equal(expected.astype(actual.dtype), actual)
        native_sensors = self.sensors.read()[0]
        np.testing.assert_array_equal(data.sensordata.astype(native_sensors.dtype), native_sensors)
        if (
            data.warning.number.any()
            or not np.isfinite(expected).all()
            or not np.isfinite(data.sensordata).all()
        ):
            raise RuntimeError("invalid physics in tracking replay")
        result = {
            "substeps": env.cfg.sim_substeps,
            "peaks": peaks,
            "unexpected_contacts": sorted(contacts),
            "minimum_pelvis_height_m": minimum_height,
            "joint_limit_excess_by_name_rad": joint_excess,
        }
        if self.ball_contact:
            result["ball_contacts"] = ball_contacts
            result["first_force_free_blade_exit"] = self.first_separation
        return result


def visual_model(scene, physics):
    """Restore meshes stripped by the training compiler without changing state layout."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    for kind, count in (("joint", physics.njnt), ("body", physics.nbody)):
        if [getattr(model, kind)(i).name for i in range(count)] != [
            getattr(physics, kind)(i).name for i in range(count)
        ]:
            raise RuntimeError(f"visual {kind} layout differs from the physics model")
    for name in ("jnt_qposadr", "jnt_dofadr"):
        np.testing.assert_array_equal(getattr(model, name), getattr(physics, name))
    for name in ("body_pos", "body_quat", "body_mass"):
        np.testing.assert_allclose(getattr(model, name), getattr(physics, name), atol=1e-14, rtol=0)
    if (model.nq, model.nv, model.na) != (physics.nq, physics.nv, physics.na):
        raise RuntimeError("visual state layout differs from the physics model")
    if model.nmesh == 0:
        raise RuntimeError("G1 visual meshes are missing")
    return model


def evaluate(
    directory,
    render=False,
    *,
    output=None,
    waist_tracking_gain=None,
    root_position_gain=None,
    contact_dt=None,
    soft_toss=False,
):
    if (
        waist_tracking_gain is not None or root_position_gain is not None or contact_dt is not None
    ) and (output is None or output.resolve() == directory.resolve()):
        raise ValueError("controller variants require a separate output directory")
    if soft_toss and contact_dt is None:
        raise ValueError("soft toss requires the fine-resolution contact model")
    saved = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    checkpoint = Path(summary["last_checkpoint"])
    owner = OmegaConf.create(saved["config"])
    evaluation_overrides = {}
    if waist_tracking_gain is not None:
        owner.env.actions.reference.waist_tracking_gain = waist_tracking_gain
        evaluation_overrides["waist_tracking_gain"] = waist_tracking_gain
    if root_position_gain is not None:
        owner.env.actions.reference.root_position_gain = root_position_gain
        evaluation_overrides["root_position_gain"] = root_position_gain
    if contact_dt is not None:
        owner.env.sim_dt = contact_dt
        evaluation_overrides.update(contact_dt=contact_dt, soft_toss=soft_toss)
        if soft_toss:
            owner.env.scene.entities.ball = {
                "root_body_name": "cricket_ball",
                "body_names": ["cricket_ball"],
            }
            owner.env.events = {
                "soft_toss": {
                    "func": "unilab.tasks.manipulation.g1_cricket.bimanual_contact.ResetSoftToss",
                    "mode": "reset",
                }
            }
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = directory
    robot = ROOT / owner.env.scene.model_file
    robot_xml = ET.parse(robot)
    mesh_dir = robot.parent / robot_xml.find("compiler").get("meshdir", ".")
    inputs = [
        directory / "run_config.json",
        checkpoint,
        ROOT / owner.env.commands.motion.params.motion_file,
        Path(__file__),
        robot,
        robot.parent / "scene_flat.xml",
    ]
    inputs += [mesh_dir / mesh.get("file") for mesh in robot_xml.findall("asset/mesh[@file]")]
    if "reference_file" in owner.env.actions.reference:
        inputs.append(ROOT / owner.env.actions.reference.reference_file)
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    hand = owner.env.handedness
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    registry.ensure_registries()
    env = registry.make(
        "G1CricketBimanualContact" if contact_dt is not None else "G1CricketBimanualTracking",
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    rows = []
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
        bat_reference = None
        if "reference_file" in owner.env.actions.reference:
            with np.load(ROOT / owner.env.actions.reference.reference_file) as reference:
                bat_reference = reference_bat_positions(model, reference["qpos"])
                root_reference = reference["qpos"][:, :3].copy()
        display = (
            visual_model(Path(env.scene_directory.name) / "cricket.xml", model) if render else None
        )
        if display is not None:
            display.vis.global_.offwidth, display.vis.global_.offheight = 960, 540
        display_data = mujoco.MjData(display) if display is not None else None
        data = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.1, 0, 0.75]
        camera.distance, camera.azimuth, camera.elevation = 2.7, -65 if hand == "right" else 65, -12
        for controller in ("reference_only", "ppo"):
            env.reset(seed=1)
            replay = TrackingReplay(env, ball_contact=contact_dt is not None)
            trace, frames = [], []
            renderer = mujoco.Renderer(display, height=540, width=960) if render else None
            total_reward = 0.0
            try:
                for tick in range(env.max_episode_length):
                    with torch.inference_mode():
                        action = (
                            policy(wrapped.get_observations())
                            if controller == "ppo"
                            else torch.zeros((1, 29))
                        )
                    if not torch.isfinite(action).all():
                        raise RuntimeError("non-finite tracking policy action")
                    initial = env.get_physics_state_snapshot()[0].copy()
                    _, reward, done, _ = wrapped.step(action)
                    substep_audit = replay.measure(env, initial)
                    total_reward += float(reward[0])
                    physical = env.get_physics_state_snapshot()[0]
                    if not np.isfinite(physical).all() or not torch.isfinite(reward).all():
                        raise RuntimeError("non-finite physical state or reward in evaluation")
                    mujoco.mj_setState(model, data, physical, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    mujoco.mj_forward(model, data)
                    robot = env.scene["robot"]
                    limits = robot.data.soft_joint_pos_limits
                    joints = robot.data.joint_pos[0]
                    if not np.isfinite(joints).all():
                        raise RuntimeError("non-finite robot joints in evaluation")
                    trace.append(
                        {
                            "time_s": float(data.time),
                            "substep_audit": substep_audit,
                            "pelvis_height_m": float(data.qpos[2]),
                            "grip_separation_m": float(
                                np.linalg.norm(
                                    data.site("bat_lower_grip").xpos
                                    - data.site(f"{hand}_palm").xpos
                                )
                            ),
                            "joint_limit_excess_rad": max(
                                0.0,
                                float(
                                    np.maximum(limits[:, 0] - joints, joints - limits[:, 1]).max()
                                ),
                            ),
                            "bat_center_m": data.site("bat_center").xpos.tolist(),
                            "reference_joint_rmse_rad": float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            env.command_manager.get_term("motion").joint_pos[0]
                                            - joints
                                        )
                                        ** 2
                                    )
                                )
                            ),
                        }
                    )
                    if bat_reference is not None:
                        index = int(env.command_manager.get_term("motion").time_steps[0])
                        trace[-1]["reference_bat_center_m"] = bat_reference[index].tolist()
                        trace[-1]["bat_tracking_error_m"] = float(
                            np.linalg.norm(data.site("bat_center").xpos - bat_reference[index])
                        )
                        trace[-1]["root_translation_error_m"] = float(
                            np.linalg.norm(data.qpos[:3] - root_reference[index])
                        )
                    if contact_dt is not None:
                        ball_joint = model.joint("ball_free")
                        qa, va = int(ball_joint.qposadr[0]), int(ball_joint.dofadr[0])
                        trace[-1]["ball_position_m"] = data.qpos[qa : qa + 3].tolist()
                        trace[-1]["ball_velocity_m_s"] = data.qvel[va : va + 3].tolist()
                    if renderer is not None:
                        mujoco.mj_setState(
                            display, display_data, physical, mujoco.mjtState.mjSTATE_FULLPHYSICS
                        )
                        mujoco.mj_forward(display, display_data)
                        renderer.update_scene(display_data, camera)
                        frame = Image.fromarray(renderer.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 64), fill="#17201d")
                        font = ImageFont.load_default(size=18)
                        draw.text(
                            (12, 8),
                            f"G1 {hand} | {controller} | two-hand {'soft toss' if soft_toss else 'dry swing'} | t={data.time:.2f}s",
                            font=font,
                        )
                        draw.text(
                            (12, 34),
                            "Frozen dry-swing actor, no ball observation | mechanical grips | 0.5x"
                            if soft_toss
                            else "Development episode, no ball-hit claim | mechanical grips | 0.5x",
                            font=font,
                        )
                        frames.append(np.asarray(frame))
                    if bool(done[0]):
                        break
                failed = bool(env.state.terminated[0])
                row = {
                    "controller": controller,
                    "hand": hand,
                    "seed": 1,
                    "return": total_reward,
                    "steps": tick + 1,
                    "terminated": failed,
                    "truncated": bool(env.state.truncated[0]),
                    "terminal_terms": [
                        name
                        for name in env.termination_manager.active_terms
                        if env.termination_manager.get_term(name)[0]
                    ],
                    "trace": trace,
                }
                if frames:
                    terminal = Image.fromarray(frames[-1])
                    ImageDraw.Draw(terminal).text(
                        (12, 80),
                        "FAIL: " + ", ".join(row["terminal_terms"])
                        if failed
                        else "Complete diagnostic; not a qualified cricket policy",
                        font=ImageFont.load_default(size=19),
                        fill="red" if failed else "white",
                    )
                    frames.extend([np.asarray(terminal)] * 25)
                    imageio.mimwrite(
                        output / f"{controller}_diagnostic.mp4",
                        frames,
                        fps=25,
                        macro_block_size=1,
                    )
                rows.append(row)
            finally:
                if renderer is not None:
                    renderer.close()
    finally:
        env.close()
    report = {
        "scope": "frozen_dry_swing_actor_soft_toss_not_learned_interception"
        if soft_toss
        else "deterministic_development_dry_swing_not_held_out_cricket",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evaluation_overrides": evaluation_overrides,
        "mujoco_version": mujoco.__version__,
        "rows": rows,
        "input_sha256": hashes,
        "substep_audit": "all intervals independently replayed; exact native endpoint and sensor agreement; simulated loads are uncalibrated",
    }
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.resolve().relative_to(ROOT))]
        for p in inputs
    ):
        raise RuntimeError("evaluation inputs changed during execution")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print([{key: value for key, value in row.items() if key != "trace"} for row in rows])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--waist-tracking-gain", type=float)
    parser.add_argument("--root-position-gain", type=float)
    parser.add_argument("--contact-dt", type=float)
    parser.add_argument("--soft-toss", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.directory,
        args.render,
        output=args.output,
        waist_tracking_gain=args.waist_tracking_gain,
        root_position_gain=args.root_position_gain,
        contact_dt=args.contact_dt,
        soft_toss=args.soft_toss,
    )
