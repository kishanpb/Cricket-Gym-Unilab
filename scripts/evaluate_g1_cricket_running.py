"""Full start-to-terminal running PPO/reference evaluation with native substep replay."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_tracking import visual_model
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[1]


def render_rollout(env, physical, directory, result):
    model = visual_model(Path(env.scene_directory.name) / "cricket.xml", env.get_playback_model())
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 4.2
    camera.azimuth = -90 if env.cfg.handedness == "right" else 90
    camera.elevation = -8
    controller = result["controller"]
    selected = np.linspace(0, len(physical) - 1, 6, dtype=int)
    sheet = Image.new("RGB", (1440, 135))
    with mujoco.Renderer(model, height=540, width=960) as renderer:
        with imageio.get_writer(
            directory / f"{controller}.mp4", fps=25, macro_block_size=1
        ) as writer:
            for index, state in enumerate(physical):
                mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0], 0, 0.8]
                renderer.update_scene(data, camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 70), fill="#17201d")
                draw.text(
                    (12, 8),
                    f"G1 {env.cfg.handedness} | {controller} | physics t={data.time:.2f}s | 0.5x",
                    font=ImageFont.load_default(size=18),
                )
                draw.text(
                    (12, 34),
                    "Development diagnostic | scheduled holder release | NOT qualified bowling",
                    font=ImageFont.load_default(size=16),
                )
                if index:
                    sample = result["trace"][index - 1]
                    draw.rectangle((0, 462, 960, 496), fill="#17201d")
                    draw.text(
                        (12, 470),
                        f"Simulated holder load: {sample['holder_peak_force_n']:.1f} N | "
                        f"hand contact: {sample['hand_touch_fraction']:.0%} | "
                        f"released: {sample['released']}",
                        font=ImageFont.load_default(size=16),
                    )
                if index == len(physical) - 1:
                    flags = [name for name, active in result["termination_flags"].items() if active]
                    draw.rectangle((0, 496, 960, 540), fill="#17201d")
                    draw.text(
                        (12, 506),
                        "Episode end: " + ", ".join(flags),
                        font=ImageFont.load_default(size=16),
                    )
                writer.append_data(np.asarray(frame))
                for column in np.flatnonzero(selected == index):
                    sheet.paste(frame.resize((240, 135)), (int(column) * 240, 0))
    sheet.save(directory / f"{controller}_review.png")


def shift_lane(owner, output, outward_offset):
    """Translate only free-body reference paths; keep world geometry and all velocities."""
    hand = owner.env.handedness
    offset = outward_offset * (1 if hand == "right" else -1)
    with TemporaryDirectory(prefix="g1-running-lane-") as temporary:
        scene = Path(temporary) / "scene.xml"
        G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
            ROOT / owner.env.scene.model_file, scene
        )
        model = mujoco.MjModel.from_xml_path(str(scene))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    with np.load(ROOT / owner.env.actions.reference.reference_file) as data:
        reference = {name: data[name].copy() for name in data.files}
    with np.load(ROOT / owner.env.commands.motion.params.motion_file) as data:
        tracking = {name: data[name].copy() for name in data.files}
    reference["qpos"][:, model.jnt_qposadr[free] + 1] += offset
    moving = np.isin(model.body_rootid, model.jnt_bodyid[free])
    tracking["body_pos_w"][:, moving, 1] += offset
    ref_path, motion_path = output / "reference.npz", output / "tracking.npz"
    np.savez_compressed(ref_path, **reference)
    np.savez_compressed(motion_path, **tracking)
    owner.env.actions.reference.reference_file = str(ref_path.resolve().relative_to(ROOT))
    owner.env.commands.motion.params.motion_file = str(motion_path.resolve().relative_to(ROOT))
    return ref_path, motion_path


def evaluate(directory, render=False, *, output=None, lane_offset=0.0):
    if not np.isfinite(lane_offset) or lane_offset < 0:
        raise ValueError("lane offset must be finite and outward")
    if lane_offset and output is None:
        raise ValueError("lane changes require a separate output directory")
    saved = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "run_summary.json").read_text())
    owner = OmegaConf.create(saved["config"])
    checkpoint = Path(summary["last_checkpoint"])
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = directory
    result_path = output / "evaluation.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    inputs = [
        directory / "run_config.json",
        directory / "run_summary.json",
        checkpoint,
        Path(__file__),
        ROOT / "scripts/evaluate_g1_cricket_tracking.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
        ROOT / owner.env.commands.motion.params.motion_file,
        ROOT / owner.env.actions.reference.reference_file,
        ROOT / "src/unilab/assets/robots/g1/g1.xml",
        ROOT / "src/unilab/assets/robots/g1/scene_flat.xml",
    ]
    if lane_offset:
        inputs.extend(shift_lane(owner, output, lane_offset))
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    registry.ensure_registries()
    env = registry.make(
        owner.training.task_name, num_envs=1, sim_backend="mujoco", env_cfg_override=override
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
        for controller in ("reference_only", "ppo"):
            env.reset(seed=1)
            replay = DeliveryReplay(env, action_name="reference")
            events = DeliveryEvents(env.cfg.handedness)
            physical = [env.get_physics_state_snapshot()[0].copy()]
            trace, total = [], 0.0
            for tick in range(env.max_episode_length):
                with torch.inference_mode():
                    action = (
                        policy(wrapped.get_observations())
                        if controller == "ppo"
                        else torch.zeros((1, 29))
                    )
                if not torch.isfinite(action).all():
                    raise RuntimeError("non-finite running policy action")
                state = replay.step(env, action.numpy(), events)
                term = env.action_manager.get_term("reference")
                total += float(state.reward[0])
                snapshot = env.get_physics_state_snapshot()[0].copy()
                physical.append(snapshot)
                trace.append(
                    {
                        "time_s": float(snapshot[0]),
                        "pelvis_position_m": snapshot[1:4].tolist(),
                        "holder_peak_force_n": float(term.peak_load[0]),
                        "holder_impulse_world_ns": term.impulse_world[0].tolist(),
                        "hand_touch_fraction": float(term.touch_fraction[0]),
                        "released": bool(term.released[0]),
                        "reward": float(state.reward[0]),
                    }
                )
                if state.terminated[0] or state.truncated[0]:
                    break
            result = events.finish(bool(state.truncated[0] and not state.terminated[0]))
            result.update(
                hand=env.cfg.handedness,
                controller=controller,
                seed=1,
                steps=tick + 1,
                episode_return=total,
                exact_endpoint_and_sensor_replay=True,
                trace=trace,
                termination_flags={
                    name: bool(env.termination_manager.get_term(name)[0])
                    for name in env.termination_manager.active_terms
                },
            )
            rows.append(result)
            np.savez_compressed(output / f"{controller}_physical.npz", state=physical)
            if render:
                render_rollout(env, physical, output, result)
    finally:
        env.close()
    for path, digest in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("running evaluation input changed")
    result = {
        "scope": "whole_body_PPO_tracking_with_scheduled_release_not_qualified_bowling",
        "input_sha256": hashes,
        "outward_lane_offset_m": lane_offset,
        "changed_axis": "reference_lane_translation" if lane_offset else None,
        "rows": rows,
        "evaluation_pool": "both controls, deterministic start, seed 1; no heldout/generalization claim",
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print([{k: v for k, v in row.items() if k != "trace"} for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lane-offset", type=float, default=0.0)
    args = parser.parse_args()
    evaluate(args.directory, args.render, output=args.output, lane_offset=args.lane_offset)
