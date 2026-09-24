"""Render every fixed first-seed PPO lane, including failures, as a diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_residual import ROOT, SEEDS, CricketReplay, load_policy, sha256
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.residual import TOSS_OFFSETS


def annotated(image, hand, offset, seconds, vx, contacts, fixture, row):
    canvas = Image.new("RGB", (960, 704), "#17201d")
    canvas.paste(Image.fromarray(image), (0, 80))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    label = "right-trained PPO" if hand == "right" else "untrained left transfer"
    draw.text((16, 10), f"G1 | {label} | lane {offset:+.2f} m", font=font, fill="white")
    draw.text(
        (16, 43),
        f"Development diagnostic | seed {SEEDS[0]} | 0.5x playback",
        font=font,
        fill="#d0d8d4",
    )
    touch = any(c["geom"] == "bat_blade" for c in contacts)
    force = max((c["force_norm_n"] for c in contacts if c["geom"] == "bat_blade"), default=0.0)
    draw.text(
        (16, 629),
        f"t={seconds:.3f} s | ball vx={vx:+.2f} m/s | blade touch={int(touch)} | contact={force:.1f} N",
        font=font,
        fill="white",
    )
    outcome = "PASS" if row["passed"] else "FAIL"
    draw.text(
        (16, 663),
        f"Full-trial gate: {outcome} | fixture load={np.linalg.norm(fixture[:3]):.1f} N | simulated, uncalibrated",
        font=font,
        fill="#95dfb2" if row["passed"] else "#ffc2b0",
    )
    return np.asarray(canvas)


def render(run_dir):
    run_dir = run_dir.resolve()
    directory = run_dir.parent
    report_path = directory / "evaluation.json"
    report = json.loads(report_path.read_text())
    for name, expected in report["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"evaluation source changed: {name}")
    checkpoint = ROOT / report["checkpoint"]["path"]
    if (
        sha256(checkpoint) != report["checkpoint"]["sha256"]
        or sha256(run_dir / "run_config.json") != report["run_config_sha256"]
    ):
        raise ValueError("checkpoint/config differs from evaluated run")
    owner = OmegaConf.create(json.loads((run_dir / "run_config.json").read_text())["config"])
    output = directory / "development_diagnostic.mp4"
    clips, stills = [], []
    with imageio.get_writer(
        output,
        fps=50,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_params=["-crf", "20"],
    ) as writer:
        for hand in ("right", "left"):
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override.update(handedness=hand, auto_reset=False)
            env = create_env(owner, num_envs=1, env_cfg_override=override)
            try:
                wrapped, policy = load_policy(owner, env, checkpoint)
                replay = CricketReplay(env)
                model = mujoco.MjModel.from_xml_path(
                    str(Path(env.scene_directory.name) / "cricket.xml")
                )
                data = mujoco.MjData(model)
                model.vis.global_.offwidth = 960
                model.vis.global_.offheight = 540
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.2, 0, 0.65]
                camera.distance, camera.azimuth, camera.elevation = (
                    3.2,
                    (125 if hand == "right" else -125),
                    -15,
                )
                ball_vx = 1 + model.nq + model.jnt_dofadr[model.joint("ball_free").id]
                with mujoco.Renderer(model, height=540, width=960) as renderer:
                    for offset in TOSS_OFFSETS:
                        row = next(
                            r
                            for r in report["rows"]
                            if r["controller"] == "ppo"
                            and r["hand"] == hand
                            and r["offset_m"] == offset
                            and r["seed"] == SEEDS[0]
                        )
                        env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [offset]
                        env.reset(seed=SEEDS[0])
                        episode_return = 0.0
                        count, poses = 0, hashlib.sha256()
                        first = still = None
                        for tick in range(env.max_episode_length):
                            with torch.inference_mode():
                                action = policy(wrapped.get_observations()).numpy()
                            state, trajectory, _, _, _, fixture, contacts = replay.step(env, action)
                            episode_return += float(state.reward[0])
                            poses.update(trajectory.tobytes())
                            for step in (4, 9):
                                seconds = (tick * 10 + step + 1) * env.cfg.sim_dt
                                mujoco.mj_setState(
                                    model,
                                    data,
                                    trajectory[step],
                                    mujoco.mjtState.mjSTATE_FULLPHYSICS,
                                )
                                mujoco.mj_forward(model, data)
                                renderer.update_scene(data, camera)
                                image = renderer.render().copy()
                                if image.std() < 10:
                                    raise RuntimeError("blank diagnostic render")
                                frame = annotated(
                                    image,
                                    hand,
                                    offset,
                                    seconds,
                                    trajectory[step, ball_vx],
                                    contacts[step],
                                    fixture[step],
                                    row,
                                )
                                if first is None:
                                    first = image
                                if still is None and seconds >= 0.30:
                                    still = frame.copy()
                                writer.append_data(frame)
                                count += 1
                            if state.terminated[0] or state.truncated[0]:
                                break
                        if (
                            episode_return != row["return"]
                            or (tick + 1) * env.step_dt != row["seconds"]
                        ):
                            raise ValueError("render replay differs from the evaluated trial")
                        for _ in range(25):
                            writer.append_data(frame)
                        stills.append(frame.copy() if still is None else still)
                        clips.append(
                            {
                                "hand": hand,
                                "offset_m": offset,
                                "seed": SEEDS[0],
                                "passed": row["passed"],
                                "failures": row["failures"],
                                "simulation_seconds": row["seconds"],
                                "motion_frames": count,
                                "hold_frames": 25,
                                "trajectory_sha256": poses.hexdigest(),
                                "first_to_last_mean_pixel_change": float(
                                    np.abs(image.astype(float) - first).mean()
                                ),
                            }
                        )
            finally:
                env.close()
    sheet = Image.new("RGB", (1440, 704))
    for index, frame in enumerate(stills):
        sheet.paste(
            Image.fromarray(frame).resize((480, 352)), ((index % 3) * 480, (index // 3) * 352)
        )
    sheet_path = directory / "development_contact_sheet.png"
    sheet.save(sheet_path)
    expected_frames = sum(c["motion_frames"] + c["hold_frames"] for c in clips)
    decoded = 0
    with imageio.get_reader(output) as reader:
        for frame in reader:
            if frame.shape != (704, 960, 3) or frame.std() < 10:
                raise RuntimeError("invalid decoded diagnostic frame")
            decoded += 1
    if decoded != expected_frames:
        raise ValueError("video dropped diagnostic frames")
    manifest = {
        "scope": "development_diagnostic_not_showcase_or_policy_promotion",
        "selection": "first declared seed, both hands, all three lanes; full episodes including failures; contact sheet at .30 s or terminal if earlier",
        "playback": "100 simulated frames/s played at 50 fps, 0.5x speed, then .5 s terminal hold per clip",
        "force_timing": "loads from the replayed 2 ms contact solve, preceding the rendered integrated pose by one substep; rendering forward is used for kinematics only",
        "evaluation_sha256": sha256(report_path),
        "checkpoint_sha256": sha256(checkpoint),
        "renderer_sha256": sha256(Path(__file__)),
        "video_sha256": sha256(output),
        "sheet_sha256": sha256(sheet_path),
        "decoded_frames": decoded,
        "clips": clips,
    }
    (directory / "development_media.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"video": str(output.relative_to(ROOT)), "decoded_frames": decoded}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    render(parser.parse_args().run_dir)
