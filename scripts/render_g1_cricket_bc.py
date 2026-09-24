"""Fixed-pool learned-policy diagnostic with solved, interval-binned tactile loads."""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_residual import load_policy, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_trial import ImpactReplay, trial
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from train_g1_cricket_bc import (
    DIRECTORY,
    ROOT,
    RUN,
    checkpoint_info,
    load_contract,
    make_env,
    summarize,
    validate_checkpoint,
    write_json,
)

SELECTION = {
    "controller": "ppo",
    "seed": 4301,
    "hands": ["right", "left"],
    "offsets_m": [-0.12, -0.10, 0.0],
    "engine": "mjbatch",
    "sim_dt": 0.00003125,
    "selection_rule": "first declared seed, both hands, all lanes, complete episodes including failures",
    "simulation_frame_seconds": 0.01,
    "video_fps": 50,
    "terminal_hold_frames": 25,
}


def frame_windows(steps, dt):
    stride = round(SELECTION["simulation_frame_seconds"] / dt)
    if stride < 1 or not np.isclose(stride * dt, 0.01, rtol=0, atol=1e-12):
        raise ValueError("physics step does not divide the declared frame interval")
    if steps != 2 * stride:
        raise ValueError("expected two 10 ms video bins per control tick")
    return ((0, stride), (stride, 2 * stride))


def tactile_bin(impacts, fixture, dt, prior_normal_peak=0.0):
    occupied = loaded = 0
    normal_peak = shear_peak = 0.0
    impulse = np.zeros(3)
    for step in impacts:
        contacts = step["contacts"]
        occupied += bool(contacts)
        loaded += any(c["normal_force_n"] > 0 for c in contacts)
        normal = shear = 0.0
        for contact in contacts:
            force = np.asarray(contact["force_on_ball_world_n"])
            direction = np.asarray(contact["normal_on_ball_world"])
            normal += contact["normal_force_n"]
            shear += float(np.linalg.norm(force - (force @ direction) * direction))
            impulse += force * dt
        normal_peak = max(normal_peak, normal)
        shear_peak = max(shear_peak, shear)
    loads = np.linalg.norm(fixture.reshape(-1, 2, 3), axis=2).max(axis=0)
    return {
        "interval_seconds": len(impacts) * dt,
        "blade_occupied_fraction": occupied / len(impacts),
        "blade_loaded_fraction": loaded / len(impacts),
        "peak_sum_normal_force_n": normal_peak,
        "episode_peak_sum_normal_force_n": max(prior_normal_peak, normal_peak),
        "peak_sum_shear_magnitude_n": shear_peak,
        "impulse_on_ball_world_ns": impulse.tolist(),
        "fixture_peak_force_n": float(loads[0]),
        "fixture_peak_torque_nm": float(loads[1]),
    }


def labels(hand, offset, seconds, vx, tactile, outcome, qualified):
    policy = "right-trained BC + PPO" if hand == "right" else "untrained left policy transfer"
    status = "PASS" if outcome["passed"] else "FAIL"
    paired = "PASS" if qualified else "FAIL"
    exit_velocity = outcome["first_separation_ball_vx_m_s"]
    exit_text = "none" if exit_velocity is None else f"{exit_velocity:+.3f}"
    return [
        f"G1 | {policy} | lane {offset:+.2f} m",
        "Development diagnostic | seed 4301 | 0.5x | native mjbatch / MuJoCo rendering",
        f"t={seconds:.3f} s | current vx={vx:+.2f} | episode first-exit vx={exit_text} m/s (must exceed 1)",
        f"Prior 10 ms: touch={int(tactile['blade_occupied_fraction'] > 0)}"
        f" loaded={int(tactile['blade_loaded_fraction'] > 0)}"
        f" | normal peak={tactile['peak_sum_normal_force_n']:.1f} N"
        f" | shear peak={tactile['peak_sum_shear_magnitude_n']:.1f} N",
        f"10 ms fixture {tactile['fixture_peak_force_n']:.1f} N/{tactile['fixture_peak_torque_nm']:.1f} N m"
        f" | impulse magnitude {np.linalg.norm(tactile['impulse_on_ball_world_ns']):.3f} N s"
        f" | episode normal max {tactile['episode_peak_sum_normal_force_n']:.1f} N",
        f"Full episode: {status} | paired qualification: {paired} | simulated, uncalibrated loads",
        "Shared UniLab task | frozen locomotion prior | rigid wrist bat | not learned bowling",
    ]


def annotate(image, lines):
    canvas = Image.new("RGB", (960, 800), "#17201d")
    canvas.paste(Image.fromarray(image), (0, 80))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for y, line in zip((12, 44, 632, 661, 690, 719, 755), lines, strict=True):
        if draw.textlength(line, font=font) > 928:
            raise ValueError("diagnostic overlay text does not fit")
        draw.text((16, y), line, font=font, fill="white")
    return np.asarray(canvas)


def select_rows(report):
    if summarize(report["rows"]) != report["comparisons"]:
        raise ValueError("evaluation comparison audit differs from retained rows")
    chosen = []
    for hand in SELECTION["hands"]:
        for offset in SELECTION["offsets_m"]:
            identity = dict(hand=hand, controller="ppo", offset_m=offset, seed=SELECTION["seed"])
            rows = [
                r
                for r in report["rows"]
                if r["engine"] == SELECTION["engine"]
                and r["sim_dt"] == SELECTION["sim_dt"]
                and all(r["outcome"][key] == value for key, value in identity.items())
            ]
            groups = [
                r
                for r in report["comparisons"]
                if all(r[key] == value for key, value in identity.items())
            ]
            if len(rows) != 1 or len(groups) != 1:
                raise ValueError("missing or duplicated predeclared video context")
            chosen.append((rows[0], groups[0]))
    return chosen


class FrameReplay(ImpactReplay):
    def __init__(self, env, render_model, renderer, render_data, camera, writer, row, group):
        super().__init__(env)
        self.render_model = render_model
        self.renderer, self.render_data, self.camera, self.writer = (
            renderer,
            render_data,
            camera,
            writer,
        )
        self.row, self.group = row, group
        self.frames = []
        self.normal_peak = 0.0
        self.poses = hashlib.sha256()
        self.first_image = self.last_image = self.still = self.last_frame = None
        self.velocity_index = (
            1 + self.model.nq + self.model.jnt_dofadr[self.model.joint("ball_free").id]
        )

    def step(self, env, action, tick):
        result, impacts = super().step(env, action, tick)
        trajectory, fixture = result[1], result[5]
        self.poses.update(trajectory.tobytes())
        for start, stop in frame_windows(self.steps, env.cfg.sim_dt):
            index = stop - 1
            seconds = (tick * self.steps + stop) * env.cfg.sim_dt
            tactile = tactile_bin(
                impacts[start:stop], fixture[start:stop], env.cfg.sim_dt, self.normal_peak
            )
            self.normal_peak = tactile["episode_peak_sum_normal_force_n"]
            model = self.render_model
            mujoco.mj_setState(
                model, self.render_data, trajectory[index], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            mujoco.mj_forward(model, self.render_data)
            self.renderer.update_scene(self.render_data, self.camera)
            image = self.renderer.render().copy()
            if image.shape != (540, 960, 3) or image.std() < 10:
                raise RuntimeError("blank or invalid simulator frame")
            lines = labels(
                self.row["outcome"]["hand"],
                self.row["outcome"]["offset_m"],
                seconds,
                trajectory[index, self.velocity_index],
                tactile,
                self.row["outcome"],
                self.group["qualified"],
            )
            self.last_frame = annotate(image, lines)
            self.writer.append_data(self.last_frame)
            self.frames.append({"integrated_seconds": seconds, **tactile})
            if self.first_image is None:
                self.first_image = image
            self.last_image = image
            if self.still is None and seconds >= 0.30:
                self.still = self.last_frame.copy()
        return result, impacts


def render():
    contract = load_contract()
    report_path = DIRECTORY / "evaluation.json"
    report = json.loads(report_path.read_text())
    check_hashes(report["input_sha256"])
    chosen = select_rows(report)
    checkpoint = RUN / "ppo.pt"
    validate_checkpoint(torch.load(checkpoint, weights_only=True), checkpoint_info("ppo"))
    output = DIRECTORY / "learned_development_diagnostic.mp4"
    sheet_path = DIRECTORY / "learned_development_contact_sheet.png"
    manifest_path = DIRECTORY / "learned_development_media.json"
    if any(path.exists() for path in (output, sheet_path, manifest_path)):
        raise FileExistsError("learned diagnostic media already exists")
    clips, stills = [], []
    owner = OmegaConf.create(contract["config"])
    with imageio.get_writer(
        output,
        fps=50,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_params=["-crf", "20"],
    ) as writer:
        for row, group in chosen:
            identity = {
                key: row["outcome"][key] for key in ("hand", "controller", "offset_m", "seed")
            }
            env = make_env(
                owner, hand=identity["hand"], dt=SELECTION["sim_dt"], engine=SELECTION["engine"]
            )
            try:
                wrapped, policy = load_policy(owner, env, checkpoint)
                model = mujoco.MjModel.from_xml_path(
                    str(Path(env.scene_directory.name) / "cricket.xml")
                )
                model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
                data = mujoco.MjData(model)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0.2, 0, 0.65]
                camera.distance = 3.2
                camera.azimuth = 125 if identity["hand"] == "right" else -125
                camera.elevation = -15
                with mujoco.Renderer(model, height=540, width=960) as renderer:
                    replay = FrameReplay(env, model, renderer, data, camera, writer, row, group)

                    def action(_tick):
                        with torch.inference_mode():
                            return policy(wrapped.get_observations()).numpy()

                    result = trial(env, replay, identity, action, 100)
                    if result != {k: v for k, v in row.items() if k not in ("engine", "sim_dt")}:
                        raise ValueError("complete render replay differs from evaluated trial")
                    for _ in range(SELECTION["terminal_hold_frames"]):
                        writer.append_data(replay.last_frame)
                    stills.append(replay.still if replay.still is not None else replay.last_frame)
                    clips.append(
                        {
                            **identity,
                            "sim_dt": SELECTION["sim_dt"],
                            "engine": SELECTION["engine"],
                            "qualified": group["qualified"],
                            "outcome": result["outcome"],
                            "complete_replay_matches_evaluation": True,
                            "trajectory_sha256": replay.poses.hexdigest(),
                            "motion_frames": len(replay.frames),
                            "frames": replay.frames,
                            "first_to_last_mean_pixel_change": float(
                                np.abs(replay.last_image.astype(float) - replay.first_image).mean()
                            ),
                        }
                    )
            finally:
                env.close()
    sheet = Image.new("RGB", (1440, 800))
    for i, frame in enumerate(stills):
        sheet.paste(Image.fromarray(frame).resize((480, 400)), ((i % 3) * 480, (i // 3) * 400))
    sheet.save(sheet_path)
    expected_frames = sum(c["motion_frames"] + SELECTION["terminal_hold_frames"] for c in clips)
    decoded = 0
    with imageio.get_reader(output) as reader:
        for frame in reader:
            if frame.shape != (800, 960, 3) or frame[80:620].std() < 10:
                raise RuntimeError("invalid decoded simulation region")
            decoded += 1
    if decoded != expected_frames:
        raise ValueError("video frame count differs from complete clips")
    check_hashes(report["input_sha256"])
    write_json(
        manifest_path,
        {
            "scope": "learned_policy_development_diagnostic_not_showcase_or_promotion",
            "selection": SELECTION,
            "provenance": "shared UniLab task, MuJoCo-trained BC/PPO actor, native mjbatch execution, MuJoCo rendering",
            "playback": "100 simulated frames/s played at 50 fps; 0.5x, then 0.5 s terminal hold per clip",
            "force_timing": "solved loads binned over preceding 10 ms; each solve precedes its integrated pose by one physics substep; render forward is kinematics only",
            "persistent_peak": "episode normal maximum retains earlier bin peaks without implying current contact; resets for each clip",
            "velocity_labels": "current velocity is the rendered endpoint; episode first-exit velocity is the retained full-episode gate metric, not a forecast",
            "tactile_scope": "simulated blade occupancy, loaded state, summed normal/shear peaks and world-frame ball impulse; not hardware taxels",
            "force_frames": "world-frame forces/impulse on ball; shear is force minus its contact-normal projection; fixture norm peaks from solved wrist sensors",
            "evaluation_sha256": sha256(report_path),
            "checkpoint_sha256": sha256(checkpoint),
            "renderer_sha256": sha256(Path(__file__)),
            "video_sha256": sha256(output),
            "sheet_sha256": sha256(sheet_path),
            "decoded_frames": decoded,
            "clips": clips,
            "policy_promoted": False,
        },
    )
    print(
        json.dumps({"video": str(output.relative_to(ROOT)), "decoded_frames": decoded}), flush=True
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    render()
