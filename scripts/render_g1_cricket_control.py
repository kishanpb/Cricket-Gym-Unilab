"""Render complete recorded controller trajectories without replaying them as physics."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unilab.tasks.manipulation.g1_cricket.bimanual import build_bimanual_scene

ROOT = Path(__file__).resolve().parents[1]


def render(directory):
    report = json.loads((directory / "evaluation.json").read_text())
    for path, expected in report["input_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"controller rendering source drift: {path}")
    sheet = Image.new("RGB", (1920, 540))
    rows = [row for row in report["rows"] if row["clip_motor_target"]]
    if [row["hand"] for row in rows] != ["right", "left"]:
        raise ValueError("expected the complete clipped right/left pair")
    for row_index, row in enumerate(rows):
        hand = row["hand"]
        with np.load(directory / row["trajectory_file"]) as recorded:
            poses, velocities = recorded["qpos"], recorded["qvel"]
        if len(poses) != len(row["trace"]) + 1:
            raise ValueError("trajectory does not cover the complete control trace")
        with TemporaryDirectory(prefix="g1-control-render-") as temporary:
            scene = Path(temporary) / "scene.xml"
            build_bimanual_scene(ROOT / "src/unilab/assets/robots/g1/g1.xml", scene, hand)
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
            data = mujoco.MjData(model)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0.1, 0, 0.75]
            camera.distance = 2.7
            camera.azimuth, camera.elevation = (-65 if hand == "right" else 65), -12
            frames = []
            with mujoco.Renderer(model, height=540, width=960) as renderer:
                for i, (pose, velocity) in enumerate(zip(poses, velocities, strict=True)):
                    data.qpos[:], data.qvel[:] = pose, velocity
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 64), fill="#17201d")
                    font = ImageFont.load_default(size=18)
                    draw.text(
                        (12, 8),
                        f"G1 {hand} | two-hand swing | balance feedback | t={i * 0.02:.2f}s",
                        font=font,
                    )
                    draw.text(
                        (12, 34),
                        "Controller diagnostic, NOT RL | no ball hit | mechanical grips | 0.5x",
                        font=font,
                    )
                    frames.append(np.asarray(frame))
            for column, index in enumerate(
                (0, min(57, len(frames) - 1), min(90, len(frames) - 1), len(frames) - 1)
            ):
                sheet.paste(
                    Image.fromarray(frames[index]).resize((480, 270)),
                    (column * 480, row_index * 270),
                )
            failed = [key for key, passed in row["feasibility_checks"].items() if not passed]
            final = Image.fromarray(frames[-1])
            ImageDraw.Draw(final).text(
                (12, 80),
                "Failed diagnostic gates: " + ", ".join(failed)
                if failed
                else "Dry-swing feasibility only; not learned cricket",
                font=ImageFont.load_default(size=19),
                fill="red" if failed else "white",
            )
            frames.extend([np.asarray(final)] * 25)
            imageio.mimwrite(
                directory / f"{hand}_control_diagnostic.mp4", frames, fps=25, macro_block_size=1
            )
    sheet.save(directory / "control_contact_sheet.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
