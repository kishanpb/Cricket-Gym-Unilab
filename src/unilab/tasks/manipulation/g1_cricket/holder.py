"""Finite-compliance ball holder for release experiments, not a learned grasp."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .prior import build_prior_scene


def build_holder_scene(source: Path, destination: Path, hand: str) -> tuple[str, ...]:
    if hand not in {"left", "right"}:
        raise ValueError("holder hand must be left or right")
    guards = build_prior_scene(source, destination, "none")
    model = mujoco.MjModel.from_xml_path(str(destination))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    wrist_name = f"{hand}_wrist_yaw_link"
    wrist = data.body(wrist_name)
    # Clear the fixed rubber-hand capsule; the palm site lies inside it.
    offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
    position = wrist.xpos + wrist.xmat.reshape(3, 3) @ offset
    rotation = wrist.xquat.copy()
    tree = ET.parse(destination)
    root = tree.getroot()
    option = root.find("option")
    assert option is not None
    option.set("timestep", ".00025")
    ball = root.find(".//body[@name='cricket_ball']")
    key = root.find("keyframe/key")
    assert ball is not None and key is not None
    qpos = model.key_qpos[0].copy()
    ball_address = model.jnt_qposadr[model.body("cricket_ball").jntadr[0]]
    qpos[ball_address : ball_address + 7] = np.concatenate((position, rotation))
    key.set("qpos", " ".join(map(str, qpos)))
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ball.set("pos", " ".join(map(str, wrist.xpos + wrist.xmat.reshape(3, 3) @ offset)))
    ball.set("quat", " ".join(map(str, wrist.xquat)))
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        name="ball_holder",
        body1=wrist_name,
        body2="cricket_ball",
        relpose=" ".join(map(str, [*offset, 1, 0, 0, 0])),
        solref=".004 1",
        torquescale=".04",
    )
    tree.write(destination)
    return guards
