"""Versioned cricket geometry with 29 body servos, 14 finger motors and a free bat."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from .articulated_hands import attach_hand, build_grasp_bench, hand_joints, load_hand_assets
from .prior import SDK_JOINTS, build_prior_scene
from .scene import CONTACT_FIELDS


def build_articulated_scene(source, assets, destination, *, contact_time_constant=0.005):
    """Construct a new model; legacy 29-joint tasks and references stay unchanged."""
    destination = Path(destination)
    build_prior_scene(Path(source), destination, "none")
    original = mujoco.MjModel.from_xml_path(str(destination))
    tree = ET.parse(destination)
    root = tree.getroot()
    root.remove(root.find("keyframe"))
    official = load_hand_assets(assets)
    for side in ("left", "right"):
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        attach_hand(root, wrist, side, official, assets)
    with TemporaryDirectory(prefix="g1-hand-scene-") as temporary:
        bench_path = Path(temporary) / "bench.xml"
        build_grasp_bench(assets, bench_path, contact_time_constant=contact_time_constant)
        bench = ET.parse(bench_path).getroot()
        bat = copy.deepcopy(bench.find("worldbody/body[@name='cricket_bat']"))
        bat.set("pos", "0.4 0.4 1.2")
        root.find("worldbody").append(bat)
        root.append(copy.deepcopy(bench.find("contact")))
    root.find("option").set("timestep", "0.00025")
    sensors = root.find("sensor")
    ET.SubElement(
        sensors,
        "contact",
        name="ball_bat",
        geom1="ball_geom",
        geom2="bat_blade",
        data=CONTACT_FIELDS,
        num="8",
        reduce="none",
    )
    for geom in root.findall("worldbody//geom"):
        name = geom.get("name", "")
        if "_hand_" in name and geom.get("contype") != "0":
            ET.SubElement(
                sensors,
                "contact",
                name=f"grasp_{name}",
                geom1=name,
                geom2="bat_handle",
                data=CONTACT_FIELDS,
                num="8",
                reduce="none",
            )
    for side in ("left", "right"):
        for joint in hand_joints(side):
            ET.SubElement(sensors, "jointpos", name=f"position_{joint}", joint=joint)
            ET.SubElement(sensors, "jointvel", name=f"velocity_{joint}", joint=joint)
            ET.SubElement(sensors, "actuatorfrc", name=f"force_{joint}", actuator=joint)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    initial = model.qpos0.copy()
    for joint in range(original.njnt):
        name = mujoco.mj_id2name(original, mujoco.mjtObj.mjOBJ_JOINT, joint)
        old_address, new_address = original.jnt_qposadr[joint], model.joint(name).qposadr[0]
        width = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(
            original.jnt_type[joint], 1
        )
        initial[new_address : new_address + width] = original.key_qpos[
            0, old_address : old_address + width
        ]
    controls = np.zeros(model.nu)
    for index, name in enumerate(SDK_JOINTS):
        controls[model.actuator(name).id] = original.key_ctrl[0, index]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="stand_open_hands",
        qpos=" ".join(map(str, initial)),
        ctrl=" ".join(map(str, controls)),
    )
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
