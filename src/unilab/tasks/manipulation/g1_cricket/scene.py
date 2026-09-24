"""Cold-path cricket scene construction around UniLab's unmodified 29-DoF G1."""

import xml.etree.ElementTree as ET
from pathlib import Path

CONTACT_FIELDS = "found force torque dist pos normal tangent"
CONTACT_SLOTS = 4
CONTACT_WIDTH = 17
BALL_CONTACT_NAMES = ("ball_bat", "ball_pitch", "ball_wicket_0", "ball_wicket_1", "ball_wicket_2")
SUPPORT_NAMES = ("left_support", "right_support")
SUPPORT_SLOTS = 32


def build_scene(source: Path, destination: Path, handedness: str) -> tuple[str, ...]:
    if handedness not in {"right", "left"}:
        raise ValueError("handedness must be right or left")
    tree = ET.parse(source)
    root = tree.getroot()
    guard_bodies = [
        body.attrib["name"]
        for body in root.findall(".//body")
        if body.find("geom") is not None and body.get("name") != f"{handedness}_wrist_yaw_link"
    ]
    compiler = root.find("compiler")
    assert compiler is not None
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", "assets")).resolve()))
    world = root.find("worldbody")
    sensors = root.find("sensor")
    assert world is not None and sensors is not None
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    asset = root.find("asset")
    assert asset is not None
    ET.SubElement(
        asset,
        "texture",
        name="cricket_sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.45 0.65 0.85",
        rgb2="0.85 0.9 0.95",
        width="512",
        height="3072",
    )
    wrist = root.find(f".//body[@name='{handedness}_wrist_yaw_link']")
    assert wrist is not None
    bat = ET.SubElement(wrist, "body", name="cricket_bat", pos="0.08 0 0")
    ET.SubElement(bat, "site", name="bat_fixture", pos="0 0 0", size="0.015")
    ET.SubElement(
        bat,
        "geom",
        name="bat_handle",
        type="capsule",
        fromto="0 0 0 0 0 -0.2",
        size="0.014",
        mass="0.12",
        rgba="0.15 0.18 0.22 1",
    )
    ET.SubElement(
        bat,
        "geom",
        name="bat_blade",
        type="box",
        pos="0 0 -0.4",
        size="0.025 0.055 0.20",
        mass="0.58",
        rgba="0.85 0.71 0.43 1",
        friction="0.6 0.01 0.001",
        condim="3",
    )
    ET.SubElement(bat, "site", name="bat_center", pos="0 0 -0.4", size="0.005")
    ET.SubElement(sensors, "force", name="bat_fixture_force", site="bat_fixture")
    ET.SubElement(sensors, "torque", name="bat_fixture_torque", site="bat_fixture")
    ET.SubElement(
        sensors, "framepos", name="bat_center_world", objtype="site", objname="bat_center"
    )

    ET.SubElement(
        world,
        "geom",
        name="pitch",
        type="plane",
        size="25 25 0.05",
        rgba="0.16 0.42 0.20 1",
        friction="0.7 0.01 0.001",
        condim="3",
    )
    ET.SubElement(
        world,
        "geom",
        name="pitch_strip",
        type="box",
        pos="9 0 0.0005",
        size="11 1.52 0.0005",
        rgba="0.55 0.47 0.32 1",
        contype="0",
        conaffinity="0",
    )
    for x in (0, 18):
        ET.SubElement(
            world,
            "geom",
            name=f"crease_{x}",
            type="box",
            pos=f"{x} 0 0.002",
            size="0.012 1.8 0.002",
            rgba="0.98 0.98 0.98 1",
            contype="0",
            conaffinity="0",
        )
    for i, y in enumerate((-0.1143, 0, 0.1143)):
        ET.SubElement(
            world,
            "geom",
            name=f"wicket_{i}",
            type="cylinder",
            pos=f"-0.6 {y} 0.3556",
            size="0.019 0.3556",
            rgba="0.94 0.92 0.79 1",
            condim="3",
        )
    for y in (-4, 4):
        ET.SubElement(
            world,
            "geom",
            name=f"net_{y}",
            type="box",
            pos=f"8 {y} 1.5",
            size="13 0.03 1.5",
            rgba="0.3 0.35 0.4 0.18",
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(world, "camera", name="cricket_side", pos="3 -5 2", xyaxes="1 0 0 0 0.3 1")
    ball = ET.SubElement(world, "body", name="cricket_ball", pos="2.5 0 0.85")
    ET.SubElement(ball, "freejoint", name="ball_free")
    ET.SubElement(
        ball,
        "geom",
        name="ball_geom",
        type="sphere",
        size="0.036",
        mass="0.156",
        rgba="0.68 0.04 0.06 1",
        condim="3",
        friction="0.5 0.01 0.001",
    )
    for name, surface in zip(
        BALL_CONTACT_NAMES, ("bat_blade", "pitch", "wicket_0", "wicket_1", "wicket_2"), strict=True
    ):
        ET.SubElement(
            sensors,
            "contact",
            name=name,
            geom1="ball_geom",
            geom2=surface,
            data=CONTACT_FIELDS,
            num=str(CONTACT_SLOTS),
            reduce="none",
        )
    for side in ("left", "right"):
        ET.SubElement(
            sensors,
            "contact",
            name=f"{side}_support",
            geom1="pitch",
            body2=f"{side}_ankle_roll_link",
            data=CONTACT_FIELDS,
            num=str(SUPPORT_SLOTS),
            reduce="none",
        )

    guard_names = []
    guard_targets = [("bat_pitch", "geom2", "pitch")]
    guard_targets += [(f"bat_wicket_{i}", "geom2", f"wicket_{i}") for i in range(3)]
    guard_targets += [(f"bat_robot_{body}", "body2", body) for body in guard_bodies]
    for name, target_type, target in guard_targets:
        ET.SubElement(
            sensors,
            "contact",
            {
                "name": name,
                "body1": "cricket_bat",
                target_type: target,
                "data": CONTACT_FIELDS,
                "num": str(CONTACT_SLOTS),
                "reduce": "none",
            },
        )
        guard_names.append(name)

    stand = ET.parse(source.parent / "scene_flat.xml").find("keyframe/key[@name='stand']")
    assert stand is not None
    qpos = stand.get("qpos", "").split()
    qpos[1] = "0.30" if handedness == "right" else "-0.30"
    qpos[2] = "0.758"
    ET.SubElement(
        ET.SubElement(root, "keyframe"),
        "key",
        name="cricket_stand",
        qpos=" ".join(qpos + ["2.5", "0", "0.85", "1", "0", "0", "0"]),
        ctrl=stand.get("ctrl", ""),
    )
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return tuple(guard_names)
