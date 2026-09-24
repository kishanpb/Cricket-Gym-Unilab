"""Pinned Unitree velocity-policy contract, separate from locally trained cricket."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import cast

import mujoco
import numpy as np

from unilab.base.entity import Entity
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv

from .scene import CONTACT_FIELDS, CONTACT_WIDTH, build_scene
from .task import G1CricketCfg

REVISION = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
ASSET_HASHES = {
    "policy.onnx": "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f",
    "deploy.yaml": "64b04c0596a7010f39f8ac6e9ec46dc750141063cdcf2d42c83ecc444e57bc63",
}
POLICY_TO_SDK = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ]
)
POLICY_DEFAULT = np.array(
    [
        -0.1,
        -0.1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.3,
        0.3,
        0.3,
        0.3,
        -0.2,
        -0.2,
        0.25,
        -0.25,
        0,
        0,
        0,
        0,
        0.97,
        0.97,
        0.15,
        -0.15,
        0,
        0,
        0,
        0,
    ]
)
SDK_DEFAULT = np.empty(29)
SDK_DEFAULT[POLICY_TO_SDK] = POLICY_DEFAULT
SDK_KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [200] * 3 + [40] * 14)
SDK_KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [5] * 3 + [10] * 14)
SDK_JOINTS = (
    [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")]
    + [
        f"{side}_{joint}_joint"
        for side in ("left", "right")
        for joint in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
GUARD_SLOTS = 32


def build_prior_scene(
    source: Path, destination: Path, hand: str, mount: str = "legacy"
) -> tuple[str, ...]:
    if hand not in {"none", "left", "right"}:
        raise ValueError("prior fixture must be none, left or right")
    if mount not in {"legacy", "forward_down"}:
        raise ValueError("prior bat mount must be legacy or forward_down")
    bat_guards = build_scene(source, destination, "right" if hand == "none" else hand)
    tree = ET.parse(destination)
    root = tree.getroot()
    sensors = root.find("sensor")
    assert sensors is not None
    if hand == "none":
        wrist = root.find(".//body[@name='right_wrist_yaw_link']")
        assert wrist is not None
        bat = wrist.find("body[@name='cricket_bat']")
        assert bat is not None
        wrist.remove(bat)
        for sensor in list(sensors):
            if sensor.get("name", "").startswith("bat_") or sensor.get("name") == "ball_bat":
                sensors.remove(sensor)
        bat_guards = ()
    ET.SubElement(
        sensors, "framequat", name="pelvis_world_quat", objtype="site", objname="imu_in_pelvis"
    )
    guards = list(bat_guards)
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if body.find("geom") is None or name in {"cricket_ball", "cricket_bat"}:
            continue
        surfaces = [(f"prior_wicket_{i}_{name}", f"wicket_{i}") for i in range(3)]
        if name not in {"left_ankle_roll_link", "right_ankle_roll_link"}:
            surfaces.append((f"prior_ground_{name}", "pitch"))
        for sensor_name, surface in surfaces:
            ET.SubElement(
                sensors,
                "contact",
                name=sensor_name,
                body1=name,
                geom2=surface,
                data=CONTACT_FIELDS,
                num=str(GUARD_SLOTS),
                reduce="none",
            )
            guards.append(sensor_name)
    actuators = root.findall("actuator/position")
    if [actuator.get("name") for actuator in actuators] != SDK_JOINTS:
        raise ValueError("native G1 joint order differs from verified SDK mapping")
    for actuator, kp, kd in zip(actuators, SDK_KP, SDK_KD, strict=True):
        actuator.set("kp", str(kp))
        actuator.set("kv", str(kd))
        name = actuator.attrib["name"]
        ET.SubElement(sensors, "actuatorfrc", name=f"prior_force_{name}", actuator=name)
    key = root.find("keyframe/key")
    assert key is not None
    qpos = np.fromstring(key.attrib["qpos"], sep=" ")
    qpos[2] = 0.8
    qpos[7:36] = SDK_DEFAULT
    key.set("qpos", " ".join(map(str, qpos)))
    key.set("ctrl", " ".join(map(str, SDK_DEFAULT)))
    if mount == "forward_down" and hand != "none":
        # Fix the mechanical mount at construction, never during a policy step.
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        inverse_wrist = data.body(f"{hand}_wrist_yaw_link").xquat * [1, -1, -1, -1]
        desired_world = np.array([np.cos(np.pi / 8), 0, -np.sin(np.pi / 8), 0])
        mount_quat = np.empty(4)
        mujoco.mju_mulQuat(mount_quat, inverse_wrist, desired_world)
        bat = root.find(".//body[@name='cricket_bat']")
        assert bat is not None
        bat.set("quat", " ".join(map(str, mount_quat)))
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return tuple(guards)


def joint_position_policy_order(env: ManagerBasedRlEnv) -> np.ndarray:
    robot = cast(Entity, env.scene["robot"])
    return (robot.data.joint_pos - robot.data.default_joint_pos)[:, POLICY_TO_SDK]


def joint_velocity_policy_order(env: ManagerBasedRlEnv) -> np.ndarray:
    robot = cast(Entity, env.scene["robot"])
    return robot.data.joint_vel[:, POLICY_TO_SDK]


def last_action_policy_order(env: ManagerBasedRlEnv) -> np.ndarray:
    return env.action_manager.action[:, POLICY_TO_SDK]


def zero_command(env: ManagerBasedRlEnv) -> np.ndarray:
    return np.zeros((env.num_envs, 3))


def reset_jitter(env: ManagerBasedRlEnv, env_ids: np.ndarray) -> None:
    robot = cast(Entity, env.scene["robot"])
    pose = robot.data.default_joint_pos[env_ids] + env.rng.uniform(
        -0.005, 0.005, (len(env_ids), 29)
    )
    robot.write_joint_state_to_sim(pose, np.zeros_like(pose), env_ids=env_ids)


class PriorContactGuard:
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        names = cast(G1CricketCfg, env.cfg).bat_guard_sensor_names
        self.contacts = env.scene.bind_sensor_data(names)
        capacities = [4 if name.startswith("bat_") else GUARD_SLOTS for name in names]
        self.capacities = np.repeat(capacities, capacities)

    def __call__(self, env: ManagerBasedRlEnv) -> np.ndarray:
        rows = self.contacts.read().reshape(env.num_envs, -1, CONTACT_WIDTH)
        if not np.isfinite(rows).all() or np.any(rows[..., 0] > self.capacities):
            raise RuntimeError("invalid prior contact snapshot")
        return (rows[..., 0] > 0).any(axis=1)
