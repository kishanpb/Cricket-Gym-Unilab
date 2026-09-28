"""Orientation-constrained mechanical palm grip, not articulated finger grasping."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .bimanual_contact import G1BimanualContactCfg
from .prior import SDK_JOINTS
from .tracking import BalancedCricketReferenceAction, BalancedCricketReferenceActionCfg


@dataclass
class G1PalmGripCfg(G1BimanualContactCfg):
    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        root = tree.getroot()
        grip = root.find(".//site[@name='bat_lower_grip']")
        constraint = root.find("equality/connect[@name='second_hand_grip']")
        sensors = root.find("sensor")
        assert grip is not None and constraint is not None and sensors is not None
        sign = 1 if self.handedness == "right" else -1
        grip.set("quat", f"{np.sqrt(0.5)} 0 0 {sign * np.sqrt(0.5)}")
        constraint.tag = "weld"
        constraint.set("torquescale", "0.055")
        for name, site in (
            ("lower_grip_orientation", "bat_lower_grip"),
            ("lower_palm_orientation", f"{self.handedness}_palm"),
        ):
            ET.SubElement(sensors, "framequat", name=name, objtype="site", objname=site)
        tree.write(destination)
        return guards


@dataclass(kw_only=True)
class PalmGripActionCfg(BalancedCricketReferenceActionCfg):
    arm_scale: float = 0.25

    def build(self, env):
        return PalmGripAction(self, env)


class PalmGripAction(BalancedCricketReferenceAction):
    def __init__(self, cfg, env):
        with np.load(cfg.reference_file) as reference:
            audit = grip_reference_audit(env.get_playback_model(), reference["qpos"])
        if not audit["passed"]:
            raise ValueError(f"unsafe palm-grip reference: {audit}")
        super().__init__(cfg, env)

    def _reference_with_feedforward(self, actions):
        super()._reference_with_feedforward(actions)
        self.target[:, 15:] += (self.cfg.arm_scale - self.cfg.scale) * np.clip(
            actions[:, 15:], -1, 1
        )


def grip_reference_audit(model, poses):
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    angles = poses[:, model.jnt_qposadr[joints]]
    delta = np.diff(angles, axis=0)
    margin = min(
        float((angles - model.jnt_range[joints, 0]).min()),
        float((model.jnt_range[joints, 1] - angles).min()),
    )
    audit = {
        "minimum_joint_margin_rad": margin,
        "minimum_knee_angle_rad": float(angles[:, [3, 9]].min()),
        "maximum_frame_change_rad": float(np.abs(delta).max()),
        "maximum_second_difference_rad": float(np.abs(np.diff(delta, axis=0)).max()),
    }
    audit["passed"] = (
        margin >= 0.04 - 1e-8
        and audit["minimum_knee_angle_rad"] >= 0.25 - 1e-8
        and audit["maximum_frame_change_rad"] < 0.2
        and audit["maximum_second_difference_rad"] < 0.1
    )
    return audit


class PalmGripState:
    def __init__(self, cfg, env):
        self.sensors = env.scene.bind_sensor_data(
            (
                "lower_grip_world",
                "lower_palm_world",
                "lower_grip_orientation",
                "lower_palm_orientation",
            )
        )

    def errors(self):
        sensors = self.sensors.read()
        position = sensors[:, :3] - sensors[:, 3:6]
        dot = np.sum(sensors[:, 6:10] * sensors[:, 10:14], axis=1)
        angle = 2 * np.arccos(np.clip(np.abs(dot), 0, 1))
        return position, angle


class PalmGripObservation(PalmGripState):
    def __call__(self, env):
        sensors = self.sensors.read()
        return np.c_[sensors[:, :3] - sensors[:, 3:6], sensors[:, 6:]]


class PalmGripFailure(PalmGripState):
    def __call__(self, env, position_limit=0.02, angle_limit=0.35):
        position, angle = self.errors()
        return (np.linalg.norm(position, axis=1) > position_limit) | (angle > angle_limit)
