"""Measured locomotion teacher for the approach stage, before gather and release."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from unilab.base import registry

from .pitch_contact import G1CricketDeliveryPitchV2Cfg
from .task import make_g1_cricket_env


def approach_speed(time):
    return np.interp(time, [0, 1, 2, 4, 5, 8], [0, 0, 1, 1, 0, 0])


def approach_command(env):
    command = np.zeros((env.num_envs, 3))
    command[:, 0] = approach_speed(env.episode_length_buf * env.step_dt)
    return command


@dataclass
class G1CricketApproachCfg(G1CricketDeliveryPitchV2Cfg):
    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        key = tree.getroot().find("keyframe/key")
        pose = np.fromstring(key.attrib["qpos"], sep=" ")
        # Translate robot and held ball together, behind the delivery stride.
        shift = np.array([-3.5, 0.2 if self.handedness == "right" else -0.2, 0])
        pose[:3] += shift
        pose[-7:-4] += shift
        key.set("qpos", " ".join(map(str, pose)))
        tree.write(destination)
        return guards


registry.register_env_config("G1CricketApproach", G1CricketApproachCfg)
registry.register_env("G1CricketApproach", make_g1_cricket_env, sim_backend="mujoco")
