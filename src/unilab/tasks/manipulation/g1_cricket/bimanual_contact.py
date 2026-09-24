"""Two-hand soft-toss diagnostics with explicit fine-resolution contact pairs."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .impact import add_blade_pair
from .pitch_contact import add_pitch_pair
from .tracking import G1BimanualTrackingCfg


@dataclass
class G1BimanualContactCfg(G1BimanualTrackingCfg):
    def validate(self):
        super().validate()
        if self.sim_dt > 0.0000625:
            raise ValueError("bimanual contact requires physics timestep <=0.0625 ms")

    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        add_blade_pair(tree.getroot(), "0.002")
        add_pitch_pair(tree.getroot())
        tree.write(destination)
        return guards


class ResetSoftToss:
    def __init__(self, cfg, env):
        self.ball = env.scene["ball"]

    def __call__(self, env, env_ids):
        states = np.array(self.ball.data.default_root_state[env_ids], copy=True)
        states[:, :3] = np.array([4.0, 0.0, 1.1]) + env.scene.env_origins[env_ids]
        states[:, 3:7] = [1, 0, 0, 0]
        states[:, 7:] = [-2.8, 0, 6.5, 0, 0, 0]
        self.ball.write_root_state_to_sim(states, env_ids=env_ids)
