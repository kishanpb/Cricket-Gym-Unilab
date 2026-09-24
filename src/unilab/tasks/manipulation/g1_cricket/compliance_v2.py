"""Opt-in 2 ms bat-ball response for matched-resolution model transfer."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from unilab.base import registry

from .impact import G1CricketImpactCfg
from .task import make_g1_cricket_env


@dataclass
class G1CricketImpactV2Cfg(G1CricketImpactCfg):
    def validate(self):
        super().validate()
        if self.sim_dt > 0.0000625:
            raise ValueError("impact v2 requires physics timestep <=0.0625 ms")

    def build_scene(self, source: Path, destination: Path) -> tuple[str, ...]:
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        pair = tree.getroot().find("contact/pair[@name='cricket_blade_impact_v1']")
        assert pair is not None
        pair.set("solref", "0.002 1")
        tree.write(destination)
        return guards


registry.register_env_config("G1CricketImpactV2", G1CricketImpactV2Cfg)
registry.register_env("G1CricketImpactV2", make_g1_cricket_env, sim_backend="mujoco")
