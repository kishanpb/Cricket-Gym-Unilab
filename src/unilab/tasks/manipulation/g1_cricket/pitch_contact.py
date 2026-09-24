"""Opt-in ball/pitch response; robot support and holder compliance are unchanged."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from unilab.base import registry

from .delivery import G1CricketDeliveryCfg
from .task import make_g1_cricket_env


def add_pitch_pair(root):
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(
        contact,
        "pair",
        name="cricket_pitch_impact_v2",
        geom1="ball_geom",
        geom2="pitch",
        condim="3",
        friction=".7 .7 .01 .001 .001",
        solref=".002 .3",
        solimp=".9 .95 .001 .5 2",
    )


@dataclass
class G1CricketDeliveryPitchV2Cfg(G1CricketDeliveryCfg):
    def validate(self):
        super().validate()
        if self.sim_dt > 0.0000625:
            raise ValueError("pitch impact v2 requires physics timestep <=0.0625 ms")

    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        add_pitch_pair(tree.getroot())
        tree.write(destination)
        return guards


registry.register_env_config("G1CricketDeliveryPitchV2", G1CricketDeliveryPitchV2Cfg)
registry.register_env("G1CricketDeliveryPitchV2", make_g1_cricket_env, sim_backend="mujoco")
