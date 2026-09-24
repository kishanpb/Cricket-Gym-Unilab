"""Versioned bat/ball compliance; all other contacts retain their original response."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .task import G1CricketCfg


def add_blade_pair(root, time_constant="0.004"):
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(
        contact,
        "pair",
        name="cricket_blade_impact_v1",
        geom1="ball_geom",
        geom2="bat_blade",
        condim="3",
        solref=f"{time_constant} 1",
        solimp="0.9 0.95 0.001 0.5 2",
        friction="0.6 0.6 0.01 0.001 0.001",
        margin="0",
        gap="0",
    )


@dataclass
class G1CricketImpactCfg(G1CricketCfg):
    def validate(self):
        super().validate()
        if self.sim_dt > 0.0005:
            raise ValueError("impact v1 requires physics timestep <=0.5 ms")

    def build_scene(self, source: Path, destination: Path) -> tuple[str, ...]:
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        add_blade_pair(tree.getroot())
        tree.write(destination)
        return guards
