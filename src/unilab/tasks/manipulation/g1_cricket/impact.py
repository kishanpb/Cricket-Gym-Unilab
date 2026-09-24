"""Versioned bat/ball compliance; all other contacts retain their original response."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .task import G1CricketCfg


@dataclass
class G1CricketImpactCfg(G1CricketCfg):
    def validate(self):
        super().validate()
        if self.sim_dt > 0.0005:
            raise ValueError("impact v1 requires physics timestep <=0.5 ms")

    def build_scene(self, source: Path, destination: Path) -> tuple[str, ...]:
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        contact = tree.getroot().find("contact")
        if contact is None:
            contact = ET.SubElement(tree.getroot(), "contact")
        ET.SubElement(
            contact,
            "pair",
            name="cricket_blade_impact_v1",
            geom1="ball_geom",
            geom2="bat_blade",
            condim="3",
            solref="0.004 1",
            solimp="0.9 0.95 0.001 0.5 2",
            friction="0.6 0.6 0.01 0.001 0.001",
            margin="0",
            gap="0",
        )
        tree.write(destination)
        return guards
