"""Versioned delivery scene; contact and cricket qualification remain independent."""

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from unilab.base import registry

from .bowling import G1CricketBowlingCfg
from .prior import GUARD_SLOTS
from .scene import CONTACT_FIELDS
from .task import make_g1_cricket_env

POPPING_X = 0.0
BOWLING_X = -1.22
TARGET_X = BOWLING_X + 20.12
TARGET_POPPING_X = TARGET_X - 1.22
RETURN_Y = 1.32


@dataclass
class G1CricketDeliveryCfg(G1CricketBowlingCfg):
    def build_scene(self, source, destination):
        guards = list(super().build_scene(source, destination))
        tree = ET.parse(destination)
        root = tree.getroot()
        world, sensors, key = root.find("worldbody"), root.find("sensor"), root.find("keyframe/key")
        assert world is not None and sensors is not None and key is not None
        qpos = np.fromstring(key.attrib["qpos"], sep=" ")
        side = 0.5 if self.handedness == "right" else -0.5
        qpos[1] += side
        qpos[-6] += side
        key.set("qpos", " ".join(map(str, qpos)))
        for index in range(3):
            wicket = world.find(f"geom[@name='wicket_{index}']")
            assert wicket is not None
            pos = np.fromstring(wicket.attrib["pos"], sep=" ")
            pos[0] = TARGET_X
            wicket.set("pos", " ".join(map(str, pos)))
            bowler_wicket = copy.deepcopy(wicket)
            bowler_wicket.set("name", f"bowler_wicket_{index}")
            pos[0] = BOWLING_X
            bowler_wicket.set("pos", " ".join(map(str, pos)))
            world.append(bowler_wicket)
        for name in ("crease_0", "crease_18"):
            crease = world.find(f"geom[@name='{name}']")
            assert crease is not None
            world.remove(crease)
        # Legal edges, not line centres: popping back edges and return inside edges.
        for end, bowling_x, popping_x, direction in (
            ("bowler", BOWLING_X, POPPING_X, 1),
            ("striker", TARGET_X, TARGET_POPPING_X, -1),
        ):
            for kind, x, half_width in (
                ("bowling", bowling_x, RETURN_Y),
                ("popping", popping_x, 1.83),
            ):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{end}_{kind}_crease",
                    type="box",
                    pos=f"{x + direction * 0.012} 0 .002",
                    size=f".012 {half_width} .002",
                    rgba=".98 .98 .98 1",
                    contype="0",
                    conaffinity="0",
                )
            for sign in (-1, 1):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"{end}_return_{sign}",
                    type="box",
                    pos=f"{popping_x - direction * 1.22} {sign * (RETURN_Y + 0.012)} .002",
                    size="1.22 .012 .002",
                    rgba=".98 .98 .98 1",
                    contype="0",
                    conaffinity="0",
                )
        for body in root.findall(".//body"):
            name = body.get("name", "")
            if body.find("geom") is None or name == "cricket_ball":
                continue
            for index in range(3):
                sensor_name = f"delivery_bowler_wicket_{index}_{name}"
                ET.SubElement(
                    sensors,
                    "contact",
                    name=sensor_name,
                    body1=name,
                    geom2=f"bowler_wicket_{index}",
                    num=str(GUARD_SLOTS),
                    data=CONTACT_FIELDS,
                    reduce="none",
                )
                guards.append(sensor_name)
        tree.write(destination)
        return tuple(guards)


registry.register_env_config("G1CricketDelivery", G1CricketDeliveryCfg)
registry.register_env("G1CricketDelivery", make_g1_cricket_env, sim_backend="mujoco")
