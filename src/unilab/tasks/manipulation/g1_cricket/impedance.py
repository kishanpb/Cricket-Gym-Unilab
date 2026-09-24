"""Opt-in throwing-shoulder controller retuning, not increased motor authority."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from unilab.base import registry

from .overarm import OverarmActionCfg
from .pitch_contact import G1CricketDeliveryPitchV2Cfg
from .task import make_g1_cricket_env


@dataclass
class G1CricketShoulderDampingCfg(G1CricketDeliveryPitchV2Cfg):
    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        actuator = tree.getroot().find(
            f"actuator/position[@name='{self.handedness}_shoulder_pitch_joint']"
        )
        assert actuator is not None
        actuator.set("kv", "2")
        tree.write(destination)
        return guards


@dataclass(kw_only=True)
class ShoulderDampingActionCfg(OverarmActionCfg):
    def build(self, env):
        # The action-config identity is part of the existing strict checkpoint contract.
        model = env.get_playback_model()
        actuator = model.actuator(f"{env.cfg.handedness}_shoulder_pitch_joint").id
        np.testing.assert_array_equal(model.actuator_gainprm[actuator, :1], [40])
        np.testing.assert_array_equal(model.actuator_biasprm[actuator, 1:3], [-40, -2])
        np.testing.assert_array_equal(model.actuator_forcerange[actuator], [-25, 25])
        return super().build(env)


registry.register_env_config("G1CricketShoulderDamping", G1CricketShoulderDampingCfg)
registry.register_env("G1CricketShoulderDamping", make_g1_cricket_env, sim_backend="mujoco")
