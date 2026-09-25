"""Bat-center tracking and loaded-contact shaping for the one-bounce task."""

import mujoco
import numpy as np

from .substep_reward import SubstepForwardExitReward


def reference_bat_positions(model, poses):
    data = mujoco.MjData(model)
    positions = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions.append(data.site("bat_center").xpos.copy())
    return np.asarray(positions)


class BatReferenceState:
    def __init__(self, cfg, env):
        self.command = env.command_manager.get_term("motion")
        self.bat = env.scene.bind_sensor_data(("bat_center_world",))
        with np.load(cfg.params["reference_file"]) as reference:
            self.positions = reference_bat_positions(env.get_playback_model(), reference["qpos"])

    def error(self, env):
        target = self.positions[self.command.time_steps] + env.scene.env_origins
        return self.bat.read() - target


class BatReferenceObservation(BatReferenceState):
    def __call__(self, env, reference_file):
        return self.error(env)


class BatTrackingReward(BatReferenceState):
    def __call__(self, env, reference_file, std=0.08):
        return np.exp(-np.sum(self.error(env) ** 2, axis=1) / std**2)


class LoadedForwardExitReward(SubstepForwardExitReward):
    def touching(self, sensors):
        rows = self.contact_rows(sensors)
        return ((rows[..., 0] > 0) & (rows[..., 1] > 0)).any(axis=-1)
