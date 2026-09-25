"""Contact-synchronous gait feedback for measured-command approach learning."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from unilab.base import registry

from .approach import G1CricketApproachCfg
from .approach_tracking import MeasuredApproachAction, MeasuredApproachActionCfg
from .bowling import HOLDER_SENSORS
from .scene import CONTACT_WIDTH, SUPPORT_NAMES, SUPPORT_SLOTS
from .task import make_g1_cricket_env

FOOT_MOTION = tuple(
    f"approach_{side}_{quantity}"
    for side in ("left", "right")
    for quantity in ("position", "linear_velocity", "angular_velocity")
)
HOLDER_WIDTH = 7 + 4 * CONTACT_WIDTH
SUPPORT_WIDTH = 2 * SUPPORT_SLOTS * CONTACT_WIDTH


@dataclass
class G1CricketApproachLearningCfg(G1CricketApproachCfg):
    def build_scene(self, source, destination):
        guards = super().build_scene(source, destination)
        tree = ET.parse(destination)
        sensors = tree.getroot().find("sensor")
        for side in ("left", "right"):
            for quantity, tag in (
                ("position", "framepos"),
                ("linear_velocity", "framelinvel"),
                ("angular_velocity", "frameangvel"),
            ):
                ET.SubElement(
                    sensors,
                    tag,
                    name=f"approach_{side}_{quantity}",
                    objtype="xbody",
                    objname=f"{side}_ankle_roll_link",
                )
        tree.write(destination)
        return guards


def loaded_contact_slip(support, motion):
    """Foot contact-point speeds using only co-timed solved-frame sensors."""
    loaded = (support[..., 0] > 0) & (support[..., 1] > 1)
    offset = support[..., 8:11] - motion[..., None, :3]
    point_velocity = motion[..., None, 3:6] + np.cross(motion[..., None, 6:9], offset)
    normal = support[..., 11:14]
    tangent = point_velocity - np.sum(point_velocity * normal, axis=-1, keepdims=True) * normal
    speed = np.where(loaded, np.linalg.norm(tangent, axis=-1), 0).max(axis=-1)
    normal_load = np.where(support[..., 0] > 0, support[..., 1], 0).sum(axis=-1)
    return speed, normal_load


@dataclass(kw_only=True)
class ApproachLearningActionCfg(MeasuredApproachActionCfg):
    def build(self, env):
        return ApproachLearningAction(self, env)


class ApproachLearningAction(MeasuredApproachAction):
    sensor_names = HOLDER_SENSORS + SUPPORT_NAMES + FOOT_MOTION

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.just_released = np.zeros(env.num_envs, dtype=bool)
        self.slip_squared = np.zeros((env.num_envs, 2))
        self.slip_peak = np.zeros((env.num_envs, 2))
        self.normal_load = np.zeros((env.num_envs, 2))

    def observe(self, sensors, integrated_velocity):
        super().observe(sensors[..., :HOLDER_WIDTH], integrated_velocity)
        support = sensors[..., HOLDER_WIDTH : HOLDER_WIDTH + SUPPORT_WIDTH].reshape(
            *sensors.shape[:2], 2, SUPPORT_SLOTS, CONTACT_WIDTH
        )
        if np.any(support[..., 0] > SUPPORT_SLOTS):
            raise RuntimeError("approach support contact capacity exceeded")
        motion = sensors[..., HOLDER_WIDTH + SUPPORT_WIDTH :].reshape(*sensors.shape[:2], 2, 9)
        speed, normal_load = loaded_contact_slip(support, motion)
        self.slip_squared[:] = np.square(speed).mean(axis=1)
        self.slip_peak[:] = speed.max(axis=1)
        self.normal_load[:] = normal_load.mean(axis=1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.just_released, self.slip_squared, self.slip_peak, self.normal_load):
            value[ids] = 0


def lane_error(env):
    command = env.command_manager.get_term("motion")
    return env.scene["robot"].data.root_link_pos_w[:, 1] - (
        command.poses[0, 1] + env.scene.env_origins[:, 1]
    )


def gait_observation(env):
    action = env.action_manager.get_term("reference")
    return np.column_stack(
        (np.sqrt(action.slip_squared), action.normal_load / 100, lane_error(env))
    )


def loaded_slip_cost(env):
    return env.action_manager.get_term("reference").slip_squared.sum(axis=1)


def lane_tracking(env):
    return np.exp(-np.square(lane_error(env) / 0.1))


registry.register_env_config("G1CricketApproachLearning", G1CricketApproachLearningCfg)
registry.register_env("G1CricketApproachLearning", make_g1_cricket_env, sim_backend="mujoco")
