"""Continuous delivery prototype: reference arms over live locomotion feedback."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

from .approach_feedback import ApproachFeedbackAction, ApproachFeedbackActionCfg
from .prior import SDK_JOINTS
from .running import END_TIME, GATHER_TIME, RELEASE_TIME

REFERENCE_START = 2.8


def smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def arm_weight(time):
    local = np.asarray(time) - REFERENCE_START
    return smoothstep(local / GATHER_TIME) * (1 - smoothstep((local - END_TIME) / 0.6))


def moving_command(env):
    time = env.episode_length_buf * env.step_dt
    command = np.zeros((env.num_envs, 3))
    command[:, 0] = np.interp(time, [0, 1, 2, 4.8, 5.8, 8], [0, 0, 1, 1, 0, 0])
    return command


def lane_heading_command(env):
    robot = env.scene["robot"].data
    command = moving_command(env)
    w, x, y, z = robot.root_link_quat_w.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lane = robot.default_root_state[:, 1] + env.scene.env_origins[:, 1]
    sideways = np.clip(2 * (lane - robot.root_link_pos_w[:, 1]), -0.4, 0.4)
    forward = command[:, 0].copy()
    command[:, 0] = np.cos(yaw) * forward + np.sin(yaw) * sideways
    command[:, 1] = -np.sin(yaw) * forward + np.cos(yaw) * sideways
    command[:, 2] = np.clip(-2 * yaw, -0.5, 0.5)
    return command


def forward_release_ready(position, velocity, shoulder, elbow):
    return (
        (position[:, 2] > shoulder[:, 2] + 0.12)
        & (elbow[:, 2] > shoulder[:, 2])
        & (velocity[:, 0] > 1)
        & (np.abs(velocity[:, 1]) < 2)
        & (velocity[:, 2] < 0)
    )


@dataclass(kw_only=True)
class MovingDeliveryActionCfg(ApproachFeedbackActionCfg):
    reference_directory: str
    arm_velocity_feedforward: bool = False
    state_release: bool = False

    def build(self, env):
        return MovingDeliveryAction(self, env)


class MovingDeliveryAction(ApproachFeedbackAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        model = env.get_playback_model()
        joints = np.array([model.joint(name).id for name in SDK_JOINTS[15:]])
        path = Path(cfg.reference_directory) / f"{env.cfg.handedness}_dense_reference.npz"
        with np.load(path) as reference:
            self.arm_reference = PchipInterpolator(
                reference["times"], reference["qpos"][:, model.jnt_qposadr[joints]], axis=0
            )
        self.arm_limits = model.jnt_range[joints].copy()
        self.velocity_lead = -model.actuator_biasprm[15:, 2] / model.actuator_gainprm[15:, 0]
        self.kinematic_model = model
        self.kinematic_data = mujoco.MjData(model)
        self.delivery_arm = [
            model.body(f"{env.cfg.handedness}_{name}_link").id
            for name in ("shoulder_roll", "elbow")
        ]

    def release_ready(self):
        positions = []
        for snapshot in self._env.get_physics_state_snapshot():
            mujoco.mj_setState(
                self.kinematic_model,
                self.kinematic_data,
                snapshot,
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            mujoco.mj_kinematics(self.kinematic_model, self.kinematic_data)
            positions.append(self.kinematic_data.xpos[self.delivery_arm].copy())
        positions = np.array(positions)
        ball = self._env.scene["ball"].data
        return forward_release_ready(
            ball.root_link_pos_w, ball.root_link_lin_vel_w, positions[:, 0], positions[:, 1]
        )

    def process_actions(self, actions):
        time = self._env.episode_length_buf * self._env.step_dt
        self._hold_action[:, 7] = time >= REFERENCE_START + RELEASE_TIME
        if self.cfg.state_release:
            self._hold_action[:, 7] = (
                (time >= REFERENCE_START + GATHER_TIME)
                & (time <= REFERENCE_START + END_TIME)
                & self.release_ready()
            )
        super().process_actions(actions)
        local = np.clip(time - REFERENCE_START, 0, END_TIME)
        target = self.arm_reference(local)
        if self.cfg.arm_velocity_feedforward:
            moving = (time >= REFERENCE_START) & (time <= REFERENCE_START + END_TIME)
            target += moving[:, None] * self.arm_reference(local, nu=1) * self.velocity_lead
        weight = arm_weight(time)[:, None]
        # Only motor targets change; legs and trunk retain the live prior's feedback.
        current = self.processed_action[:, 15:]
        current[:] = np.clip(
            current + weight * (target - current), self.arm_limits[:, 0], self.arm_limits[:, 1]
        )
