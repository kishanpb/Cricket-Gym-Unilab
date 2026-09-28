"""Whole-body residuals around measured approach commands, not a qualified gait."""

from dataclasses import dataclass

import numpy as np

from unilab.tasks.motion_tracking.common.manager_terms import MotionCommand, MotionCommandCfg
from unilab.utils.rotation import np_quat_apply_batched

from .bowling import HOLDER_SENSORS
from .prior import SDK_JOINTS
from .scene import CONTACT_WIDTH
from .tracking import CricketReferenceAction, CricketReferenceActionCfg


@dataclass(kw_only=True)
class MeasuredApproachCommandCfg(MotionCommandCfg):
    reference_file: str

    def build(self, env):
        return MeasuredApproachCommand(self, env)


class MeasuredApproachCommand(MotionCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.sampling_mode not in ("start", "uniform"):
            raise ValueError("measured approach supports start or uniform command sampling")
        if (
            np.any(self._pose_range)
            or np.any(self._velocity_range)
            or any(self._joint_position_range)
            or any(self._joint_default_position_range)
            or not cfg.params.truncate_on_clip_end
        ):
            raise ValueError("measured approach requires unperturbed resets and clip truncation")
        with np.load(cfg.reference_file) as saved:
            self.poses = saved["qpos"].copy()
            self.velocities = saved["qvel"].copy()
            self.controls = saved["controls"].copy()
            names = tuple(saved["joint_names"])
        model = env.get_playback_model()
        count = len(self.controls)
        if (
            count == 0
            or self.controls.shape != (count, 29)
            or self.poses.shape != (count + 1, model.nq)
            or self.velocities.shape != (count + 1, model.nv)
            or names != tuple(SDK_JOINTS)
            or self.motion.num_frames != count + 1
            or self.motion.num_clips != 1
            or not np.isclose(self.motion.fps * env.step_dt, 1)
            or not all(np.isfinite(x).all() for x in (self.poses, self.velocities, self.controls))
        ):
            raise ValueError("measured approach requires one finite state/command-aligned clip")
        joints = [model.joint(name).id for name in SDK_JOINTS]
        self.joint_qpos = model.jnt_qposadr[joints]
        self.joint_qvel = model.jnt_dofadr[joints]
        motion = self.motion.get_motion_at_frame(np.arange(count + 1))
        np.testing.assert_array_equal(motion.joint_pos, self.poses[:, self.joint_qpos])
        np.testing.assert_array_equal(motion.joint_vel, self.velocities[:, self.joint_qvel])
        self.ball = env.scene["ball"]
        ball_joint = model.body("cricket_ball").jntadr[0]
        self.ball_qpos = model.jnt_qposadr[ball_joint]
        self.ball_qvel = model.jnt_dofadr[ball_joint]

    def _resample_command(self, env_ids):
        frames = self.sampler.sample_frames(env_ids)
        # The final measured state has no outgoing command.
        terminal = frames == len(self.controls)
        while terminal.any():
            frames[terminal] = self.sampler.sample_frames(env_ids[terminal])
            terminal = frames == len(self.controls)
        pose, velocity = self.poses[frames], self.velocities[frames]

        def root_state(qadr, dadr):
            quaternion = pose[:, qadr + 3 : qadr + 7]
            return np.concatenate(
                (
                    pose[:, qadr : qadr + 3] + self._env.scene.env_origins[env_ids],
                    quaternion,
                    velocity[:, dadr : dadr + 3],
                    np_quat_apply_batched(quaternion, velocity[:, dadr + 3 : dadr + 6]),
                ),
                axis=1,
            )

        self.robot.write_root_state_to_sim(root_state(0, 0), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            pose[:, self.joint_qpos], velocity[:, self.joint_qvel], env_ids=env_ids
        )
        self.ball.write_root_state_to_sim(
            root_state(self.ball_qpos, self.ball_qvel), env_ids=env_ids
        )
        motion = self.motion.get_motion_at_frame(frames)
        self._ingest_motion_rows(env_ids, motion)
        self._resample_ingested_ids = env_ids
        self._resample_motion = motion


@dataclass(kw_only=True)
class MeasuredApproachActionCfg(CricketReferenceActionCfg):
    def build(self, env):
        return MeasuredApproachAction(self, env)


class MeasuredApproachAction(CricketReferenceAction):
    sensor_names = HOLDER_SENSORS

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.released = np.zeros(env.num_envs, dtype=bool)
        self.peak_load = np.zeros(env.num_envs)
        self.impulse_world = np.zeros((env.num_envs, 3))
        self.touch_fraction = np.zeros(env.num_envs)
        self.constraints = env.equality_constraints
        env.set_substep_observer(self.sensor_names, "cricket_ball", self.observe)

    @property
    def processed_action(self):
        return self.target

    def process_actions(self, actions):
        self._raw[:] = actions
        self.target[:] = self.command.controls[self.command.time_steps]
        self.target += self.cfg.scale * np.clip(actions, -1, 1)

    def observe(self, sensors, integrated_velocity):
        force, quaternion = sensors[..., :3], sensors[..., 3:7]
        self.peak_load[:] = np.linalg.norm(force, axis=-1).max(axis=1)
        world = np_quat_apply_batched(quaternion.reshape(-1, 4), force.reshape(-1, 3))
        self.impulse_world[:] = world.reshape(force.shape).sum(axis=1) * self._env.physics_dt
        contacts = sensors[..., 7:].reshape(*sensors.shape[:2], 4, CONTACT_WIDTH)
        if np.any(contacts[..., 0] > 4):
            raise RuntimeError("measured approach hand contact capacity exceeded")
        self.touch_fraction[:] = (contacts[..., 0] > 0).any(axis=-1).mean(axis=1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.released, self.peak_load, self.impulse_world, self.touch_fraction):
            value[ids] = 0
        rows = np.arange(self.num_envs)[ids]
        self.constraints.set_equality_active(rows, np.ones((len(rows), 1), dtype=bool))


def measured_clip_end(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= len(command.controls) - 1


def measured_phase(env):
    command = env.command_manager.get_term("motion")
    return (command.time_steps / len(command.controls))[:, None]
