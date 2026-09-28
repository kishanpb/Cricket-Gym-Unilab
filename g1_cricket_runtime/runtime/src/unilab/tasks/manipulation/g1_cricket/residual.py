"""Learned arm corrections around an explicitly frozen external locomotion prior."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from unilab.managers.action_manager import ActionTerm, ActionTermCfg
from unilab.utils.rotation import np_quat_apply_inverse_batched

from .prior import ASSET_HASHES, POLICY_DEFAULT, POLICY_TO_SDK, SDK_KD, SDK_KP

RESIDUAL_LIMITS = np.array([0.35, 0.35, 0.35, 0.35, 0.15, 0.15, 0.15], dtype=np.float32)
TOSS_OFFSETS = (-0.12, -0.10, 0.0)


@dataclass(kw_only=True)
class FrozenPriorResidualCfg(ActionTermCfg):
    asset_directory: str

    def build(self, env):
        return FrozenPriorResidual(self, env)


class FrozenPriorResidual(ActionTerm):
    def __init__(self, cfg, env):
        import onnxruntime as ort

        super().__init__(cfg, env)
        directory = Path(cfg.asset_directory).expanduser()
        for name, expected in ASSET_HASHES.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"unexpected external Unitree asset: {name}")
        contract = OmegaConf.load(directory / "deploy.yaml")
        for name, expected in (
            ("joint_ids_map", POLICY_TO_SDK),
            ("default_joint_pos", POLICY_DEFAULT),
            ("stiffness", SDK_KP),
            ("damping", SDK_KD),
        ):
            np.testing.assert_array_equal(contract[name], expected)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "policy.onnx"), options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].shape != [1, 480] or self.session.get_outputs()[
            0
        ].shape != [1, 29]:
            raise ValueError("unexpected frozen prior dimensions")
        self.arm_ids = np.arange(22, 29) if env.cfg.handedness != "left" else np.arange(15, 22)
        self._raw = np.zeros((env.num_envs, 7), dtype=np.float32)
        self.baseline_action = np.zeros((env.num_envs, 29), dtype=np.float32)
        self.residual_radians = np.zeros_like(self._raw)
        self.processed_action = self._entity.data.default_joint_pos.copy()

    @property
    def action_dim(self):
        return 7

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions
        observations = self._env.observation_manager.compute(update_history=False)["policy"]
        for index, observation in enumerate(observations):
            self.baseline_action[index] = self.session.run(
                ["actions"], {"obs": observation.astype(np.float32)[None]}
            )[0][0]
        if not np.isfinite(self.baseline_action).all():
            raise RuntimeError("non-finite frozen prior action")
        sdk = np.empty_like(self.baseline_action)
        sdk[:, POLICY_TO_SDK] = self.baseline_action
        self.processed_action[:] = self._entity.data.default_joint_pos + 0.25 * sdk
        self.residual_radians[:] = np.clip(actions, -1, 1) * RESIDUAL_LIMITS
        self.processed_action[:, self.arm_ids] += self.residual_radians

    def apply_actions(self):
        self._entity.set_joint_position_target(self.processed_action)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.baseline_action[ids] = 0
        self.residual_radians[ids] = 0
        self.processed_action[ids] = self._entity.data.default_joint_pos[ids]


def last_baseline_action(env):
    return env.action_manager.get_term("residual").baseline_action


class CricketObservation:
    def __init__(self, cfg, env):
        self.robot, self.ball = env.scene["robot"], env.scene["ball"]
        self.bat = env.scene.bind_sensor_data(("bat_center_world",))
        self.hand = 1.0 if env.cfg.handedness == "right" else -1.0

    def __call__(self, env):
        root = self.robot.data.root_link_quat_w
        relative = np_quat_apply_inverse_batched(
            root, self.ball.data.root_link_pos_w - self.bat.read()
        )
        velocity = np_quat_apply_inverse_batched(root, self.ball.data.root_link_lin_vel_w)
        bat_quat = self.robot.data.body_link_quat_w[:, 1]
        return np.concatenate(
            (
                self.robot.data.projected_gravity_b,
                self.robot.data.root_link_lin_vel_w,
                self.robot.data.root_link_ang_vel_w,
                self.robot.data.root_link_pos_w[:, 2:3],
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                self.robot.data.joint_vel * 0.05,
                relative,
                velocity,
                bat_quat,
                env.action_manager.action,
                last_baseline_action(env),
                np.full((env.num_envs, 1), self.hand),
            ),
            axis=1,
        )


class ResetToss:
    def __init__(self, cfg, env):
        self.ball = env.scene["ball"]
        self.sign = 1 if env.cfg.handedness == "right" else -1
        self.center_y = 0.0432396123 if self.sign == 1 else -0.0432296123

    def __call__(self, env, env_ids, offsets=TOSS_OFFSETS, enabled=True):
        if not enabled:
            return
        states = self.ball.data.default_root_state[env_ids].copy()
        states[:, :3] = [1.1013225616, self.center_y, 0.8717760803]
        states[:, 1] += self.sign * env.rng.choice(offsets, len(env_ids))
        states[:, 7:13] = 0
        states[:, 7] = -2.5
        self.ball.write_root_state_to_sim(states, env_ids=env_ids)


class BattingReward:
    """Approach shaping and a one-shot sampled blade-contact bonus, not a success gate."""

    def __init__(self, cfg, env):
        self.ball = env.scene["ball"]
        self.bat = env.scene.bind_sensor_data(("bat_center_world",))
        self.contact = env.scene.bind_sensor_data(("ball_bat",))
        self.hit_seen = np.zeros(env.num_envs, dtype=bool)

    def reset(self, env_ids=None):
        self.hit_seen[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env):
        distance = np.linalg.norm(self.ball.data.root_link_pos_w - self.bat.read(), axis=1)
        velocity = self.ball.data.root_link_lin_vel_w[:, 0]
        approach = 5 * np.exp(-((distance / 0.18) ** 2)) * (velocity < 0)
        touching = (self.contact.read().reshape(env.num_envs, 4, 17)[..., 0] > 0).any(axis=1)
        fresh_hit = touching & ~self.hit_seen
        self.hit_seen |= touching
        bonus = fresh_hit * (5 + np.clip(velocity, 0, 6)) / env.step_dt
        return approach + bonus


def failure_penalty(env):
    return env.termination_manager.terminated.astype(float) / env.step_dt
