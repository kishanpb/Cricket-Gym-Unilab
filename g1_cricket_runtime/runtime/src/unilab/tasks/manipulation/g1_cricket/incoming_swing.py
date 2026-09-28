"""Ball-aware legs, swing timing and optional arm corrections to a two-hand reference."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from tensordict import TensorDict

from unilab.training.g1_incoming import IncomingSwingActor

from .articulated_learning import ResetArticulatedDelivery
from .coordinated_learning import SwingPath
from .loaded_balance import LoadedBalanceAction, LoadedBalanceActionCfg, LoadedBalanceObservation
from .prior import SDK_JOINTS
from .strike_timing import StrikeTimedPath, StrikeTiming
from .support_allocation import SupportAllocator
from .task import BallObservation
from .tracking import ankle_balance, root_position_balance


class ResetIncomingSwingDelivery(ResetArticulatedDelivery):
    """Sample the complete retained development pool, including its known bad feeds."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.draw_counts = np.zeros(len(self.pool), dtype=int)

    def __call__(self, env, env_ids, evaluation_pool=False, delivery_index=None):
        if delivery_index is not None:
            super().__call__(env, env_ids, delivery_index=delivery_index)
            self.draw_counts[delivery_index] += len(env_ids)
            return
        indices = (
            env_ids % len(self.pool)
            if evaluation_pool
            else env.rng.integers(len(self.pool), size=len(env_ids))
        )
        for index in range(len(self.pool)):
            ids = env_ids[indices == index]
            if len(ids):
                super().__call__(env, ids, delivery_index=index)
                self.draw_counts[index] += len(ids)


class TempoSwingReference:
    rates = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    accelerations = np.array([-1.5, 0.0, 1.5])

    def __init__(self, model, poses, times, *, fine_feedforward=False, strike_power=0.0):
        started = perf_counter()
        timing = StrikeTiming(strike_power)
        self.path = SwingPath(model, poses, times, smoothing=0.0003)
        if strike_power:
            self.path = StrikeTimedPath(self.path, timing)
        if fine_feedforward:
            times = np.linspace(times[0], times[-1], 1001)
            self.rates = np.linspace(0.5, 2, 25)
            self.accelerations = np.linspace(-1.5, 1.5, 7)
        self.dofs = model.jnt_dofadr[[model.joint(name).id for name in SDK_JOINTS]]
        allocator = SupportAllocator(model)
        forces = np.empty((len(times), len(self.rates), len(self.accelerations), 29))
        for i, phase in enumerate(times):
            allocator.prepare(self.path.pose(model, phase))
            for j, rate in enumerate(self.rates):
                for k, acceleration in enumerate(self.accelerations):
                    _, velocity, motion_acceleration = self.path.derivatives(
                        phase, rate, acceleration
                    )
                    forces[i, j, k] = allocator.solve(velocity, motion_acceleration).torque
        self.initial_force = forces[
            0, np.searchsorted(self.rates, 1), np.searchsorted(self.accelerations, 0)
        ].copy()
        self.interpolate = RegularGridInterpolator((times, self.rates, self.accelerations), forces)
        limits = model.actuator_forcerange[[model.actuator(name).id for name in SDK_JOINTS]]
        excess = np.maximum(forces - limits[:, 1], limits[:, 0] - forces)
        errors = []
        for phase, rate, acceleration in (
            (0.73, 0.625, -0.75),
            (1.91, 1.25, 0.75),
            (2.73, 1.75, -0.75),
        ):
            allocator.prepare(self.path.pose(model, phase))
            _, velocity, motion_acceleration = self.path.derivatives(phase, rate, acceleration)
            exact = allocator.solve(velocity, motion_acceleration).torque
            errors.append(
                float(np.max(np.abs(exact - self.interpolate([[phase, rate, acceleration]])[0])))
            )
        self.audit = {
            "scope": "cold feedforward interpolation spot checks, not measured support or physical qualification",
            "interpolation_check_count": len(errors),
            "maximum_spot_check_torque_error_nm": max(errors),
            "rate_bounds": self.rates[[0, -1]].tolist(),
            "maximum_rate_acceleration": float(self.accelerations[-1]),
            "fine_feedforward": fine_feedforward,
            "strike_power": strike_power,
            "timing_scope": "fixed reference timing; table rates and accelerations apply before timing composition",
            "table_shape": list(forces.shape),
            "requested_table_max_cap_excess_nm": float(max(0, excess.max())),
            "requested_table_cap_excess_elements": int((excess > 1e-8).sum()),
            "motor_scope": "prescribed table demand over all base rates, not live applied torque or a feasibility certificate",
            "construction_seconds": perf_counter() - started,
        }

    def sample(self, phase, rate, acceleration):
        tangent, velocity, _ = self.path.derivatives(phase, rate, acceleration)
        phase = np.clip(phase, self.path.times[0], self.path.times[-1])
        force = self.interpolate(np.column_stack((phase, rate, acceleration)))
        return tangent[:, self.dofs], velocity[:, self.dofs], force - self.initial_force


@dataclass(kw_only=True)
class IncomingSwingActionCfg(LoadedBalanceActionCfg):
    fine_feedforward: bool = False
    strike_power: float = 0.0
    arm_residual_scale: float = 0.0
    arm_residual_rate: float = 2.0
    ankle_feedback_scale: float = 0.0

    def build(self, env):
        return IncomingSwingAction(self, env)


class IncomingSwingAction(LoadedBalanceAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.arm_residual_scale < 0 or cfg.arm_residual_rate <= 0:
            raise ValueError("arm residual scale must be nonnegative and rate positive")
        if not 0 <= cfg.ankle_feedback_scale <= 1:
            raise ValueError("ankle feedback scale must be between zero and one")
        model = env.get_playback_model()
        self.gain = model.actuator_gainprm[self.controller.actuators, 0]
        self.velocity_gain = -model.actuator_biasprm[self.controller.actuators, 2] / self.gain
        with np.load(env.cfg.reference_file) as reference:
            self.swing = TempoSwingReference(
                model,
                reference["qpos"],
                reference["times"],
                fine_feedforward=cfg.fine_feedforward,
                strike_power=cfg.strike_power,
            )
        self.commands = np.zeros((env.num_envs, self.action_dim), dtype=np.float32)
        self.arm_residual = np.zeros((env.num_envs, 14))
        self.phase = np.zeros(env.num_envs)
        self.rate = np.ones(env.num_envs)
        self.rate_dot = np.zeros(env.num_envs)

    @property
    def action_dim(self):
        return 27 if self.cfg.arm_residual_scale else 13

    @property
    def raw_action(self):
        return self.commands

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.commands[ids] = 0
        self.arm_residual[ids] = 0
        self.phase[ids], self.rate[ids], self.rate_dot[ids] = 0, 1, 0
        self.reference[ids] = self.controller.reference[self.controller.q]

    def process_actions(self, actions):
        self.commands[:] = actions
        super().process_actions(actions[:, :12])

    def update_arm_residual(self, dt):
        desired = np.clip(self.commands[:, 13:], -1, 1) * self.cfg.arm_residual_scale
        self.arm_residual += np.clip(
            desired - self.arm_residual,
            -self.cfg.arm_residual_rate * dt,
            self.cfg.arm_residual_rate * dt,
        )

    def apply_actions(self):
        # The parent owns the leg residuals and live, torque-limited finger feedback.
        super().apply_actions()
        dt = self._env.cfg.sim_dt
        tempo = np.clip(self.commands[:, 12], -1, 1)
        requested = 1 + np.where(tempo < 0, 0.5 * tempo, tempo)
        delta = np.clip(requested - self.rate, -1.5 * dt, 1.5 * dt)
        self.rate_dot[:] = np.clip(delta / dt, -1.5, 1.5)
        tangent, velocity, force = self.swing.sample(self.phase, self.rate, self.rate_dot)
        increment = tangent + self.velocity_gain * velocity + force / self.gain
        self.reference[:] = self.controller.reference[self.controller.q] + tangent
        self.target[:, :12] += increment[:, :12]
        self.target[:, 12:] = (
            self.controller.reference[self.controller.q[12:]]
            + self.controller.offset[12:]
            + increment[:, 12:]
        )
        if self.cfg.arm_residual_scale:
            self.update_arm_residual(dt)
            self.target[:, 15:] += self.arm_residual
        if self.cfg.ankle_feedback_scale:
            position, root_velocity, _ = self.swing.path.derivatives(
                self.phase, self.rate, self.rate_dot
            )
            root_position = self.controller.reference[:3] + position[:, :3]
            data = self._entity.data
            for row in range(self._env.num_envs):
                correction = ankle_balance(
                    self.controller.reference[3:7],
                    data.root_link_quat_w[row],
                    data.root_link_ang_vel_b[row],
                    4,
                )
                correction += root_position_balance(
                    self.controller.reference[3:7],
                    data.root_link_pos_w[row]
                    - self._env.scene.env_origins[row]
                    - root_position[row],
                    data.root_link_lin_vel_w[row] - root_velocity[row, :3],
                    2,
                )
                correction = np.clip(self.cfg.ankle_feedback_scale * correction, -0.3, 0.3)
                self.target[row, [4, 10]] += correction[1]
                self.target[row, [5, 11]] += correction[0]
        self.target[:] = np.clip(
            self.target, self.controller.limits[:, 0], self.controller.limits[:, 1]
        )
        self._entity.set_joint_position_target(self.target, joint_ids=self.body_ids)
        self.phase[:] = np.minimum(
            self.swing.path.times[-1], self.phase + (self.rate + 0.5 * delta) * dt
        )
        self.rate += delta


class IncomingSwingObservation(LoadedBalanceObservation):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ball = BallObservation(cfg, env)
        action = env.action_manager.get_term("batting")
        self.arm_position_columns = (
            action.legs.config["num_obs"] + 9 + np.asarray(action.body_ids[15:])
        )
        self.arm_velocity_columns = self.arm_position_columns + len(self.robot.joint_names)
        self.leg_position_columns = (
            action.legs.config["num_obs"] + 9 + np.asarray(action.body_ids[:12])
        )
        self.leg_velocity_columns = self.leg_position_columns + len(self.robot.joint_names)

    def __call__(self, env, reference_relative_arms=False, reference_relative_legs=False):
        action = env.action_manager.get_term("batting")
        body = super().__call__(env)
        # A hierarchical actor exposes fewer commands than the physical controller.
        body = np.column_stack((body[:, : -env.action_manager.action.shape[1]], action.raw_action))
        if reference_relative_arms or reference_relative_legs:
            position, velocity, _ = action.swing.path.derivatives(
                action.phase, action.rate, action.rate_dot
            )
        if reference_relative_arms:
            # Keep the static-guard arm coordinates while exposing swing tracking errors.
            arm_dofs = action.swing.dofs[15:]
            body[:, self.arm_position_columns] -= position[:, arm_dofs]
            body[:, self.arm_velocity_columns] -= 0.05 * velocity[:, arm_dofs]
        if reference_relative_legs:
            leg_position = position[:, action.swing.dofs[:12]]
            leg_velocity = velocity[:, action.swing.dofs[:12]]
            # Legs occur in both the locomotion prefix and the whole-body input.
            body[:, 9:21] -= action.legs.config["dof_pos_scale"] * leg_position
            body[:, 21:33] -= action.legs.config["dof_vel_scale"] * leg_velocity
            body[:, self.leg_position_columns] -= leg_position
            body[:, self.leg_velocity_columns] -= 0.05 * leg_velocity
        return np.column_stack(
            (
                body[:, :204],
                self.ball(env),
                action.phase / action.swing.path.times[-1],
                action.rate / 2,
                body[:, 204:],
            )
        )


@dataclass(kw_only=True)
class FrozenGuardSwingActionCfg(IncomingSwingActionCfg):
    initial_weights: str

    def build(self, env):
        return FrozenGuardSwingAction(self, env)


class FrozenGuardSwingAction(IncomingSwingAction):
    """Keep learned leg feedback fixed while PPO controls timing and arm residuals."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.arm_residual_scale <= 0:
            raise ValueError("frozen-guard batting requires the fourteen arm residuals")
        self.commands = np.zeros((env.num_envs, 27), dtype=np.float32)
        obs = TensorDict({"policy": torch.zeros(env.num_envs, 227)}, batch_size=[env.num_envs])
        self.guard = (
            IncomingSwingActor(
                obs,
                {"actor": ["policy"]},
                "actor",
                27,
                initial_weights=cfg.initial_weights,
                distribution_cfg={
                    "class_name": "rsl_rl.modules.GaussianDistribution",
                    "init_std": 0.05,
                    "std_type": "log",
                },
            )
            .requires_grad_(False)
            .eval()
        )

    @property
    def action_dim(self):
        return 15

    def reset(self, env_ids=None):
        super().reset(env_ids)
        dones = torch.zeros(self._env.num_envs, dtype=torch.bool)
        dones[slice(None) if env_ids is None else env_ids] = True
        with torch.inference_mode():
            self.guard.reset(dones)

    def process_actions(self, actions):
        observation = self._env.observation_manager.get_term_cfg("policy", "body_grip")
        values = observation.func(self._env, reference_relative_legs=True).astype(np.float32)
        obs = TensorDict({"policy": torch.from_numpy(values)}, batch_size=[self._env.num_envs])
        with torch.inference_mode():
            legs = self.guard(obs).numpy()[:, :12]
        super().process_actions(np.column_stack((legs, actions)))
