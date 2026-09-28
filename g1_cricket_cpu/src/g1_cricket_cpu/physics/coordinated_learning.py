"""Learn swing tempo while preserving a common two-hand reference path."""
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator, make_smoothing_spline
from .articulated_batting import body_support_torque
from .articulated_learning import ArticulatedBattingAction, ArticulatedBattingActionCfg, ArticulatedObservation
from .articulated_swing import support_motion_increment
from .prior import SDK_JOINTS
from .tracking import ankle_balance, root_position_balance

def hip_orientation_correction(reference, quaternion, angular_velocity, gains):
    error = np.empty(3)
    mujoco.mju_subQuat(error, quaternion, reference)
    reference_rotation, rotation = (np.empty(9), np.empty(9))
    mujoco.mju_quat2Mat(reference_rotation, reference)
    mujoco.mju_quat2Mat(rotation, quaternion)
    velocity = reference_rotation.reshape(3, 3).T @ rotation.reshape(3, 3) @ angular_velocity
    return np.clip(np.asarray(gains) * (0.7 * error + 0.1 * velocity), -0.15, 0.15)

class SwingPath:
    """C2 interpolation in generalized coordinates for this fixed-axis reference."""

    def __init__(self, model, poses, times, smoothing=0.0):
        self.initial = poses[0].copy()
        self.times = times.copy()
        tangent = np.empty((len(poses), model.nv))
        for row, pose in enumerate(poses):
            mujoco.mj_differentiatePos(model, tangent[row], 1, self.initial, pose)
        if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and np.max(np.abs(tangent[:, 3:6])) > 1e-08:
            raise ValueError('swing balance requires a constant root orientation')
        for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
            dof = model.jnt_dofadr[joint]
            rotation = tangent[:, dof + 3:dof + 6]
            axis = rotation[np.linalg.norm(rotation, axis=1).argmax()]
            if np.max(np.abs(np.cross(rotation, axis))) > 1e-08:
                raise ValueError('swing path requires fixed-axis free-joint rotations')
            if np.max(np.linalg.norm(np.diff(rotation, axis=0), axis=1)) > 0.5:
                raise ValueError('reference rotation crosses a tangent-coordinate branch')
        if smoothing:
            smoothed = make_smoothing_spline(times, tangent, lam=smoothing)(times)
            smoothed[[0, -1]] = tangent[[0, -1]]
            tangent = smoothed
        self.spline = CubicSpline(times, tangent, bc_type='clamped')

    def derivatives(self, phase, rate, rate_dot):
        phase = np.asarray(phase)
        s = np.clip(phase, self.times[0], self.times[-1])
        position = self.spline(s)
        first, second = (self.spline(s, 1), self.spline(s, 2))
        held = (phase < self.times[0]) | (phase >= self.times[-1])
        velocity = first * np.asarray(rate)[..., None]
        acceleration = second * np.asarray(rate)[..., None] ** 2 + first * np.asarray(rate_dot)[..., None]
        velocity = np.where(held[..., None], 0, velocity)
        acceleration = np.where(held[..., None], 0, acceleration)
        return (position, velocity, acceleration)

    def pose(self, model, phase):
        pose = self.initial.copy()
        mujoco.mj_integratePos(model, pose, self.derivatives(phase, 0, 0)[0], 1)
        return pose

class SwingFeedforward:
    """Cold support solves; interpolated torques are assistance, not contact evidence."""
    rates = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    accelerations = np.array([-1.5, 0.0, 1.5])

    def __init__(self, model, path):
        values = np.empty((len(path.times), len(self.rates), len(self.accelerations), 29))
        support_residual = 0.0
        for i, phase in enumerate(path.times):
            pose = path.pose(model, phase)
            gravity = body_support_torque(model, pose)[0]
            for j, rate in enumerate(self.rates):
                for k, rate_dot in enumerate(self.accelerations):
                    _, velocity, acceleration = path.derivatives(phase, rate, rate_dot)
                    if np.linalg.norm(velocity) + np.linalg.norm(acceleration) < 1e-09:
                        values[i, j, k] = gravity
                    else:
                        increment, residual = support_motion_increment(model, pose, velocity, acceleration)
                        values[i, j, k] = gravity + increment
                        support_residual = max(support_residual, residual)
        self.interpolate = RegularGridInterpolator((path.times, self.rates, self.accelerations), values)
        errors = []
        for phase in path.times[25:-25:25] + 0.01:
            pose = path.pose(model, phase)
            gravity = body_support_torque(model, pose)[0]
            for rate, rate_dot in ((0.625, -0.75), (1.25, 0.75), (1.75, -0.75)):
                _, velocity, acceleration = path.derivatives(phase, rate, rate_dot)
                exact = gravity + support_motion_increment(model, pose, velocity, acceleration)[0]
                estimate = self.interpolate([[phase, rate, rate_dot]])[0]
                errors.append(float(np.max(np.abs(exact - estimate))))
        self.audit = {'scope': 'Sparse interior interpolation diagnostic, not a uniform error bound or actuator-cap audit', 'support_base_residual_max': support_residual, 'interpolation_check_count': len(errors), 'interpolation_max_torque_error_nm': max(errors), 'interpolation_is_approximate': True}
        actuators = [model.actuator(name).id for name in SDK_JOINTS]
        limits = model.actuator_forcerange[actuators]
        excess = np.maximum(values - limits[:, 1], limits[:, 0] - values)
        self.audit['requested_feedforward_max_cap_excess_nm'] = float(max(0, excess.max()))
        self.audit['requested_feedforward_cap_excess_elements'] = int((excess > 0).sum())
        self.audit['feedforward_is_not_a_feasibility_certificate'] = True

@dataclass(kw_only=True)
class CoordinatedBattingActionCfg(ArticulatedBattingActionCfg):
    scale: float = 0.0
    arm_scale: float = 0.0
    yaw_balance_gain: float = 0.0
    pitch_balance_gain: float = 0.0

    def build(self, env):
        return CoordinatedBattingAction(self, env)

class CoordinatedBattingAction(ArticulatedBattingAction):
    feedforward_type = SwingFeedforward

    def __init__(self, cfg, env):
        if cfg.scale != 0 or cfg.arm_scale != 0:
            raise ValueError('coordinated batting has no independent joint residuals')
        super().__init__(cfg, env)
        model = env.get_playback_model()
        self.path = SwingPath(model, self.controller.poses, self.controller.times, smoothing=0.0003)
        self.feedforward = self.feedforward_type(model, self.path)
        self.body_dofs = model.jnt_dofadr[[model.joint(name).id for name in SDK_JOINTS]]
        self.actions = np.zeros((env.num_envs, 1), dtype=np.float32)
        self.phase = np.zeros(env.num_envs)
        self.rate = np.ones(env.num_envs)
        self.rate_dot = np.zeros(env.num_envs)

    @property
    def action_dim(self):
        return 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids], self.rate[ids], self.rate_dot[ids] = (0, 1, 0)

    def apply_actions(self):
        env, controller = (self._env, self.controller)
        action = np.clip(self.actions[:, 0], -1, 1)
        requested = 1 + np.where(action < 0, 0.5 * action, action)
        delta = np.clip(requested - self.rate, -1.5 * env.cfg.sim_dt, 1.5 * env.cfg.sim_dt)
        self.rate_dot[:] = delta / env.cfg.sim_dt
        tangent, velocity, _ = self.path.derivatives(self.phase, self.rate, self.rate_dot)
        self.reference[:] = self.path.initial[controller.q] + tangent[:, self.body_dofs]
        root_position = self.path.initial[:3] + tangent[:, :3]
        force = self.feedforward.interpolate(np.column_stack((self.phase, self.rate, self.rate_dot)))
        self.target[:] = self.reference + controller.velocity_gain * velocity[:, self.body_dofs] + force / controller.gain
        data = self._entity.data
        for row in range(env.num_envs):
            correction = ankle_balance(self.path.initial[3:7], data.root_link_quat_w[row], data.root_link_ang_vel_b[row], 4)
            correction += root_position_balance(self.path.initial[3:7], data.root_link_pos_w[row] - env.scene.env_origins[row] - root_position[row], data.root_link_lin_vel_w[row] - velocity[row, :3], 2)
            correction = np.clip(correction, -0.3, 0.3)
            self.target[row, [4, 10]] += correction[1]
            self.target[row, [5, 11]] += correction[0]
            if self.cfg.yaw_balance_gain or self.cfg.pitch_balance_gain:
                hip_correction = hip_orientation_correction(self.path.initial[3:7], data.root_link_quat_w[row], data.root_link_ang_vel_b[row], [0, self.cfg.pitch_balance_gain, self.cfg.yaw_balance_gain])
                self.target[row, [0, 6]] += hip_correction[1]
                self.target[row, [2, 8]] += hip_correction[2]
        self._set_body_targets()
        fingers = controller.fingers
        target = np.clip(controller.finger_target, fingers.limits[:, 0], fingers.limits[:, 1])
        torque = fingers.kp * (target - data.joint_pos[:, self.finger_ids]) - fingers.kd * data.joint_vel[:, self.finger_ids]
        self._entity.set_joint_effort_target(np.clip(torque, fingers.force_limits[:, 0], fingers.force_limits[:, 1]), joint_ids=self.finger_ids)
        self.phase[:] = np.minimum(self.path.times[-1], self.phase + (self.rate + 0.5 * delta) * env.cfg.sim_dt)
        self.rate += delta
        self.substep += 1

    def _set_body_targets(self):
        self._entity.set_joint_position_target(np.clip(self.target, self.controller.limits[:, 0], self.controller.limits[:, 1]), joint_ids=self.body_ids)

class CoordinatedObservation(ArticulatedObservation):

    def __call__(self, env):
        action = env.action_manager.get_term('batting')
        return np.column_stack((super().__call__(env), action.phase / action.path.times[-1], action.rate / 2))
