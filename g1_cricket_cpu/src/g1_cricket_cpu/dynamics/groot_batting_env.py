"""CPU Gymnasium bridge to the audited GR00T/reference batting rollout."""
import copy
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from g1_cricket_cpu.dynamics.groot_articulated_probe import GrootSwingController, load_policy
from g1_cricket_cpu.dynamics.groot_incoming_probe import IncomingAudit, reset_delivery, swing_reference
from g1_cricket_cpu.dynamics.impact_schedule import ImpactSchedule
from g1_cricket_cpu.dynamics.mjbatch_stepper import MjBatchStepper
from g1_cricket_cpu.dynamics.groot_swing_timing import arrival_timing
from g1_cricket_cpu.scripts.evaluate_g1_articulated_swing import rollout_case
from g1_cricket_cpu.physics.articulated_hands import hand_contact_state
from g1_cricket_cpu.physics.articulated_learning import ARTICULATED_DELIVERY_POOL
RESIDUAL_SCALE = 0.15
ACTION_DIM = 17
WHOLE_BODY_ACTION_DIM = 29
LEG_RESIDUAL_SCALE = 0.03
EPISODE_STEPS = 400

def recovery_residual_gain(time_s):
    if time_s <= 3.0:
        return 1.0
    if time_s >= 4.0:
        return 0.0
    phase = time_s - 3.0
    return 1.0 - phase * phase * (3.0 - 2.0 * phase)

class ResidualGrootController(GrootSwingController):

    def __init__(self, *args, recovery_fade=False, whole_body=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.whole_body = whole_body
        self.residual = np.zeros(WHOLE_BODY_ACTION_DIM if whole_body else ACTION_DIM)
        self.recovery_fade = recovery_fade

    def apply(self, data):
        gain = recovery_residual_gain(data.time) if self.recovery_fade else 1.0
        if self.whole_body:
            self.leg_target_offset[:] = LEG_RESIDUAL_SCALE * self.residual[:12] * gain
        super().apply(data)
        ids = self.actuators[12:29]
        upper = self.residual[12:] if self.whole_body else self.residual
        data.ctrl[ids] = np.clip(data.ctrl[ids] + RESIDUAL_SCALE * upper * gain, self.limits[12:29, 0], self.limits[12:29, 1])

def batting_reward(distance, metrics, action, result=None):
    """Training shaping is separate from the unchanged hit/physical acceptance gates."""
    reward = 0.01 * np.exp(-distance ** 2 / 0.15)
    reward -= 0.001 * np.mean(action ** 2)
    reward -= 0.01 * min(metrics['max_relative_grip_position_m'] / 0.02, 5) ** 2
    if result is not None:
        qualified = result['physical']['passed'] and result['valid_one_pitch_hit']
        if qualified:
            forward_speed = max(0, result['separation_velocity_m_s'][0])
            reward += 10 + min(forward_speed, 40) + 10 * result['boundary_runs']
        else:
            reward -= 10
    return float(reward)

class GrootBattingEnv(gym.Env):
    """Privileged state and simulated tactile observations; no vision-policy claim."""
    metadata = {'render_modes': []}
    device = 'cpu'

    def __init__(self, upstream, checkout, scene, *, recovery_fade=False, whole_body=False, reference_files=None, condition_timing=False, swing_power=0.0, physics_backend='mujoco'):
        if physics_backend not in ('mujoco', 'mjbatch'):
            raise ValueError('Expected mujoco or mjbatch physics backend')
        self.physics_backend = physics_backend
        self.upstream, self.checkout = (Path(upstream), Path(checkout))
        self.recovery_fade = recovery_fade
        self.whole_body = whole_body
        self.action_dim = WHOLE_BODY_ACTION_DIM if whole_body else ACTION_DIM
        self.reference_files = dict(zip(('right', 'left'), reference_files, strict=True)) if reference_files is not None else {}
        self.condition_timing = condition_timing
        self.swing_power = swing_power
        self.template = mujoco.MjModel.from_xml_path(str(scene))
        self.template.opt.timestep = 0.000125
        self.template.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        self.template.pair('cricket_blade_impact_v1').solref[:] = [0.001, 0.285]
        self.references = {hand: swing_reference(self.checkout, hand, 0.12, reference_file=self.reference_files.get(hand), swing_power=swing_power) for hand in ('right', 'left')}
        self.action_space = gym.spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)
        width = self.template.nq + self.template.nv + self.template.nu + 18 + self.action_dim + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.rollout = None
        self.case_queue = []
        self.episodes = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if seed is not None:
            self.case_queue.clear()
        if options is not None:
            hand, feed = (options['hand'], options['feed_index'])
            if hand not in self.references or feed not in range(len(ARTICULATED_DELIVERY_POOL)):
                raise ValueError('Expected a hand and a member of the complete delivery pool')
        else:
            if not self.case_queue:
                self.case_queue = self.np_random.permutation(2 * len(ARTICULATED_DELIVERY_POOL)).tolist()
            case = self.case_queue.pop()
            hand = 'right' if case < len(ARTICULATED_DELIVERY_POOL) else 'left'
            feed = case % len(ARTICULATED_DELIVERY_POOL)
        self.hand, self.feed = (hand, feed)
        self.model = copy.copy(self.template)
        poses, times = self.references[hand]
        if self.condition_timing:
            preview = mujoco.MjData(self.model)
            preview.qpos[:] = poses[0]
            reset_delivery(self.model, preview, hand, feed)
            timing = arrival_timing(self.model, preview, 0.12)
            poses, times = swing_reference(self.checkout, hand, timing['delay_s'], reference_file=self.reference_files.get(hand), swing_power=self.swing_power)
        self.controller = ResidualGrootController(self.model, poses, times, load_policy(self.upstream), reference_waist=True, root_feedback_gain=4, finger_preload=0.6, finger_stiffness=3, finger_damping=0.3, native_finger_servo=True, time_based_policy=True, recovery_fade=self.recovery_fade, whole_body=self.whole_body)
        self.audit = IncomingAudit(self.model, time_based_sampling=True)
        self.schedule = ImpactSchedule(coarse_integrator='implicitfast')
        self.rollout = rollout_case(self.model, poses, times, inertial=False, controller=self.controller, reset=lambda data: reset_delivery(self.model, data, hand, feed), observe=self.audit, step_schedule=self.schedule, physics_step=MjBatchStepper(self.model) if self.physics_backend == 'mjbatch' else None)
        self.data, self.metrics = next(self.rollout)
        self.steps, self.episode_return = (0, 0.0)
        self.actions = []
        self.result = None
        return (self.observation(), {'hand': hand, 'feed_index': feed})

    def observation(self):
        tactile = np.zeros((2, 3, 3))
        for contact in hand_contact_state(self.model, self.data):
            name = contact['geom']
            for side, hand in enumerate(('left', 'right')):
                for finger, token in enumerate(('thumb', 'index', 'middle')):
                    if name.startswith(hand + '_hand_' + token):
                        tactile[side, finger, 0] += contact['normal_force_n'] / 100
                        tactile[side, finger, 1] += np.linalg.norm(contact['wrench_contact_frame'][1:3]) / 100
                        tactile[side, finger, 2] = max(tactile[side, finger, 2], contact['tangential_slip_m_s'])
        return np.r_[self.data.qpos, self.data.qvel * 0.05, self.data.actuator_force / 100, tactile.ravel(), self.controller.residual, self.data.time / 8, float(self.hand == 'left')].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            count = 'twenty-nine' if self.whole_body else 'seventeen'
            raise ValueError(f'Expected {count} finite residual actions')
        if self.rollout is None or self.steps >= EPISODE_STEPS:
            raise RuntimeError('Reset before stepping a finished environment')
        self.controller.residual[:] = np.clip(action, -1, 1)
        self.actions.append(self.controller.residual.copy())
        self.data, self.metrics = next(self.rollout)
        self.steps += 1
        terminated = self.steps == EPISODE_STEPS
        info = {}
        if terminated:
            try:
                next(self.rollout)
            except StopIteration as finished:
                self.physical, self.states, self.controls, self.tactile = finished.value
            self.result = {'physical': self.physical, **self.audit.result()}
            self.result['qualified_hit'] = self.physical['passed'] and self.result['valid_one_pitch_hit']
            info['result'] = self.result
        distance = np.linalg.norm(self.data.geom('ball_geom').xpos - self.data.geom('bat_blade').xpos)
        reward = batting_reward(distance, self.metrics, self.controller.residual, self.result)
        self.episode_return += reward
        if terminated:
            self.episodes.append({'hand': self.hand, 'feed_index': self.feed, 'return': self.episode_return, **self.result})
        return (self.observation(), reward, terminated, False, info)

    def close(self):
        if self.rollout is not None:
            self.rollout.close()
            self.rollout = None
