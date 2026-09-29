"""CPU residual control over the native AMP/GR00T bowling rollout."""
import json
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from .amp_bowling_probe import delivery_phase, rollout_case, underarm_release_ready, overarm_release_ready
from .amp_running_probe import JOINTS, load_actor
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import RETURN_Y, TARGET_POPPING_X
CASES = tuple(((hand, dt) for hand in ('right', 'left') for dt in (6.25e-05, 3.125e-05)))
ROLLOUT_OPTIONS = ('gather_deceleration', 'groot_gather', 'arm_clearance', 'native_target_limits', 'delivery_rate', 'minimum_release_speed', 'native_delivery_impedance', 'arm_reference_directory', 'reference_start', 'arm_inertia_compensation', 'start_x', 'gather_start', 'deceleration_start')
FLIGHT_FAILURES = {'first_bounce_outside_delivery_zone', 'not_exactly_one_bounce_before_target', 'target_corridor_missed'}
FLIGHT_REWARD_COLUMNS = ('time_s', 'total_reward', 'flight_credit_change', 'flight_credit')

def first_bounce_credit(release, bounce, failures):
    """Bounded measured-flight shaping, never a substitute for qualification."""
    if release is None or bounce is None or failures:
        return 0.0
    x, y, _ = bounce['position']
    start = release['position'][0]
    if start >= TARGET_POPPING_X or x >= TARGET_POPPING_X:
        return 0.0
    progress = np.clip((x - start) / (TARGET_POPPING_X - start), 0, 1)
    alignment = np.clip(1 - abs(y) / RETURN_Y, 0, 1)
    return float(5 * progress * alignment)

def bowling_reward(speed_gain, limit_excess, new_failures, action, result=None, *, release_curriculum=False):
    """Dense shaping is not delivery qualification; only full gates earn success."""
    reward = 0.1 * speed_gain - 0.001 * np.mean(action ** 2)
    reward -= min(100 * limit_excess, 1) + len(new_failures)
    if result is not None:
        terminal = 50 if result['passed'] else -5
        if release_curriculum and (not result['passed']):
            if result['release'] is None:
                terminal = -10
            elif not set(result['failures']) - FLIGHT_FAILURES:
                terminal = 5
        reward += terminal
    return float(reward)

def support_observation(monitor, time):
    landings = []
    for side in (monitor.events.front, monitor.events.hand):
        history = monitor.events.landings[side]
        if history:
            latest = history[-1]
            landings.extend(np.r_[1, (time - latest['time']) / 8, np.asarray(latest['bounds'])[:, :2].ravel()])
        else:
            landings.extend(np.zeros(6))
    return np.r_[monitor.swing_local / 8, monitor.swing_rate, monitor.swing_started, monitor.foot_loads / 100, landings]

class AmpBowlingEnv(gym.Env):
    """Privileged state, holder force and simulated foot/hand contact aggregates."""
    metadata = {'render_modes': []}
    device = 'cpu'

    def __init__(self, upstream, unilab, parent, *, release_curriculum=False, learn_release=False, allow_upward_release=False, flight_curriculum=False, post_saturation_residual=False, bowling_arm_actions=False, physics_backend='mujoco', rollout_options=None, scene_files=None):
        if physics_backend not in ('mujoco', 'mjbatch'):
            raise ValueError('Expected mujoco or mjbatch physics backend')
        self.physics_backend = physics_backend
        self.release_curriculum = release_curriculum
        self.learn_release = learn_release
        self.allow_upward_release = allow_upward_release
        self.flight_curriculum = flight_curriculum
        self.post_saturation_residual = post_saturation_residual
        self.bowling_arm_actions = bowling_arm_actions
        self.parent, self.unilab = (Path(parent), Path(unilab))
        summary = json.loads((self.parent / 'summary.json').read_text()) if rollout_options is None else rollout_options
        self.options = {key: summary[key] for key in ROLLOUT_OPTIONS}
        self.options['delivery_style'] = summary.get('delivery_style', 'overarm')
        if allow_upward_release and (not learn_release or self.options['delivery_style'] != 'overarm'):
            raise ValueError('Upward release requires learned overarm release')
        self.options['underarm_minimum_loft_deg'] = summary.get('underarm_minimum_loft_deg', 0.0)
        self.options['support_triggered_swing'] = summary.get('support_triggered_swing', False)
        self.options['synchronize_reference_arms'] = summary.get('synchronize_reference_arms', False)
        self.options['release_arm_hold'] = summary.get('release_arm_hold', False)
        self.options['native_bowling_elbow_impedance'] = summary.get('native_bowling_elbow_impedance', False)
        self.options['compensate_torso_heading'] = summary.get('compensate_torso_heading', False)
        self.options['bowling_torso_pitch'] = summary.get('bowling_torso_pitch', 0.0)
        self.options['carry_roll'] = summary.get('carry_roll')
        self.options['carry_elbow'] = summary.get('carry_elbow')
        self.options['steady_carry'] = summary.get('steady_carry', False)
        self.options['preserve_lane_direction'] = summary.get('preserve_lane_direction', False)
        self.options['lane_damping'] = summary.get('lane_damping', 0.0)
        self.options['heading_damping'] = summary.get('heading_damping', 0.0)
        self.options['lane_offset'] = summary.get('lane_offset', 0.5)
        self.options['approach_gate_x'] = summary.get('approach_gate_x')
        self.observe_sequence = self.options['approach_gate_x'] is not None
        for key in ('groot_gather', 'arm_reference_directory'):
            if self.options[key] is not None:
                self.options[key] = Path(self.options[key])
        self.actor = load_actor(Path(upstream))
        self.scene_files = scene_files if scene_files is not None else {hand: self.parent / f'{hand}.xml' for hand in ('right', 'left')}
        self.templates = {hand: mujoco.MjModel.from_xml_path(str(self.scene_files[hand])) for hand in ('right', 'left')}
        template = self.templates['right']
        action_width = (7 if bowling_arm_actions else 29) + int(learn_release)
        self.action_space = gym.spaces.Box(-1, 1, (action_width,), dtype=np.float32)
        width = template.nq + template.nv + template.nu + 11 + action_width + 4
        width += 2 if self.observe_sequence else 0
        width += 17 if self.options['support_triggered_swing'] else 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.rollout = None
        self.result, self.records = (None, None)
        self.case_queue = []
        self.episodes = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        if seed is not None:
            self.case_queue.clear()
        if options is not None:
            case = (options['hand'], options['timestep'])
            if case not in CASES:
                raise ValueError('Expected a member of the complete hand/resolution pool')
        else:
            if not self.case_queue:
                self.case_queue = self.np_random.permutation(len(CASES)).tolist()
            case = CASES[self.case_queue.pop()]
        self.hand, self.dt = case
        self.rollout = rollout_case(self.actor, self.unilab, self.parent, *case, retain=False, learn_release=self.learn_release, post_saturation_residual=self.post_saturation_residual, allow_upward_release=self.allow_upward_release, physics_backend=self.physics_backend, scene_file=self.scene_files[self.hand], **self.options)
        self.model, self.data, self.monitor = next(self.rollout)
        self.actuators = np.array([self.model.actuator(name + '_joint').id for name in JOINTS])
        self.contact_groups = list(self.monitor.feet.values()) + [np.array([i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith(side + '_hand')]) for side in ('left', 'right')]
        self.action = np.zeros(self.action_space.shape)
        self.peak_speed = 0.0
        self.previous_failures = set()
        self.steps, self.episode_return = (0, 0.0)
        self.flight_credit = 0.0
        self.reward_rows = []
        self.result, self.records = (None, None)
        return (self.observation(), {'hand': self.hand, 'timestep': self.dt})

    def observation(self):
        contacts = np.zeros((4, 2))
        force = np.empty(6)
        for index, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, index, force)
            for group, geoms in enumerate(self.contact_groups):
                if any((geom in geoms for geom in contact.geom)):
                    contacts[group] += [max(force[0], 0), np.linalg.norm(force[1:3])]
        adr = self.monitor.force_adr
        sequence = [self.monitor.sequence_time / 8, self.monitor.sequence_rate] if self.observe_sequence else []
        support = support_observation(self.monitor, self.data.time) if self.options['support_triggered_swing'] else []
        return np.r_[self.data.qpos, self.data.qvel * 0.05, self.data.actuator_force[self.actuators] / self.model.actuator_forcerange[self.actuators, 1], contacts.ravel() / 100, self.data.sensordata[adr:adr + 3] / 100, self.action, self.data.time / 8, self.hand == 'left', self.monitor.events.release_record is not None, self.peak_speed / 20, sequence, support].astype(np.float32)

    def eligible_speed(self):
        phase, _ = delivery_phase(self.monitor.sequence_time + 2.8 - self.options['reference_start'], self.options['delivery_rate'])
        if self.options['support_triggered_swing']:
            if not self.monitor.swing_started:
                return 0.0
            phase = 2.8 + self.monitor.swing_local
        if not 4 <= phase <= 5.5 or self.monitor.events.release_record is not None:
            return 0.0
        ball = self.model.joint('ball_free')
        q, v = (int(ball.qposadr[0]), int(ball.dofadr[0]))
        shoulder, elbow, _ = self.data.xpos[self.monitor.arm]
        ready = overarm_release_ready(self.data.qpos[None, q:q + 3], self.data.qvel[None, v:v + 3], shoulder[None], elbow[None], allow_upward=self.allow_upward_release)[0]
        if self.options['delivery_style'] == 'underarm':
            ready = underarm_release_ready(self.data.qpos[None, q:q + 3], self.data.qvel[None, v:v + 3], shoulder[None], self.options['underarm_minimum_loft_deg'])[0]
        return float(np.clip(self.data.qvel[v], 0, 20)) if ready else 0.0

    def expand_action(self, action):
        if not self.bowling_arm_actions:
            return action
        indices = [i for i, name in enumerate(JOINTS) if name.startswith(self.hand + '_') and any((part in name for part in ('shoulder', 'elbow', 'wrist')))]
        expanded = np.zeros(29 + int(self.learn_release), dtype=action.dtype)
        expanded[indices] = action[:7]
        if self.learn_release:
            expanded[-1] = action[-1]
        return expanded

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f'Expected {self.action_space.shape[0]} finite residual actions')
        if self.rollout is None:
            raise RuntimeError('Reset before stepping a finished environment')
        self.action = np.clip(action, -1, 1)
        physical_action = self.expand_action(self.action)
        try:
            self.model, self.data, self.monitor = self.rollout.send(physical_action)
            terminated = False
        except StopIteration as finished:
            self.result, self.records = finished.value
            self.rollout = None
            terminated = True
        self.steps += 1
        speed = max(self.peak_speed, self.eligible_speed())
        events = self.monitor.events
        excess = np.maximum(self.model.jnt_range[self.monitor.joints, 0] - self.data.qpos[self.monitor.q], self.data.qpos[self.monitor.q] - self.model.jnt_range[self.monitor.joints, 1]).max(initial=0)
        reward = bowling_reward(speed - self.peak_speed, excess, events.failures - self.previous_failures, physical_action, self.result, release_curriculum=self.release_curriculum)
        credit = 0.0
        if self.flight_curriculum:
            failures = events.failures.copy()
            if self.result is not None:
                failures.update(set(self.result['failures']) - FLIGHT_FAILURES)
            if excess > 1e-06:
                failures.add('joint_limit')
            credit = first_bounce_credit(events.release_record, events.first_bounce, failures)
            reward += credit - self.flight_credit
            self.reward_rows.append([float(self.data.time), reward, credit - self.flight_credit, credit])
            self.flight_credit = credit
        self.peak_speed = speed
        self.previous_failures = events.failures.copy()
        self.episode_return += reward
        if terminated:
            self.episodes.append({'return': self.episode_return, **self.result})
            if self.flight_curriculum:
                self.records['reward_rows'] = np.asarray(self.reward_rows)
                self.episodes[-1]['first_bounce_credit'] = self.flight_credit
        return (self.observation(), reward, terminated, False, {'result': self.result} if terminated else {})

    def close(self):
        if self.rollout is not None:
            self.rollout.close()
            self.rollout = None
