"""Full bilateral delivery screen: published AMP running plus reference arms."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import torch
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import DeliveryEvents, RETURN_Y, arm_geometry, capsule_bounds
from g1_cricket_cpu.physics.pitch_contact import G1CricketDeliveryPitchV2Cfg
from g1_cricket_cpu.physics.moving_delivery import REFERENCE_START, arm_weight, forward_release_ready
from g1_cricket_cpu.physics.running import END_TIME, GATHER_TIME
from g1_cricket_cpu.physics.prior import SDK_JOINTS
from .amp_running_probe import JOINTS, ObservationHistory, command_at, default_angles, load_actor, local_rotation, policy_gains
from .twist2_cpu_probe import fingerprint
from .bowling_sweep_reference import transition
from .bowling_support_clock import SupportSwingClock, validate_cocked_pause
from .mjbatch_stepper import MjBatchStepper, backend_inputs
from .wicket_contact import VERSION as WICKET_CONTACT_VERSION, add_wicket_pairs
from .bowling_launch_control import LaunchController, launch_ready
from .maximum_effort import maximum_effort

def build_scene(unilab, destination, hand, *, wicket_contact=False):
    source = unilab / 'src/unilab/assets/robots/g1/g1.xml'
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(source, destination)
    tree = ET.parse(destination)
    sensors = tree.getroot().find('sensor')
    for sensor in list(sensors):
        if sensor.get('name') not in {'holder_force', 'holder_quat'}:
            sensors.remove(sensor)
    if wicket_contact:
        add_wicket_pairs(tree.getroot())
    tree.write(destination, encoding='unicode')

def advance_one_bounce_frontier(previous, events, time, position):
    if events.release_record is None or events.bounces_before_target != 1:
        return previous
    if events.crossing is not None:
        return events.crossing
    return dict(time=float(time), position=position.tolist())

class DeliveryMonitor:

    def __init__(self, model, hand):
        self.model = model
        self.events = DeliveryEvents(hand)
        self.one_bounce_frontier = None
        self.ball = model.geom('ball_geom').id
        self.pitch = model.geom('pitch').id
        self.arm = [model.body(f'{hand}_{n}_link').id for n in ('shoulder_roll', 'elbow', 'wrist_roll')]
        self.feet = {side: np.flatnonzero((model.geom_bodyid == model.body(f'{side}_ankle_roll_link').id) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))) for side in ('left', 'right')}
        self.joints = model.actuator_trnid[:, 0]
        self.q = model.jnt_qposadr[self.joints]
        self.force_adr = model.sensor('holder_force').adr[0]
        self.site = model.site('holder_site').id
        self.robot_contacts = {}
        self.joint_excess = np.zeros(len(self.joints))
        self.foot_loads = np.zeros(2)

    def observe(self, data):
        m, e = (self.model, self.events)
        upper, angle = arm_geometry(*data.xpos[self.arm])
        e.observe_arm(float(data.time), upper, angle)
        loads = np.zeros(2)
        ball_contacts = []
        force = np.empty(6)
        for i, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            g1, g2 = map(int, contact.geom)
            mujoco.mj_contactForce(m, data, i, force)
            if self.ball in (g1, g2):
                other = g2 if g1 == self.ball else g1
                ball_contacts.append((m.geom(other).name, float(contact.dist), float(np.linalg.norm(force[:3]))))
            elif self.pitch in (g1, g2):
                other = g2 if g1 == self.pitch else g1
                found = False
                for index, ids in enumerate(self.feet.values()):
                    if other in ids:
                        loads[index] += max(force[0], 0)
                        found = True
                if not found:
                    e.failures.add('nonfoot_ground_contact')
            else:
                e.failures.add('robot_self_or_wicket_contact')
                key = ':'.join(sorted((m.geom(g1).name, m.geom(g2).name)))
                record = self.robot_contacts.setdefault(key, dict(samples=0, first_time_s=float(data.time), max_penetration_m=0.0, peak_force_n=0.0))
                record['samples'] += 1
                record['max_penetration_m'] = max(record['max_penetration_m'], float(-contact.dist))
                record['peak_force_n'] = max(record['peak_force_n'], float(np.linalg.norm(force[:3])))
        for index, (side, ids) in enumerate(self.feet.items()):
            bounds = capsule_bounds(data.geom_xpos[ids], data.geom_xmat[ids].reshape(-1, 3, 3), m.geom_size[ids], m.geom_type[ids])
            e.observe_support(float(data.time), side, loads[index] > 1, bounds)
        self.foot_loads = loads
        e.observe_ball(float(data.time), data.geom_xpos[self.ball], ball_contacts)
        self.one_bounce_frontier = advance_one_bounce_frontier(self.one_bounce_frontier, e, data.time, data.geom_xpos[self.ball])
        e.height = min(e.height, float(data.qpos[2]))
        e.up = min(e.up, float(1 - 2 * np.square(data.qpos[4:6]).sum()))
        excess = np.maximum(m.jnt_range[self.joints, 0] - data.qpos[self.q], data.qpos[self.q] - m.jnt_range[self.joints, 1]).max(initial=0)
        e.limit_excess = max(e.limit_excess, float(excess))
        self.joint_excess = np.maximum(self.joint_excess, np.maximum(m.jnt_range[self.joints, 0] - data.qpos[self.q], data.qpos[self.q] - m.jnt_range[self.joints, 1]))
        fraction = float((np.abs(data.actuator_force) / m.actuator_forcerange[:, 1]).max())
        e.force_fraction = max(e.force_fraction, fraction)
        force = data.sensordata[self.force_adr:self.force_adr + 3]
        e.peak_holder_force = max(e.peak_holder_force, float(np.linalg.norm(force)))
        e.holder_impulse += data.site_xmat[self.site].reshape(3, 3) @ force * m.opt.timestep
        support = max(np.abs(data.qfrc_applied).max(), np.abs(data.xfrc_applied).max())
        return np.r_[data.time, data.qpos[:3], loads, excess, fraction, support, force]

def delivery_command(time, yaw, lane_error, gather_deceleration=False, deceleration_start=3.0, preserve_lane_direction=False, lateral_velocity=0.0, lane_damping=0.0, yaw_velocity=0.0, heading_damping=0.0):
    speed = 2.5 * (np.clip(deceleration_start + 1 - time, 0, 1) if gather_deceleration else 1)
    command = command_at(time, speed, yaw, 2.0, lane_error, preserve_direction=preserve_lane_direction, lateral_velocity=lateral_velocity, lane_damping=lane_damping)
    damping = heading_damping * gather_blend(time, deceleration_start)
    command[2] = np.clip(-2 * yaw - damping * yaw_velocity, -1, 1)
    return command

def native_target_bounds(target, bounds):
    return np.clip(target, bounds[:, 0] + 0.03, bounds[:, 1] - 0.03)

def gather_blend(time, start=3.0):
    fraction = np.clip(time - start, 0, 1)
    return fraction * fraction * (3 - 2 * fraction)

class ApproachClock:

    def __init__(self, gather_start, gate_x=None):
        self.gather_start, self.gate_x = (gather_start, gate_x)
        self.opened = gate_x is None
        self.delay = 0.0

    def advance(self, time, root_x):
        if not self.opened:
            self.delay = max(time - self.gather_start, 0.0)
            self.opened = root_x >= self.gate_x
        running = self.opened or time < self.gather_start
        return (time - self.delay, float(running))

def delivery_phase(time, rate=1.0):
    elapsed = max(time - REFERENCE_START - GATHER_TIME, 0.0)
    ramp = 0.2
    fraction = min(elapsed / ramp, 1.0)
    advance = ramp * (fraction ** 3 - 0.5 * fraction ** 4) + max(elapsed - ramp, 0)
    phase = time + (rate - 1) * advance
    speed = 1 + (rate - 1) * fraction ** 2 * (3 - 2 * fraction)
    return (phase, speed)

def arm_inertia_request(model, data, dofs, acceleration):
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    return mass[np.ix_(dofs, dofs)] @ acceleration + data.qfrc_bias[dofs]

def residual_torque(action, caps):
    fractions = np.array([0.1 if any((part in name for part in ('hip', 'knee', 'ankle'))) else 0.3 for name in JOINTS])
    return action * fractions * caps

def bounded_torque(request, correction, caps, *, post_saturation_residual=False):
    if post_saturation_residual:
        request = np.clip(request, -caps, caps)
    return np.clip(request + correction, -caps, caps)

def torso_heading_command(yaw, blend):
    return np.array([0.0, 0.0, blend * np.clip(-yaw, -0.6, 0.6)])

def bowling_pitch_command(local, amplitude):
    return amplitude * transition(local, 0.8, 1.5) * (1 - transition(local, 2.1, 2.7))

def carry_roll_targets(target, velocity, minimum):
    target, velocity = (target.copy(), velocity.copy())
    for side, sign in (('left', 1), ('right', -1)):
        index = JOINTS.index(f'{side}_shoulder_roll')
        if sign * target[index] < minimum:
            target[index] = sign * minimum
            velocity[index] = 0
    return (target, velocity)

def carry_elbow_targets(target, minimum):
    target = target.copy()
    for side in ('left', 'right'):
        index = JOINTS.index(f'{side}_elbow')
        target[index] = max(target[index], minimum)
    return target

def steady_carry_targets(target):
    target = target.copy()
    neutral = default_angles()
    for i, name in enumerate(JOINTS):
        if any((part in name for part in ('shoulder', 'elbow', 'wrist'))):
            target[i] = neutral[i]
    return target

def release_hold_targets(elapsed, pose, target, velocity):
    """Brake at release for 0.2 s, then smoothly rejoin the arm reference."""
    phase = np.clip((elapsed - 0.2) / 0.2, 0, 1)
    blend = phase * phase * (3 - 2 * phase)
    rate = 6 * phase * (1 - phase) / 0.2
    return (pose + blend * (target - pose), blend * velocity + rate * (target - pose))

def overarm_release_ready(position, velocity, shoulder, elbow, *, allow_upward=False):
    if not allow_upward:
        return forward_release_ready(position, velocity, shoulder, elbow)
    return (position[..., 2] > shoulder[..., 2] + 0.12) & (elbow[..., 2] > shoulder[..., 2]) & (velocity[..., 0] > 1) & (np.abs(velocity[..., 1]) < 2)

def underarm_release_ready(position, velocity, shoulder, minimum_loft_deg=0.0):
    return (position[..., 2] < shoulder[..., 2] - 0.1) & (velocity[..., 0] > 1) & (velocity[..., 2] > np.tan(np.deg2rad(minimum_loft_deg)) * velocity[..., 0]) & (np.abs(velocity[..., 1]) < 2)

def rollout_case(actor, unilab, output, hand, dt, *, gather_deceleration=False, groot_gather=None, arm_clearance=False, native_target_limits=False, delivery_rate=1.0, minimum_release_speed=1.0, native_delivery_impedance=False, native_bowling_elbow_impedance=False, compensate_torso_heading=False, bowling_torso_pitch=0.0, carry_roll=None, carry_elbow=None, steady_carry=False, preserve_lane_direction=False, lane_damping=0.0, heading_damping=0.0, lane_offset=0.5, support_triggered_swing=False, synchronize_reference_arms=False, approach_gate_x=None, arm_reference_directory=None, reference_start=REFERENCE_START, arm_inertia_compensation=False, release_arm_hold=False, learn_release=False, allow_upward_release=False, maximum_arm_effort=False, post_saturation_residual=False, physics_backend='mujoco', delivery_style='overarm', underarm_minimum_loft_deg=0.0, start_x=-10.5, gather_start=3.0, deceleration_start=3.0, retain=True, scene_file=None, model_launch=False, model_launch_lookahead=False, shoulder_plan=None):
    if allow_upward_release and (not learn_release or delivery_style != 'overarm'):
        raise ValueError('Upward release requires learned overarm release')
    if model_launch_lookahead and (not model_launch):
        raise ValueError('Launch lookahead requires model launch')
    if model_launch and (delivery_style != 'overarm' or not support_triggered_swing or learn_release):
        raise ValueError('Model launch requires support-triggered overarm without learned release')
    if shoulder_plan is not None:
        if delivery_style != 'overarm' or not support_triggered_swing or model_launch or maximum_arm_effort or learn_release:
            raise ValueError('Shoulder plan requires support-triggered overarm without another arm/release override')
        from .bowling_launch_shooting import KNOTS, shoulder_schedule, shoulder_torque
        plan_parameters = np.asarray(shoulder_plan['parameters'], dtype=float)
        plan_mode = shoulder_plan['control_mode']
        plan_axis = shoulder_plan.get('shoulder_axis', 'all')
        shoulder_schedule(KNOTS[1], plan_parameters, hand)
        shoulder_torque(np.zeros(3), np.zeros(3), np.ones(3), 0, plan_mode, plan_axis)
    scene = output / f'{hand}.xml' if scene_file is None else scene_file
    if scene_file is None and (not scene.exists()):
        build_scene(unilab, scene, hand)
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)
    joints = np.array([model.joint(n + '_joint').id for n in JOINTS])
    q, v = (model.jnt_qposadr[joints], model.jnt_dofadr[joints])
    ids = np.array([model.actuator(n + '_joint').id for n in JOINTS])
    kp, kd = policy_gains()
    native_kp, native_kd = (model.actuator_gainprm[ids, 0], -model.actuator_biasprm[ids, 2])
    caps = model.actuator_forcerange[ids, 1]
    plan_selected = np.array([i for i, name in enumerate(JOINTS) if name.startswith(hand + '_shoulder_')])
    plan_rows, plan_native_rows = ([], [])
    launch_controller = LaunchController(model, hand) if model_launch else None
    launch_torque = None
    launch_rows = []
    maximum_effort_rows = []
    launch_warmstart_rows = []
    lookahead = None
    if model_launch_lookahead:
        from .bowling_launch_lookahead import LaunchLookahead
        lookahead = LaunchLookahead(model, hand)
    launch_fraction = 1.0
    launch_selection_rows, launch_lookahead_rows, launch_prediction_rows = ([], [], [])
    sdk = np.array([JOINTS.index(name.removesuffix('_joint')) for name in SDK_JOINTS])
    lower = sdk[:15]
    balance = None
    if groot_gather is not None:
        from .groot_articulated_probe import load_policy
        balance = load_policy(groot_gather)
    balance_rows = []
    torso_command_rows = []
    effective_kp, effective_kd = (kp.copy(), kd.copy())
    arms = np.array([i for i, n in enumerate(JOINTS) if any((p in n for p in ('shoulder', 'elbow', 'wrist')))])
    bowling = np.array([i for i in arms if JOINTS[i].startswith(hand + '_')])
    impedance_joints = arms if native_delivery_impedance else np.array([JOINTS.index(f'{hand}_elbow')])
    bowling_reference = np.array([list(arms).index(i) for i in bowling])
    clearance = None
    clearance_rows = []
    if arm_clearance:
        from .arm_clearance import ArmClearance
        clearance = ArmClearance(model, joints[arms])
    reference_directory = arm_reference_directory or unilab / 'g1_cricket_results/running_fore_aft_support_v1'
    reference_path = reference_directory / f'{hand}_dense_reference.npz'
    with np.load(reference_path) as reference:
        arm_reference = PchipInterpolator(reference['times'], reference['qpos'][:, q[arms]], axis=0)
        if support_triggered_swing:
            validate_cocked_pause(PchipInterpolator(reference['times'], reference['qpos'][:, q[bowling]], axis=0))
    lane = lane_offset if hand == 'right' else -lane_offset
    data.qpos[:7] = [start_x, lane, 0.8, 1, 0, 0, 0]
    data.qpos[q] = default_angles()
    if carry_elbow is not None:
        data.qpos[q] = carry_elbow_targets(data.qpos[q], carry_elbow)
    if carry_roll is not None:
        data.qpos[q], _ = carry_roll_targets(data.qpos[q], np.zeros(29), carry_roll)
    mujoco.mj_kinematics(model, data)
    ball = model.joint('ball_free')
    bq, bv = (int(ball.qposadr[0]), int(ball.dofadr[0]))
    wrist = model.body(f'{hand}_wrist_yaw_link').id
    offset = np.array([0.15, 0.06 if hand == 'left' else -0.06, 0])
    data.qpos[bq:bq + 3] = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset
    data.qpos[bq + 3:bq + 7] = data.xquat[wrist]
    mujoco.mj_forward(model, data)
    monitor = DeliveryMonitor(model, hand)
    clock = ApproachClock(gather_start, approach_gate_x)
    swing_clock = SupportSwingClock()
    swing_rows = []
    sequence_rows = []
    history = ObservationHistory()
    previous, target, velocity = (np.zeros(29), default_angles(), np.zeros(29))
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    states, observations, actions, targets, metrics, released = ([], [], [], [], [], [])
    impedance_rows = []
    feedforward_rows = []
    residual_rows = []
    release_action_rows = []
    feedforward = np.zeros(29)
    release_pose = None
    period = round(0.02 / dt)
    terminal = 'complete'
    batch_step = MjBatchStepper(model) if physics_backend == 'mjbatch' else None
    for step in range(round(8 / dt)):
        if step % period == 0:
            time = float(data.time)
            monitor.sequence_time, monitor.sequence_rate = clock.advance(time, data.qpos[0])
            if support_triggered_swing:
                front_load = monitor.foot_loads[0 if hand == 'right' else 1]
                swing_local, swing_speed = swing_clock.advance(time, monitor.sequence_time - reference_start, monitor.events, front_load)
                swing_rows.append([swing_local, swing_speed])
                monitor.swing_local, monitor.swing_rate = (swing_local, swing_speed)
                monitor.swing_started = swing_clock.started_at is not None
            residual = (yield (model, data, monitor))
            if residual is None:
                residual = np.zeros(29 + int(learn_release))
            residual = np.asarray(residual, dtype=float)
            action_width = 29 + int(learn_release)
            if residual.shape != (action_width,) or not np.isfinite(residual).all():
                raise ValueError(f'residual action must contain {action_width} finite values')
            residual = np.clip(residual, -1, 1)
            release_permitted = not learn_release or residual[-1] >= 0
            if learn_release:
                release_action_rows.append(float(residual[-1]))
                residual = residual[:29]
            correction = residual_torque(residual, caps)
            residual_rows.append(residual.copy())
            sequence_rows.append([monitor.sequence_time, monitor.sequence_rate])
            rot = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
            command = delivery_command(monitor.sequence_time, yaw, data.qpos[1] - lane, gather_deceleration, deceleration_start, preserve_lane_direction, data.qvel[1], lane_damping, (rot @ data.qvel[3:6])[2], heading_damping)
            obs = history.append((data.qvel[3:6], local_rotation(data.qpos[3:7]), command, data.qpos[q], data.qvel[v], previous))
            with torch.no_grad():
                previous = actor(torch.from_numpy(obs)[None])[0].numpy()
            target = default_angles() + 0.25 * previous
            if steady_carry:
                target = steady_carry_targets(target)
            if carry_elbow is not None:
                target = carry_elbow_targets(target, carry_elbow)
            phase, phase_speed = delivery_phase(monitor.sequence_time + (REFERENCE_START - reference_start), delivery_rate)
            phase_speed *= monitor.sequence_rate
            if synchronize_reference_arms:
                phase, phase_speed = (REFERENCE_START + swing_local, swing_speed)
            local = np.clip(swing_local if synchronize_reference_arms else phase - REFERENCE_START, 0, END_TIME)
            weight = float(arm_weight(phase))
            carry_target = target[bowling].copy()
            target[arms] += weight * (arm_reference(local) - target[arms])
            velocity[:] = 0
            if REFERENCE_START <= phase <= REFERENCE_START + END_TIME:
                velocity[arms] = weight * phase_speed * arm_reference(local, nu=1)
            bowling_phase, bowling_speed = (phase, phase_speed)
            bowling_local, bowling_weight = (local, weight)
            if support_triggered_swing:
                bowling_phase = REFERENCE_START + swing_local
                bowling_speed = swing_speed
                bowling_local = np.clip(swing_local, 0, END_TIME)
                bowling_weight = float(arm_weight(bowling_phase))
                target[bowling] = carry_target + bowling_weight * (arm_reference(bowling_local)[bowling_reference] - carry_target)
                velocity[bowling] = 0
                if 0 <= swing_local <= END_TIME:
                    velocity[bowling] = bowling_weight * swing_speed * arm_reference(bowling_local, nu=1)[bowling_reference]
            if carry_roll is not None:
                target, velocity = carry_roll_targets(target, velocity, carry_roll)
            target[arms] = np.clip(target[arms], *model.jnt_range[joints[arms]].T)
            if native_delivery_impedance or native_bowling_elbow_impedance:
                effective_kp[impedance_joints] = kp[impedance_joints] + weight * (native_kp[impedance_joints] - kp[impedance_joints])
                effective_kd[impedance_joints] = kd[impedance_joints] + weight * (native_kd[impedance_joints] - kd[impedance_joints])
                if support_triggered_swing:
                    selected = np.intersect1d(impedance_joints, bowling)
                    effective_kp[selected] = kp[selected] + bowling_weight * (native_kp[selected] - kp[selected])
                    effective_kd[selected] = kd[selected] + bowling_weight * (native_kd[selected] - kd[selected])
            if balance is not None:
                blend = gather_blend(monitor.sequence_time, gather_start)
                torso_command = torso_heading_command(yaw, blend) if compensate_torso_heading else np.zeros(3)
                torso_command[1] = bowling_pitch_command(bowling_local, bowling_torso_pitch)
                balance.set_observation(dict(q=data.qpos[q[sdk]].copy(), dq=data.qvel[v[sdk]].copy(), floating_base_pose=data.qpos[:7].copy(), floating_base_vel=data.qvel[:6].copy()))
                balance_target = balance.get_action(time=time, base_height_command=0.78, torso_orientation_rpy=torso_command, interpolated_navigate_cmd=command)['body_action'][0]
                balance_target = np.clip(balance_target, *model.jnt_range[joints[lower]].T)
                target[lower] += blend * (balance_target - target[lower])
                effective_kp[lower] = kp[lower] + blend * (np.asarray(balance.config['kps'])[:15] - kp[lower])
                effective_kd[lower] = kd[lower] + blend * (np.asarray(balance.config['kds'])[:15] - kd[lower])
                balance_rows.append(np.r_[time, blend, balance.obs_buffer, balance.action, balance_target])
                torso_command_rows.append(torso_command)
            if clearance is not None:
                target[arms], diagnostic = clearance.project(data, target[arms])
                clearance_rows.append(np.r_[time, diagnostic])
            if native_target_limits:
                target = native_target_bounds(target, model.jnt_range[joints])
            feedforward[:] = 0
            if arm_inertia_compensation:
                window = transition(bowling_local, 1.4, 1.6) * (1 - transition(bowling_local, 2.34, 2.7))
                acceleration = arm_reference(bowling_local, nu=2)[bowling_reference] * bowling_speed ** 2
                feedforward[bowling] = window * arm_inertia_request(model, data, v[bowling], acceleration)
            if monitor.events.release_record is None and release_permitted and (not support_triggered_swing or swing_clock.can_release(time, monitor.events, enforce_footwork=learn_release)) and (REFERENCE_START + GATHER_TIME <= bowling_phase <= REFERENCE_START + END_TIME):
                shoulder, elbow, _ = data.xpos[monitor.arm]
                ready = overarm_release_ready(data.qpos[None, bq:bq + 3], data.qvel[None, bv:bv + 3], shoulder[None], elbow[None], allow_upward=allow_upward_release)[0]
                if delivery_style == 'underarm':
                    ready = underarm_release_ready(data.qpos[None, bq:bq + 3], data.qvel[None, bv:bv + 3], shoulder[None], underarm_minimum_loft_deg)[0]
                if model_launch or shoulder_plan is not None:
                    ready = launch_ready(data.qpos[bq:bq + 3], data.qvel[bv:bv + 3], shoulder, elbow, -float(model.opt.gravity[2]), float(model.geom('ball_geom').size[0]))
                if ready and data.qvel[bv] > minimum_release_speed:
                    upper, angle = arm_geometry(*data.xpos[monitor.arm])
                    monitor.events.release(time, data.qpos[bq:bq + 3], data.qvel[bv:bv + 3], float(shoulder[2]), upper, angle)
                    data.eq_active[model.equality('ball_holder').id] = False
                    if release_arm_hold:
                        release_pose = data.qpos[q[bowling]].copy()
            if release_pose is not None:
                target[bowling], velocity[bowling] = release_hold_targets(time - monitor.events.release_record['time'], release_pose, target[bowling], velocity[bowling])
            if shoulder_plan is not None:
                plan_command = shoulder_schedule(bowling_local, plan_parameters, hand, plan_axis) * caps[plan_selected]
                plan_blend = float(np.interp(bowling_local, KNOTS, [0, 1, 1, 1, 0]))
                plan_rows.append(np.r_[bowling_local, plan_blend, plan_command])
            if launch_controller is not None:
                launch_warmstart_rows.append(data.qacc_warmstart.copy())
                launch_torque = None
                diagnostic = np.zeros(9)
                if monitor.events.release_record is None and 1.7 <= bowling_local <= 2.0:
                    baseline = bounded_torque(effective_kp * (target - data.qpos[q]) + effective_kd * (velocity - data.qvel[v]) + feedforward, correction, caps, post_saturation_residual=post_saturation_residual)
                    launch_torque, diagnostic = launch_controller.command(data, baseline)
                launch_rows.append(np.r_[launch_torque is not None, np.zeros(3) if launch_torque is None else launch_torque, diagnostic])
                if lookahead is not None:
                    selection, predictions = (np.zeros(2), np.zeros((5, 7)))
                    predicted_state = np.zeros_like(state)
                    if launch_torque is not None:
                        selection, predictions, predicted_state = lookahead.choose(data, np.r_[target, velocity], np.r_[effective_kp, effective_kd], feedforward, launch_torque)
                        launch_fraction = selection[0]
                    launch_selection_rows.append(selection)
                    launch_lookahead_rows.append(predictions)
                    launch_prediction_rows.append(predicted_state)
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            states.append(state.copy())
            observations.append(obs)
            actions.append(previous.copy())
            targets.append(np.r_[target, velocity])
            impedance_rows.append(np.r_[effective_kp, effective_kd])
            feedforward_rows.append(feedforward.copy())
            released.append(monitor.events.release_record is not None)
        torque = bounded_torque(effective_kp * (target - data.qpos[q]) + effective_kd * (velocity - data.qvel[v]) + feedforward, correction, caps, post_saturation_residual=post_saturation_residual)
        if launch_torque is not None:
            selected = launch_controller.selected
            if launch_fraction == 1:
                torque[selected] = launch_torque
            elif launch_fraction:
                torque[selected] += launch_fraction * (launch_torque - torque[selected])
        plan_active = shoulder_plan is not None and plan_blend > 0
        if plan_active:
            plan_baseline = torque[plan_selected].copy()
            torque[plan_selected] = shoulder_torque(plan_baseline, plan_command, caps[plan_selected], plan_blend, plan_mode, plan_axis)
            plan_time = float(data.time)
        effort_active = maximum_arm_effort and 1.7 <= bowling_local <= 2.3
        if effort_active:
            requested = torque[bowling].copy()
            torque[bowling] = maximum_effort(requested, model.actuator_forcerange[ids[bowling]])
            effort_time = float(data.time)
        data.ctrl[ids] = data.qpos[q] + (torque + native_kd * data.qvel[v]) / native_kp
        if batch_step is None:
            mujoco.mj_step(model, data)
        else:
            batch_step(data)
        mujoco.mj_forward(model, data)
        if plan_active:
            plan_native_rows.append(np.r_[plan_time, plan_baseline, torque[plan_selected], data.actuator_force[ids[plan_selected]]])
        if effort_active:
            maximum_effort_rows.append(np.r_[effort_time, requested, torque[bowling], data.actuator_force[ids[bowling]]])
        metrics.append(monitor.observe(data))
        if not np.isfinite(np.r_[data.qpos, data.qvel, metrics[-1]]).all() or data.warning.number.any():
            raise RuntimeError('invalid native bowling dynamics')
        if data.qpos[2] < 0.48 or 1 - 2 * np.square(data.qpos[4:6]).sum() < 0.65:
            terminal = 'fallen'
            break
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states.append(state.copy())
    monitor.sequence_time, monitor.sequence_rate = clock.advance(float(data.time), data.qpos[0])
    if support_triggered_swing:
        monitor.swing_local, monitor.swing_rate = swing_clock.advance(float(data.time), monitor.sequence_time - reference_start, monitor.events, monitor.foot_loads[0 if hand == 'right' else 1])
        monitor.swing_started = swing_clock.started_at is not None
    name = f'{hand}_dt{dt:g}'
    trace = output / f'{name}.npz'
    records = dict(states=states, sequence_rows=sequence_rows, swing_rows=swing_rows, observations=observations, actions=actions, targets=targets, metrics=metrics, released=released, balance_rows=balance_rows, torso_command_rows=torso_command_rows, clearance_rows=clearance_rows, impedance_rows=impedance_rows, feedforward_rows=feedforward_rows, residual_rows=residual_rows)
    records = {key: np.asarray(value) for key, value in records.items()}
    if maximum_arm_effort:
        records['maximum_effort_rows'] = np.asarray(maximum_effort_rows).reshape(-1, 22)
    if shoulder_plan is not None:
        records['shoulder_plan_rows'] = np.asarray(plan_rows).reshape(-1, 5)
        records['shoulder_plan_native_rows'] = np.asarray(plan_native_rows).reshape(-1, 10)
    if learn_release:
        records['release_action_rows'] = np.asarray(release_action_rows)
    if model_launch:
        records['launch_rows'] = np.asarray(launch_rows)
        records['launch_warmstart_rows'] = np.asarray(launch_warmstart_rows)
    if model_launch_lookahead:
        records['launch_selection_rows'] = np.asarray(launch_selection_rows)
        records['launch_lookahead_rows'] = np.asarray(launch_lookahead_rows)
        records['launch_prediction_rows'] = np.asarray(launch_prediction_rows)
    if retain:
        np.savez_compressed(trace, **records)
    result = monitor.events.finish(terminal == 'complete')
    result.update(case=name, terminal=terminal, duration_s=float(data.time), hand=hand, timestep=dt, max_lateral_excursion_m=float(np.abs(np.asarray(metrics)[:, 2] - lane).max()), max_artificial_support=float(np.asarray(metrics)[:, 8].max()), landings=monitor.events.landings, robot_contact_details=monitor.robot_contacts, joint_limit_excess_rad={model.joint(j).name: float(value) for j, value in zip(monitor.joints, monitor.joint_excess) if value > 0})
    if retain:
        result['trace'] = fingerprint(trace)
    return (result, records)

def run_case(actor, unilab, output, hand, dt, **options):
    rollout = rollout_case(actor, unilab, output, hand, dt, **options)
    while True:
        try:
            next(rollout)
        except StopIteration as finished:
            return finished.value[0]

def run(upstream, unilab, output, *, gather_deceleration=False, groot_gather=None, arm_clearance=False, native_target_limits=False, delivery_rate=1.0, minimum_release_speed=1.0, native_delivery_impedance=False, native_bowling_elbow_impedance=False, compensate_torso_heading=False, bowling_torso_pitch=0.0, carry_roll=None, carry_elbow=None, steady_carry=False, preserve_lane_direction=False, lane_damping=0.0, heading_damping=0.0, lane_offset=0.5, support_triggered_swing=False, synchronize_reference_arms=False, approach_gate_x=None, arm_reference_directory=None, reference_start=REFERENCE_START, arm_inertia_compensation=False, release_arm_hold=False, physics_backend='mujoco', delivery_style='overarm', underarm_minimum_loft_deg=0.0, start_x=-10.5, gather_start=3.0, deceleration_start=3.0, wicket_contact=False, model_launch=False, model_launch_lookahead=False, learn_release=False, allow_upward_release=False):
    if allow_upward_release and (not learn_release or delivery_style != 'overarm'):
        raise ValueError('Upward release requires learned overarm release')
    if model_launch_lookahead and (not model_launch):
        raise ValueError('Launch lookahead requires model launch')
    if model_launch and (delivery_style != 'overarm' or not support_triggered_swing):
        raise ValueError('Model launch requires support-triggered overarm')
    if physics_backend not in {'mujoco', 'mjbatch'}:
        raise ValueError('Unknown physics backend')
    if delivery_style not in {'overarm', 'underarm'}:
        raise ValueError('Unknown delivery style')
    if not 0 <= underarm_minimum_loft_deg <= 60 or (delivery_style != 'underarm' and underarm_minimum_loft_deg != 0):
        raise ValueError('Underarm loft must be in [0, 60] degrees and requires underarm style')
    if not np.isfinite(delivery_rate) or delivery_rate <= 0:
        raise ValueError('delivery rate must be finite and positive')
    if release_arm_hold and arm_inertia_compensation:
        raise ValueError('release arm hold requires reference inertia feedforward disabled')
    if synchronize_reference_arms and (not support_triggered_swing):
        raise ValueError('synchronized reference arms require the support-triggered swing')
    if not np.isfinite(minimum_release_speed) or minimum_release_speed < 1:
        raise ValueError('minimum release speed must be finite and at least 1 m/s')
    if not 0 <= reference_start <= 5.3:
        raise ValueError('reference must start within the complete episode')
    if not np.isfinite(start_x):
        raise ValueError('start position must be finite')
    if not 0 <= gather_start <= 7:
        raise ValueError('gather transition must fit within the complete episode')
    if not 0 <= deceleration_start <= 7:
        raise ValueError('deceleration must fit within the complete episode')
    if groot_gather is not None and (not gather_deceleration):
        raise ValueError('GR00T gather requires the decelerating command')
    if compensate_torso_heading and groot_gather is None:
        raise ValueError('torso heading compensation requires GR00T gather')
    if not 0 <= bowling_torso_pitch <= 0.3:
        raise ValueError('bowling torso pitch outside the declared design range')
    if bowling_torso_pitch and groot_gather is None:
        raise ValueError('bowling torso pitch requires GR00T gather')
    if carry_roll is not None and (not 0.25 <= carry_roll <= 0.8):
        raise ValueError('carry roll outside the declared design range')
    if carry_elbow is not None and (not 0.6 <= carry_elbow <= 1.3):
        raise ValueError('carry elbow outside the declared design range')
    if not 0 <= lane_damping <= 2:
        raise ValueError('lane damping outside the declared design range')
    if not 0 <= heading_damping <= 1:
        raise ValueError('heading damping outside the declared design range')
    if not 0 < lane_offset < RETURN_Y:
        raise ValueError('lane offset must lie between the wicket and return crease')
    if support_triggered_swing and (delivery_rate != 1.0 or approach_gate_x is not None or (not steady_carry)):
        raise ValueError('Support-triggered swing requires steady carry, unit rate and an independent wall-time approach')
    if approach_gate_x is not None:
        if not start_x < approach_gate_x < -1.22:
            raise ValueError('approach gate must be between reset and bowling crease')
        if not gather_deceleration or gather_start != deceleration_start:
            raise ValueError('approach gate requires synchronized gather and deceleration')
    torch.set_num_threads(1)
    actor = load_actor(upstream)
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(Path('integrations/g1_dynamics').glob('*.py'))
    if physics_backend == 'mjbatch':
        paths += [path.relative_to(Path.cwd()) for path in backend_inputs()]
        paths += [Path('tests/test_mjbatch_stepper.py'), Path('reports/g1_mjbatch_live_backend_plan.md')]
    paths += sorted((unilab / 'src/unilab/tasks/manipulation/g1_cricket').glob('*.py'))
    paths += sorted((p for p in (unilab / 'src/unilab/assets/robots/g1').rglob('*') if p.is_file()))
    if arm_reference_directory is not None:
        paths += [arm_reference_directory / f'{hand}_dense_reference.npz' for hand in ('right', 'left')]
        paths += [arm_reference_directory / 'summary.json']
    paths += [unilab / 'scripts/g1_cricket_delivery_trial.py', upstream / 'checkpoints/model_6200.pt', Path('reports/g1_amp_bowling_plan.md')]
    paths += sorted((unilab / 'g1_cricket_results/running_fore_aft_support_v1').glob('*_dense_reference.npz'))
    if groot_gather is not None:
        paths += sorted((groot_gather / 'decoupled_wbc').rglob('*.py'))
        paths += sorted((groot_gather / 'decoupled_wbc/sim2mujoco/resources/robots/g1').rglob('*.yaml'))
        paths += sorted((groot_gather / 'decoupled_wbc/sim2mujoco/resources/robots/g1/policy').glob('*.onnx'))
    inputs = {str(p): fingerprint(p) for p in paths}
    with zipfile.ZipFile(output / 'source_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.suffix in {'.py', '.xml', '.md', '.cpp', '.h'}:
                archive.write(path, arcname=path)
    rows = []
    for hand in ('right', 'left'):
        if wicket_contact:
            build_scene(unilab, output / f'{hand}.xml', hand, wicket_contact=True)
        for dt in (6.25e-05, 3.125e-05):
            print('START', hand, dt, flush=True)
            row = run_case(actor, unilab, output, hand, dt, gather_deceleration=gather_deceleration, groot_gather=groot_gather, arm_clearance=arm_clearance, native_target_limits=native_target_limits, delivery_rate=delivery_rate, minimum_release_speed=minimum_release_speed, native_delivery_impedance=native_delivery_impedance, native_bowling_elbow_impedance=native_bowling_elbow_impedance, compensate_torso_heading=compensate_torso_heading, bowling_torso_pitch=bowling_torso_pitch, carry_roll=carry_roll, carry_elbow=carry_elbow, steady_carry=steady_carry, preserve_lane_direction=preserve_lane_direction, lane_damping=lane_damping, heading_damping=heading_damping, lane_offset=lane_offset, support_triggered_swing=support_triggered_swing, synchronize_reference_arms=synchronize_reference_arms, approach_gate_x=approach_gate_x, arm_reference_directory=arm_reference_directory, reference_start=reference_start, arm_inertia_compensation=arm_inertia_compensation, release_arm_hold=release_arm_hold, physics_backend=physics_backend, delivery_style=delivery_style, underarm_minimum_loft_deg=underarm_minimum_loft_deg, start_x=start_x, gather_start=gather_start, deceleration_start=deceleration_start, model_launch=model_launch, model_launch_lookahead=model_launch_lookahead, learn_release=learn_release, allow_upward_release=allow_upward_release)
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    for path, expected in inputs.items():
        assert fingerprint(Path(path)) == expected, path
    result = dict(scope='Published AMP locomotion plus reference arms, native-model shoulder steering and mechanical ball holder; not learned cricket or a free finger grasp' if model_launch else 'Published AMP locomotion plus reference arms and mechanical ball holder; not locally learned cricket or a free finger grasp', delivery_style=delivery_style, physics_backend=physics_backend, underarm_minimum_loft_deg=underarm_minimum_loft_deg, regulation_note='Underarm range comparison only; original overarm qualification failures remain visible. MCC Law 21 requires prior agreement for underarm bowling.' if delivery_style == 'underarm' else 'Full overarm delivery checks apply.', inputs=inputs, gather_deceleration=gather_deceleration, arm_clearance=arm_clearance, native_target_limits=native_target_limits, delivery_rate=delivery_rate, minimum_release_speed=minimum_release_speed, native_delivery_impedance=native_delivery_impedance, native_bowling_elbow_impedance=native_bowling_elbow_impedance, compensate_torso_heading=compensate_torso_heading, bowling_torso_pitch=bowling_torso_pitch, carry_roll=carry_roll, carry_elbow=carry_elbow, steady_carry=steady_carry, preserve_lane_direction=preserve_lane_direction, lane_damping=lane_damping, heading_damping=heading_damping, lane_offset=lane_offset, support_triggered_swing=support_triggered_swing, synchronize_reference_arms=synchronize_reference_arms, approach_gate_x=approach_gate_x, arm_reference_directory=str(arm_reference_directory) if arm_reference_directory is not None else None, reference_start=reference_start, arm_inertia_compensation=arm_inertia_compensation, release_arm_hold=release_arm_hold, start_x=start_x, gather_start=gather_start, deceleration_start=deceleration_start, groot_gather=str(groot_gather) if groot_gather is not None else None, rows=rows, promotion_allowed=False, model_launch=model_launch, model_launch_lookahead=model_launch_lookahead, learn_release=learn_release, allow_upward_release=allow_upward_release, scene_version=WICKET_CONTACT_VERSION if wicket_contact else 'existing_delivery_pitch_v2', artifacts={p.name: fingerprint(p) for p in output.iterdir() if p.is_file()})
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'unilab', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--gather-deceleration', action='store_true')
    parser.add_argument('--groot-gather', type=Path)
    parser.add_argument('--arm-clearance', action='store_true')
    parser.add_argument('--native-target-limits', action='store_true')
    parser.add_argument('--delivery-rate', type=float, default=1.0)
    parser.add_argument('--minimum-release-speed', type=float, default=1.0)
    parser.add_argument('--native-delivery-impedance', action='store_true')
    parser.add_argument('--native-bowling-elbow-impedance', action='store_true')
    parser.add_argument('--compensate-torso-heading', action='store_true')
    parser.add_argument('--bowling-torso-pitch', type=float, default=0.0)
    parser.add_argument('--carry-roll', type=float)
    parser.add_argument('--carry-elbow', type=float)
    parser.add_argument('--steady-carry', action='store_true')
    parser.add_argument('--preserve-lane-direction', action='store_true')
    parser.add_argument('--lane-damping', type=float, default=0.0)
    parser.add_argument('--heading-damping', type=float, default=0.0)
    parser.add_argument('--lane-offset', type=float, default=0.5)
    parser.add_argument('--support-triggered-swing', action='store_true')
    parser.add_argument('--synchronize-reference-arms', action='store_true')
    parser.add_argument('--approach-gate-x', type=float)
    parser.add_argument('--arm-reference-directory', type=Path)
    parser.add_argument('--reference-start', type=float, default=REFERENCE_START)
    parser.add_argument('--arm-inertia-compensation', action='store_true')
    parser.add_argument('--release-arm-hold', action='store_true')
    parser.add_argument('--physics-backend', choices=('mujoco', 'mjbatch'), default='mujoco')
    parser.add_argument('--delivery-style', choices=('overarm', 'underarm'), default='overarm')
    parser.add_argument('--underarm-minimum-loft-deg', type=float, default=0.0)
    parser.add_argument('--start-x', type=float, default=-10.5)
    parser.add_argument('--gather-start', type=float, default=3.0)
    parser.add_argument('--deceleration-start', type=float, default=3.0)
    parser.add_argument('--wicket-contact', action='store_true', help='Opt-in numerical ball/stump contact; not material-calibrated')
    parser.add_argument('--model-launch', action='store_true', help='Experimental model-based shoulder control and ballistic release trigger; not learned cricket')
    parser.add_argument('--model-launch-lookahead', action='store_true', help='Filter model-launch torques with 20 ms native predictions; not a safety guarantee')
    parser.add_argument('--learn-release', action='store_true', help='Use the release-action interface with zero actions; does not train a policy')
    parser.add_argument('--allow-upward-release', action='store_true')
    args = parser.parse_args()
    run(args.upstream, args.unilab, args.output, gather_deceleration=args.gather_deceleration, groot_gather=args.groot_gather, arm_clearance=args.arm_clearance, native_target_limits=args.native_target_limits, delivery_rate=args.delivery_rate, minimum_release_speed=args.minimum_release_speed, native_delivery_impedance=args.native_delivery_impedance, native_bowling_elbow_impedance=args.native_bowling_elbow_impedance, compensate_torso_heading=args.compensate_torso_heading, bowling_torso_pitch=args.bowling_torso_pitch, carry_roll=args.carry_roll, carry_elbow=args.carry_elbow, steady_carry=args.steady_carry, preserve_lane_direction=args.preserve_lane_direction, lane_damping=args.lane_damping, heading_damping=args.heading_damping, lane_offset=args.lane_offset, support_triggered_swing=args.support_triggered_swing, synchronize_reference_arms=args.synchronize_reference_arms, approach_gate_x=args.approach_gate_x, arm_reference_directory=args.arm_reference_directory, reference_start=args.reference_start, arm_inertia_compensation=args.arm_inertia_compensation, release_arm_hold=args.release_arm_hold, physics_backend=args.physics_backend, delivery_style=args.delivery_style, underarm_minimum_loft_deg=args.underarm_minimum_loft_deg, start_x=args.start_x, gather_start=args.gather_start, deceleration_start=args.deceleration_start, wicket_contact=args.wicket_contact, model_launch=args.model_launch, model_launch_lookahead=args.model_launch_lookahead, learn_release=args.learn_release, allow_upward_release=args.allow_upward_release)
