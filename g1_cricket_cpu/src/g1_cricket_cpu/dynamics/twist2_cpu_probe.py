"""Headless CPU reproduction of TWIST2's released G1 sim2sim controller.

Control conventions adapted from amazon-far/TWIST2 (MIT, Yanjie Ze, 2025).
See TWIST2_LICENSE. This tests native dynamics, not cricket or grasp success.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp
UPSTREAM_COMMIT = 'b06178f19a22f2138cbd31f60c6d494bc263f67d'
DEFAULT = np.array([-0.2, 0, 0, 0.4, -0.2, 0] * 2 + [0, 0, 0] + [0, 0.4, 0, 1.2, 0, 0, 0] + [0, -0.4, 0, 1.2, 0, 0, 0], dtype=float)
KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40, 40, 40, 40, 4, 4, 4] * 2)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5, 5, 5, 5, 0.2, 0.2, 0.2] * 2)
STAND = np.r_[0, 0, 0.8, 0, 0, 0, DEFAULT]

class NumpyMotionUnpickler(pickle.Unpickler):
    """Only deserialize numeric arrays from the external motion asset."""

    def find_class(self, module, name):
        allowed = {('numpy', 'dtype'): np.dtype, ('numpy', 'ndarray'): np.ndarray, ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar, ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct, ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar, ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Disallowed motion pickle global: {module}.{name}')
        return allowed[module, name]

def load_motion(path):
    with path.open('rb') as stream:
        data = NumpyMotionUnpickler(stream).load()
    dt = 1.0 / float(data['fps'])
    pos = np.asarray(data['root_pos'])
    rot = Rotation.from_quat(data['root_rot'])
    vel = np.gradient(pos, dt, axis=0)
    omega = np.empty_like(pos)
    omega[1:-1] = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
    omega[0] = (rot[1] * rot[0].inv()).as_rotvec() / dt
    omega[-1] = (rot[-1] * rot[-2].inv()).as_rotvec() / dt
    return (data, rot, vel, omega, dt)

def motion_commands(path):
    """Match MotionLib interpolation and the 50 Hz reference publisher."""
    data, rot, vel, omega, dt = load_motion(path)
    length = dt * (len(rot) - 1)
    times = np.arange(int(length / 0.02)) * 0.02
    frame = times / dt
    lo = np.floor(frame).astype(int)
    hi = np.minimum(lo + 1, len(rot) - 1)
    blend = (frame - lo)[:, None]
    sampled_rot = Slerp(np.arange(len(rot)) * dt, rot)(times)
    height = ((1 - blend) * data['root_pos'][lo] + blend * data['root_pos'][hi])[:, 2:3]
    joints = (1 - blend) * data['dof_pos'][lo] + blend * data['dof_pos'][hi]
    return np.column_stack((sampled_rot.inv().apply(vel[lo])[:, :2], height, sampled_rot.as_euler('xyz')[:, :2], sampled_rot.inv().apply(omega[lo])[:, 2], joints))

def observation(data, mimic, previous_action, history):
    velocity = data.qvel[6:35].copy()
    velocity[[4, 5, 10, 11]] = 0
    rpy = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler('xyz')
    current = np.r_[mimic, data.qvel[3:6] * 0.25, rpy[:2], data.qpos[7:36] - DEFAULT, velocity * 0.05, previous_action]
    result = np.r_[current, np.asarray(history).ravel(), mimic].astype(np.float32)
    history.append(current)
    return result[None]

def wrist_pair(data, bodies):
    left, right = bodies
    rotation = data.xmat[left].reshape(3, 3)
    position = rotation.T @ (data.xpos[right] - data.xpos[left])
    relative_rotation = rotation.T @ data.xmat[right].reshape(3, 3)
    return (position, relative_rotation)

def cricket_commands(model, scene, reference):
    source = mujoco.MjModel.from_xml_path(str(scene))
    names = [model.joint(index).name for index in range(1, model.njnt)]
    joints = np.array([source.joint(name).id for name in names])
    with np.load(reference) as saved:
        poses, times = (saved['qpos'], saved['times'])
    root_rot = Rotation.from_quat(poses[:, 3:7], scalar_first=True)
    velocity = np.gradient(poses[:, :3], times, axis=0)
    sample_times = np.arange(400) * 0.02
    positions = poses[:, source.jnt_qposadr[joints]]
    commands = np.column_stack((root_rot.inv().apply(velocity)[:, :2], poses[:, 2], root_rot.as_euler('xyz')[:, :2], np.zeros(len(times)), positions))
    if np.max(np.abs((root_rot * root_rot[0].inv()).as_rotvec())) > 1e-08:
        raise ValueError('This transfer probe expects the retained fixed-root-orientation reference')
    sampled = np.column_stack([np.interp(sample_times, times, column) for column in commands.T])
    sampled[sample_times >= times[-1], :2] = 0
    initial = np.r_[poses[0, :7], positions[0]]
    initial[:2] = 0
    compatibility = {'source_actuators': source.nu, 'target_actuators': model.nu, 'source_mass_kg': float(source.body_mass.sum()), 'target_mass_kg': float(model.body_mass.sum()), 'body_joint_names': names, 'joint_axes_match': bool(np.array_equal(source.jnt_axis[joints], model.jnt_axis[1:])), 'joint_limits_match': bool(np.array_equal(source.jnt_range[joints], model.jnt_range[1:])), 'omitted': 'All fourteen articulated finger joints, bat and ball. This is body-reference tracking, not grip transfer.', 'reference': str(reference), 'scene': str(scene)}
    return (sampled, initial, compatibility)

def simulate(model, session, commands, seconds, initial=None):
    data = mujoco.MjData(model)
    data.qpos[:] = np.r_[0, 0, 0.793, 1, 0, 0, 0, DEFAULT]
    data.qpos[7 + 16] = 0.2
    data.qpos[7 + 23] = -0.2
    if initial is not None:
        data.qpos[:] = initial
    mujoco.mj_forward(model, data)
    history = deque([np.zeros(127) for _ in range(10)], maxlen=10)
    previous = np.zeros(29)
    target = DEFAULT.copy()
    records = {key: [] for key in ('time', 'qpos', 'qvel', 'action', 'reference', 'torque', 'joint_torque', 'ground_force_N', 'nonfoot_ground_contacts', 'wrist_pair_position_error_m', 'wrist_pair_rotation_error_rad')}
    reference_data = mujoco.MjData(model)
    wrists = [model.body(f'{hand}_wrist_yaw_link').id for hand in ('left', 'right')]
    timings = []
    steps = int(round(seconds / model.opt.timestep))
    input_name = session.get_inputs()[0].name
    warnings_before = data.warning.number.copy()
    start = time.perf_counter()
    max_torque = np.zeros(29)
    max_joint_torque = np.zeros(29)
    minimum_height = float(data.qpos[2])
    maximum_tilt = 0.0
    nonfoot_steps = 0
    peak_ground_force = 0.0
    finite = True
    for step in range(steps):
        mimic = commands[min(step // 20, len(commands) - 1)]
        if step % 10 == 0:
            obs = observation(data, mimic, previous, history)
            before = time.perf_counter()
            previous = session.run(None, {input_name: obs})[0].reshape(29)
            timings.append(time.perf_counter() - before)
            target = DEFAULT + 0.5 * np.clip(previous, -10, 10)
        data.ctrl[:] = np.clip((target - data.qpos[7:36]) * KP - data.qvel[6:35] * KD, -KP, KP)
        mujoco.mj_step(model, data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite:
            break
        minimum_height = min(minimum_height, float(data.qpos[2]))
        rpy = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler('xyz')
        maximum_tilt = max(maximum_tilt, float(np.max(np.abs(rpy[:2]))))
        max_torque = np.maximum(max_torque, np.abs(data.actuator_force))
        max_joint_torque = np.maximum(max_joint_torque, np.abs(data.qfrc_actuator[6:35]))
        ground_force = 0.0
        nonfoot = 0
        for index in range(data.ncon):
            contact = data.contact[index]
            body_ids = model.geom_bodyid[[contact.geom1, contact.geom2]]
            if 0 not in body_ids:
                continue
            body = model.body(int(max(body_ids))).name
            nonfoot += int(body not in ('left_ankle_roll_link', 'right_ankle_roll_link'))
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            ground_force += float(force[0])
        nonfoot_steps += int(nonfoot > 0)
        peak_ground_force = max(peak_ground_force, ground_force)
        if step % 10 == 9:
            reference_data.qpos[:] = data.qpos
            mujoco.mj_kinematics(model, reference_data)
            actual_pair, actual_rotation = wrist_pair(reference_data, wrists)
            reference_data.qpos[7:36] = mimic[6:]
            mujoco.mj_kinematics(model, reference_data)
            expected_pair, expected_rotation = wrist_pair(reference_data, wrists)
            pair_error = np.linalg.norm(actual_pair - expected_pair)
            pair_rotation_error = Rotation.from_matrix(actual_rotation @ expected_rotation.T).magnitude()
            values = (data.time, data.qpos.copy(), data.qvel.copy(), previous.copy(), mimic.copy(), data.actuator_force.copy(), data.qfrc_actuator[6:35].copy(), ground_force, nonfoot, pair_error, pair_rotation_error)
            for key, value in zip(records, values):
                records[key].append(value)
    elapsed = time.perf_counter() - start
    traces = {key: np.asarray(value) for key, value in records.items()}
    warnings = (data.warning.number - warnings_before).tolist()
    complete = finite and step + 1 == steps and (abs(data.time - seconds) < 1e-06)
    checks = {'complete_finite_episode': complete, 'no_mujoco_warnings': not any(warnings), 'pelvis_above_0_5m': minimum_height > 0.5, 'roll_pitch_below_0_7rad': maximum_tilt < 0.7, 'no_nonfoot_ground_contact': nonfoot_steps == 0}
    result = {'duration_s': float(data.time), 'wall_time_s': elapsed, 'realtime_factor': float(data.time / elapsed), 'cpu_inference_median_ms': float(np.median(timings) * 1000), 'cpu_inference_p95_ms': float(np.percentile(timings, 95) * 1000), 'minimum_pelvis_height_m': minimum_height, 'maximum_roll_pitch_rad': maximum_tilt, 'final_displacement_xy_m': data.qpos[:2].tolist(), 'path_length_xy_m': float(np.linalg.norm(np.diff(traces['qpos'][:, :2], axis=0), axis=1).sum()), 'joint_reference_rmse_rad': float(np.sqrt(np.mean((traces['qpos'][:, 7:36] - traces['reference'][:, 6:]) ** 2))), 'maximum_actuator_force_Nm': max_torque.tolist(), 'maximum_joint_actuation_Nm': max_joint_torque.tolist(), 'maximum_wrist_pair_position_error_m': float(traces['wrist_pair_position_error_m'].max()), 'maximum_wrist_pair_rotation_error_rad': float(traces['wrist_pair_rotation_error_rad'].max()), 'wrist_error_sampling_hz': 100, 'peak_ground_normal_force_N': peak_ground_force, 'nonfoot_ground_contact_substeps': nonfoot_steps, 'warnings': warnings, 'checks': checks, 'smoke_pass': all(checks.values())}
    return (result, traces)

def fingerprint(path):
    return {'size_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cricket-checkout', type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('Unexpected TWIST2 revision; review its deployment contract first')
    xml = args.upstream / 'assets/g1/g1_sim2sim_29dof.xml'
    checkpoint = args.upstream / 'assets/ckpts/twist2_1017_20k.onnx'
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = 0.001
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(checkpoint), sess_options=options, providers=['CPUExecutionProvider'])
    if model.nu != 29 or model.nq != 36 or session.get_inputs()[0].shape[-1] != 1432:
        raise ValueError('Unexpected model or policy dimensions')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [xml, checkpoint, Path(__file__), args.upstream / 'LICENSE', args.upstream / 'deploy_real/server_low_level_g1_sim.py', args.upstream / 'deploy_real/server_motion_lib.py']
    cases = [('stand', np.tile(STAND, (500, 1)), 10.0, None)]
    for index in (1, 2):
        path = args.upstream / f'assets/example_motions/0807_yanjie_walk_{index:03d}.pkl'
        sources.append(path)
        commands = motion_commands(path)
        cases.append((path.stem, commands, len(commands) * 0.02, None))
    compatibility = {}
    if args.cricket_checkout:
        scene = args.cricket_checkout / 'g1_cricket_results/articulated_stance/scene.xml'
        sources.append(scene)
        for hand in ('right', 'left'):
            reference = args.cricket_checkout / f'g1_cricket_results/articulated_swing_compact/{hand}_reference.npz'
            sources.append(reference)
            commands, initial, compatibility[hand] = cricket_commands(model, scene, reference)
            cases.append((f'cricket_body_reference_{hand}', commands, 8.0, initial))
    results = {}
    for name, commands, seconds, initial in cases:
        result, trace = simulate(model, session, commands, seconds, initial)
        if name.startswith('cricket_body_reference_'):
            result['body_reference_grip_geometry_pass'] = result['maximum_wrist_pair_position_error_m'] <= 0.01 and result['maximum_wrist_pair_rotation_error_rad'] <= 0.1
        trace_path = args.output / f'{name}.npz'
        np.savez_compressed(trace_path, **trace)
        result['trace'] = {'name': trace_path.name, **fingerprint(trace_path)}
        results[name] = result
        print(json.dumps({'case': name, **result}), flush=True)
    report = {'upstream': 'https://github.com/amazon-far/TWIST2', 'commit': commit, 'code_license': 'MIT', 'runtime': {'mujoco': mujoco.__version__, 'onnxruntime': ort.__version__}, 'control_hz': 100, 'reference_hz': 50, 'sim_dt_s': 0.001, 'providers': session.get_providers(), 'scope': 'Upstream 29-DOF G1 native dynamics only; no articulated fingers, bat, ball or cricket transfer', 'reset': 'Upstream stand for stand/walk; retained first pose for cricket body tests. No runtime pose placement, welds or external support', 'motion_end': 'Full first two recorded walks; no looping or selected windows', 'cricket_body_reference_compatibility': compatibility, 'torque_limit_note': "Upstream software clipping equals KP; XML actuator limits remain active. Not our cricket model's validated caps.", 'native_joint_force_limits_Nm': model.jnt_actfrcrange[1:].tolist(), 'body_reference_grip_gate': 'At 100 Hz: wrist-pair translation <=0.01 m and rotation <=0.1 rad. Necessary geometry screen only, not contact/grip proof.', 'sources': {str(path): fingerprint(path) for path in sources}, 'cases': results}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
if __name__ == '__main__':
    main()
