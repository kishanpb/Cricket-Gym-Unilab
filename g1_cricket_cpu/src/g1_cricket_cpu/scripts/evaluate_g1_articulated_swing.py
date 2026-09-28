"""Complete no-ball swing and recovery comparisons, with both grips retained."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from g1_cricket_cpu.scripts.evaluate_g1_articulated_stance import forbidden_contact
from g1_cricket_cpu.physics.articulated_batting import fit_stance
from g1_cricket_cpu.physics.articulated_hands import hand_contact_state
from g1_cricket_cpu.physics.articulated_swing import SWING_AMPLITUDE, ArticulatedSwingController, retarget_swing
ROOT = Path(__file__).resolve().parents[1]
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS

def run_case(model, poses, times, *, inertial, controller=None, reset=None, observe=None, step_schedule=None):
    rollout = rollout_case(model, poses, times, inertial=inertial, controller=controller, reset=reset, observe=observe, step_schedule=step_schedule)
    while True:
        try:
            next(rollout)
        except StopIteration as finished:
            return finished.value

def rollout_case(model, poses, times, *, inertial, controller=None, reset=None, observe=None, step_schedule=None, physics_step=None):
    """Yield the initial and sampled states; return the complete audit on exhaustion."""
    if controller is None:
        controller = ArticulatedSwingController(model, poses, times, inertial=inertial)
    data, geometry = (mujoco.MjData(model), mujoco.MjData(model))
    data.qpos[:] = poses[0]
    if reset is not None:
        reset(data)
    mujoco.mj_forward(model, data)
    bat = model.body('cricket_bat').id
    bat_q, bat_v = (model.joint('bat_free').qposadr[0], model.joint('bat_free').dofadr[0])
    wrists = [model.body(f'{s}_wrist_yaw_link').id for s in ('left', 'right')]
    feet = [model.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')]
    foot_start = data.xpos[feet].copy()
    initial_rotation = data.xmat[bat].reshape(3, 3).copy()
    grip_position = (data.xpos[wrists] - data.xpos[bat]) @ initial_rotation
    grip_rotation = initial_rotation.T @ data.xmat[wrists].reshape(2, 3, 3)
    robot = {model.body('pelvis').id}
    for body in range(model.nbody):
        if model.body_parentid[body] in robot:
            robot.add(body)
    joints = model.actuator_trnid[:, 0]
    q, v = (model.jnt_qposadr[joints], model.jnt_dofadr[joints])
    limits = model.jnt_range[joints]
    caps = np.r_[model.actuator_forcerange[:29, 1], controller.fingers.force_limits[:, 1]]
    metrics = dict(minimum_root_height_m=float(data.qpos[2]), max_root_tilt_rad=0.0, max_foot_displacement_m=0.0, max_relative_grip_position_m=0.0, max_relative_grip_angle_rad=0.0, max_penetration_m=0.0, max_robot_self_penetration_m=0.0, forbidden_contact_steps=0, max_joint_limit_excess_rad=0.0, max_motor_force_excess_nm=0.0, saturated_steps=0, minimum_loaded_fingers_after_settle=6, max_loaded_slip_m_s=0.0, max_backswing_angle_rad=0.0, min_followthrough_angle_rad=0.0, peak_blade_speed_m_s=0.0, recovery_guard_position_m=0.0, recovery_guard_angle_rad=0.0, recovery_bat_speed_m_s=0.0, recovery_bat_angular_speed_rad_s=0.0, recovery_joint_speed_rad_s=0.0, recovery_root_speed_m_s=0.0, recovery_root_displacement_m=0.0)
    steps = round(times[-1] / model.opt.timestep)
    every = round(0.02 / model.opt.timestep)
    states, controls, tactile = ([], [], [])
    recovery_peaks = {}
    endpoint_contacts = model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    state = np.empty(mujoco.mj_stateSize(model, FULL))
    sampling = ((step + 1) % every == 0 for step in range(steps)) if step_schedule is None else step_schedule(model, data, times[-1])
    saturated_time = 0.0
    yield (data, metrics)
    for sample in sampling:
        controller.apply(data)
        velocity = data.qvel.copy()
        if physics_step is None:
            mujoco.mj_step(model, data)
        else:
            physics_step(data)
        if endpoint_contacts:
            mujoco.mj_forward(model, data)
            velocity = data.qvel.copy()
        if observe is not None:
            observe(data)
        assert not np.any(data.qfrc_applied) and (not np.any(data.xfrc_applied))
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('nonfinite articulated swing')
        contacts = hand_contact_state(model, data, velocity=velocity)
        forbidden = False
        for index, c in enumerate(data.contact):
            bodies = model.geom_bodyid[c.geom]
            if all((b in robot for b in bodies)):
                metrics['max_robot_self_penetration_m'] = max(metrics['max_robot_self_penetration_m'], -float(c.dist))
            if bat in bodies and any((b in robot for b in bodies)):
                metrics['max_penetration_m'] = max(metrics['max_penetration_m'], -float(c.dist))
            if forbidden_contact(model, c.geom, robot, set(feet), closing=True):
                wrench = np.empty(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                forbidden |= wrench[0] > 0.1
        metrics['forbidden_contact_steps'] += int(forbidden)
        geometry.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, geometry)
        rotation = geometry.xmat[bat].reshape(3, 3)
        relative_position = (geometry.xpos[wrists] - geometry.xpos[bat]) @ rotation
        relative_rotation = rotation.T @ geometry.xmat[wrists].reshape(2, 3, 3)
        metrics['max_relative_grip_position_m'] = max(metrics['max_relative_grip_position_m'], float(np.linalg.norm(relative_position - grip_position, axis=1).max()))
        angle = Rotation.from_matrix(grip_rotation.transpose(0, 2, 1) @ relative_rotation).magnitude()
        metrics['max_relative_grip_angle_rad'] = max(metrics['max_relative_grip_angle_rad'], float(angle.max()))
        metrics['max_foot_displacement_m'] = max(metrics['max_foot_displacement_m'], float(np.linalg.norm(geometry.xpos[feet] - foot_start, axis=1).max()))
        tilt = np.empty(3)
        mujoco.mju_subQuat(tilt, data.qpos[3:7], poses[0, 3:7])
        metrics['max_root_tilt_rad'] = max(metrics['max_root_tilt_rad'], float(np.linalg.norm(tilt[:2])))
        metrics['minimum_root_height_m'] = min(metrics['minimum_root_height_m'], float(data.qpos[2]))
        excess = np.maximum(limits[:, 0] - data.qpos[q], data.qpos[q] - limits[:, 1])
        metrics['max_joint_limit_excess_rad'] = max(metrics['max_joint_limit_excess_rad'], float(excess.max()))
        force = np.abs(data.actuator_force)
        metrics['max_motor_force_excess_nm'] = max(metrics['max_motor_force_excess_nm'], float((force - caps).max()))
        metrics['saturated_steps'] += int(np.any(force > 0.99 * caps))
        if np.any(force > 0.99 * caps):
            saturated_time += model.opt.timestep
        if data.time > 0.2:
            loaded = {(c['geom'].split('_hand_')[0], c['geom'].split('_hand_')[1].split('_')[0]) for c in contacts if c['normal_force_n'] > 0.1 and any((p in c['geom'] for p in ('thumb', 'index', 'middle')))}
            metrics['minimum_loaded_fingers_after_settle'] = min(metrics['minimum_loaded_fingers_after_settle'], len(loaded))
            metrics['max_loaded_slip_m_s'] = max(metrics['max_loaded_slip_m_s'], max((c['tangential_slip_m_s'] for c in contacts if c['normal_force_n'] > 0.1), default=0))
        pitch_angle = float(Rotation.from_matrix(rotation @ initial_rotation.T).as_rotvec()[1])
        metrics['max_backswing_angle_rad'] = max(metrics['max_backswing_angle_rad'], pitch_angle)
        metrics['min_followthrough_angle_rad'] = min(metrics['min_followthrough_angle_rad'], pitch_angle)
        angular = rotation @ data.qvel[bat_v + 3:bat_v + 6]
        blade_velocity = data.qvel[bat_v:bat_v + 3] + np.cross(angular, rotation @ [0, 0, -0.4])
        metrics['peak_blade_speed_m_s'] = max(metrics['peak_blade_speed_m_s'], float(np.linalg.norm(blade_velocity)))
        if data.time >= 4.5:
            joint_index = int(np.argmax(np.abs(data.qvel[v])))
            joint_speed = float(abs(data.qvel[v[joint_index]]))
            if joint_speed > metrics['recovery_joint_speed_rad_s']:
                recovery_peaks['joint_speed'] = {'time_s': float(data.time), 'joint': model.joint(int(joints[joint_index])).name, 'signed_velocity_rad_s': float(data.qvel[v[joint_index]])}
            guard_delta = data.qpos[bat_q:bat_q + 3] - poses[0, bat_q:bat_q + 3]
            if np.linalg.norm(guard_delta) > metrics['recovery_guard_position_m']:
                recovery_peaks['guard_position'] = {'time_s': float(data.time), 'world_error_m': guard_delta.tolist(), 'root_world_error_m': (data.qpos[:3] - poses[0, :3]).tolist()}
            for key, value in {'recovery_guard_position_m': np.linalg.norm(guard_delta), 'recovery_guard_angle_rad': Rotation.from_matrix(rotation @ initial_rotation.T).magnitude(), 'recovery_bat_speed_m_s': np.linalg.norm(data.qvel[bat_v:bat_v + 3]), 'recovery_bat_angular_speed_rad_s': np.linalg.norm(angular), 'recovery_joint_speed_rad_s': joint_speed, 'recovery_root_speed_m_s': np.linalg.norm(data.qvel[:3]), 'recovery_root_displacement_m': np.linalg.norm(data.qpos[:3] - poses[0, :3])}.items():
                metrics[key] = max(metrics[key], float(value))
        if sample:
            mujoco.mj_getState(model, data, state, FULL)
            states.append(state.copy())
            controls.append(data.ctrl.copy())
            tactile.append({'contact_time_s': float(data.time if endpoint_contacts else data.time - model.opt.timestep), 'contacts': contacts})
            yield (data, metrics)
    checks = {'upright': metrics['minimum_root_height_m'] > 0.6 and metrics['max_root_tilt_rad'] < 0.35, 'planted_feet': metrics['max_foot_displacement_m'] < 0.03, 'relative_grip_retained': metrics['max_relative_grip_position_m'] < 0.02 and metrics['max_relative_grip_angle_rad'] < 0.15, 'loaded_fingers': metrics['minimum_loaded_fingers_after_settle'] >= 4, 'contact_geometry': max(metrics['max_penetration_m'], metrics['max_robot_self_penetration_m']) < 0.002, 'no_forbidden_contact': metrics['forbidden_contact_steps'] == 0, 'joint_and_motor_limits': metrics['max_joint_limit_excess_rad'] < 0.02 and metrics['max_motor_force_excess_nm'] < 1e-10, 'complete_swing': metrics['max_backswing_angle_rad'] > 0.55 and metrics['min_followthrough_angle_rad'] < -0.7 and (metrics['peak_blade_speed_m_s'] > 1), 'returned_to_guard': metrics['recovery_guard_position_m'] < 0.03 and metrics['recovery_guard_angle_rad'] < 0.15 and (metrics['recovery_root_displacement_m'] < 0.05), 'settled_recovery': metrics['recovery_bat_speed_m_s'] < 0.1 and metrics['recovery_bat_angular_speed_rad_s'] < 0.2 and (metrics['recovery_joint_speed_rad_s'] < 0.2) and (metrics['recovery_root_speed_m_s'] < 0.1)}
    return ({**metrics, 'recovery_peaks': recovery_peaks, 'saturation_fraction': metrics['saturated_steps'] / steps if step_schedule is None else saturated_time / times[-1], 'max_motion_support_residual': float(controller.motion_support_residual.max()), 'checks': checks, 'passed': all(checks.values())}, np.asarray(states), np.asarray(controls), tactile)

def render(model, states, path, label, hand):
    model.vis.global_.offheight, model.vis.global_.offwidth = (720, 600)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.load_default(size=19)
    cameras = []
    for angle in (30, 140) if hand == 'right' else (-30, -140):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.1, 0, 0.75]
        camera.distance, camera.azimuth, camera.elevation = (2.8, angle, -10)
        cameras.append(camera)
    with mujoco.Renderer(model, height=720, width=600) as renderer, imageio.get_writer(path, fps=50, codec='libx264', macro_block_size=1) as writer:
        for index, state in enumerate(states):
            mujoco.mj_setState(model, data, state, FULL)
            mujoco.mj_forward(model, data)
            views = []
            for camera in cameras:
                renderer.update_scene(data, camera, scene_option=option)
                views.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1200, 60), fill='#152a22')
            draw.text((12, 6), f'G1 articulated swing | {label} | 1x | t={data.time:.2f}s', font=font)
            draw.text((12, 32), 'No-ball motor-control diagnostic; free robot and bat; not PPO', font=font)
            writer.append_data(np.asarray(frame))
            if index in (24, 74, 99, 124, 174, 249):
                frame.save(path.with_name(f'{path.stem}_{index:03}.png'))

def evaluate(output, capture, *, lock_root_height=False, reference_only=False, acceleration_weight=0.0, stance_lateral_offset=None):
    output.mkdir(parents=True, exist_ok=False)
    parent = ROOT / 'g1_cricket_results/articulated_stance'
    model = mujoco.MjModel.from_xml_path(str(parent / 'scene.xml'))
    times = np.arange(251) * 0.02
    rows, fits, stance_fits = ([], {}, {})
    inputs = [parent / 'scene.xml']
    if stance_lateral_offset is not None:
        grasp = ROOT / 'g1_cricket_results/articulated_grasp/summary.json'
        bench = json.loads(grasp.read_text())
        inputs.append(grasp)
    for hand in ('right', 'left'):
        if stance_lateral_offset is None:
            path = parent / f'{hand}_closed_250us.npz'
            inputs.append(path)
            with np.load(path) as saved:
                initial = saved['initial_qpos']
        else:
            solution = min(bench['fit_candidates'][hand], key=lambda r: r['residual_norm'])['solution']
            initial, stance_fits[hand] = fit_stance(model, hand, solution, lateral_offset=stance_lateral_offset)
        poses, fits[hand] = retarget_swing(model, initial, times, hand, lock_root_height=lock_root_height, acceleration_weight=acceleration_weight)
        np.savez_compressed(output / f'{hand}_reference.npz', qpos=poses, times=times)
        if reference_only:
            continue
        for inertial in (False, True):
            for dt in (0.00025, 0.000125):
                model.opt.timestep = dt
                label = f"{hand}_{('inertial' if inertial else 'gravity')}_{round(dt * 1000000.0)}us"
                result, states, controls, tactile = run_case(model, poses, times, inertial=inertial)
                result.update(case=label, hand=hand, inertial=inertial, timestep_s=dt)
                rows.append(result)
                np.savez_compressed(output / f'{label}.npz', states=states, controls=controls)
                (output / f'{label}_contacts.json').write_text(json.dumps(tactile) + '\n')
                if capture and dt == 0.000125:
                    render(model, states, output / f'{label}.mp4', label, hand)
                print(json.dumps(result), flush=True)
    sources = [Path(__file__), *[ROOT / f'src/unilab/tasks/manipulation/g1_cricket/{name}.py' for name in ('articulated_swing', 'articulated_batting', 'articulated_hands', 'bimanual', 'batting_dynamics', 'tracking')], ROOT / 'scripts/evaluate_g1_articulated_stance.py']
    report = {'scope': 'offline kinematic reference only; no physical rollout or learned cricket' if reference_only else 'complete no-ball PD swing and recovery, not learned cricket', 'lock_reference_root_height': lock_root_height, 'reference_acceleration_weight': acceleration_weight, 'stance_lateral_offset_m': stance_lateral_offset, 'stance_fit': stance_fits, 'reference_amplitude': SWING_AMPLITUDE, 'kinematic_fit': fits, 'rows': rows, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lock-root-height', action='store_true')
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--acceleration-weight', type=float, default=0.0)
    parser.add_argument('--stance-lateral-offset', type=float)
    args = parser.parse_args()
    if args.render and args.reference_only:
        parser.error('reference-only mode does not render physical motion')
    evaluate(args.output, args.render, lock_root_height=args.lock_root_height, reference_only=args.reference_only, acceleration_weight=args.acceleration_weight, stance_lateral_offset=args.stance_lateral_offset)
