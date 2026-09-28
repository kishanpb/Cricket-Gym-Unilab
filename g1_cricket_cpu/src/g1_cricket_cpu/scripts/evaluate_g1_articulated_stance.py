"""Full floating-body grip evaluation; stationary control, not learned batting."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from g1_cricket_cpu.physics.articulated_batting import ArticulatedStanceController, fit_stance
from g1_cricket_cpu.physics.articulated_hands import hand_contact_state
from g1_cricket_cpu.physics.articulated_scene import build_articulated_scene
ROOT = Path(__file__).resolve().parents[1]
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS

def forbidden_contact(model, geoms, robot, feet, *, closing):
    bodies = model.geom_bodyid[geoms]
    names = [model.geom(int(g)).name for g in geoms]
    bat = model.body('cricket_bat').id
    if bat in bodies and any((b in robot for b in bodies)):
        return not ('bat_handle' in names and any(('_hand_' in n and n.endswith('_geom_1') for n in names)))
    if 'pitch' in names:
        return any((b in robot and b not in feet for b in bodies)) or (closing and bat in bodies)
    return any((n.startswith('wicket_') for n in names)) and any((b in robot or b == bat for b in bodies))

def run_case(model, pose, *, closing, duration=3.0, controller=None):
    data = mujoco.MjData(model)
    kinematics = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    control = ArticulatedStanceController(model, pose) if controller is None else controller
    finger_closed = control.finger_target.copy()
    finger_open = np.array([0, 0, 0.05, -0.05, -0.05, -0.05, -0.05])
    feet = [model.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')]
    wrists = [model.body(f'{s}_wrist_yaw_link').id for s in ('left', 'right')]
    bat = model.body('cricket_bat').id
    bat_q = model.joint('bat_free').qposadr[0]
    feet_initial = data.xpos[feet].copy()
    wrist_initial = data.xpos[wrists].copy()
    joint_ids = np.r_[model.actuator_trnid[control.actuators, 0], control.fingers.joints]
    joint_q = model.jnt_qposadr[joint_ids]
    joint_limits = model.jnt_range[joint_ids]
    state = np.empty(mujoco.mj_stateSize(model, FULL))
    states, controls, tactile = ([], [], [])
    metrics = dict(max_bat_translation_m=0.0, max_bat_rotation_rad=0.0, max_wrist_translation_m=0.0, max_foot_translation_m=0.0, max_root_tilt_rad=0.0, minimum_root_height_m=float(pose[2]), max_joint_limit_excess_rad=0.0, max_handle_penetration_m=0.0, max_robot_self_penetration_m=0.0, max_bat_robot_penetration_m=0.0, peak_loaded_slip_m_s=0.0, peak_finger_torque_nm=0.0, max_finger_torque_excess_nm=0.0, max_body_torque_excess_nm=0.0, forbidden_contact_count=0, minimum_loaded_fingers_after_settle=6)
    robot = {model.body('pelvis').id}
    for body in range(model.nbody):
        if model.body_parentid[body] in robot:
            robot.add(body)
    foot_bodies = set(feet)
    for step in range(round(duration / model.opt.timestep)):
        if not closing:
            phase = min(data.time / 0.5, 1.0)
            blend = phase * phase * (3 - 2 * phase)
            control.finger_target = (1 - blend) * finger_closed + blend * np.r_[finger_open, -finger_open]
        control.apply(data)
        velocity = data.qvel.copy()
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError('nonfinite full-body state')
        contacts = hand_contact_state(model, data, velocity=velocity)
        kinematics.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, kinematics)
        metrics['max_handle_penetration_m'] = max(metrics['max_handle_penetration_m'], max((-c['distance_m'] for c in contacts), default=0))
        for index, contact in enumerate(data.contact):
            bodies = model.geom_bodyid[contact.geom]
            if all((b in robot for b in bodies)):
                metrics['max_robot_self_penetration_m'] = max(metrics['max_robot_self_penetration_m'], -float(contact.dist))
            if bat in bodies and any((b in robot for b in bodies)):
                metrics['max_bat_robot_penetration_m'] = max(metrics['max_bat_robot_penetration_m'], -float(contact.dist))
            if forbidden_contact(model, contact.geom, robot, foot_bodies, closing=closing):
                wrench = np.empty(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                metrics['forbidden_contact_count'] += int(wrench[0] > 0.1)
        metrics['max_bat_translation_m'] = max(metrics['max_bat_translation_m'], float(np.linalg.norm(data.qpos[bat_q:bat_q + 3] - pose[bat_q:bat_q + 3])))
        angle = 2 * np.arccos(np.clip(abs(data.qpos[bat_q + 3:bat_q + 7] @ pose[bat_q + 3:bat_q + 7]), 0, 1))
        metrics['max_bat_rotation_rad'] = max(metrics['max_bat_rotation_rad'], float(angle))
        for name, ids, initial in (('wrist', wrists, wrist_initial), ('foot', feet, feet_initial)):
            metrics[f'max_{name}_translation_m'] = max(metrics[f'max_{name}_translation_m'], float(np.linalg.norm(kinematics.xpos[ids] - initial, axis=1).max()))
        metrics['minimum_root_height_m'] = min(metrics['minimum_root_height_m'], float(data.qpos[2]))
        tilt = np.empty(3)
        mujoco.mju_subQuat(tilt, data.qpos[3:7], pose[3:7])
        metrics['max_root_tilt_rad'] = max(metrics['max_root_tilt_rad'], float(np.linalg.norm(tilt[:2])))
        angles = data.qpos[joint_q]
        metrics['max_joint_limit_excess_rad'] = max(metrics['max_joint_limit_excess_rad'], float(np.maximum(joint_limits[:, 0] - angles, angles - joint_limits[:, 1]).max()))
        for name, actuators in (('body', control.actuators), ('finger', control.fingers.actuators)):
            caps = model.actuator_forcerange[actuators, 1] if name == 'body' else control.fingers.force_limits[:, 1]
            metrics[f'max_{name}_torque_excess_nm'] = max(metrics[f'max_{name}_torque_excess_nm'], float((np.abs(data.actuator_force[actuators]) - caps).max()))
        metrics['peak_finger_torque_nm'] = max(metrics['peak_finger_torque_nm'], float(np.abs(data.actuator_force[control.fingers.actuators]).max()))
        if data.time > 0.2:
            loaded = {(c['geom'].split('_hand_')[0], c['geom'].split('_hand_')[1].split('_')[0]) for c in contacts if c['normal_force_n'] > 0.1 and any((p in c['geom'] for p in ('thumb', 'index', 'middle')))}
            metrics['minimum_loaded_fingers_after_settle'] = min(metrics['minimum_loaded_fingers_after_settle'], len(loaded))
            metrics['peak_loaded_slip_m_s'] = max(metrics['peak_loaded_slip_m_s'], max((c['tangential_slip_m_s'] for c in contacts if c['normal_force_n'] > 0.1), default=0.0))
        if (step + 1) % round(0.02 / model.opt.timestep) == 0:
            mujoco.mj_getState(model, data, state, FULL)
            states.append(state.copy())
            controls.append(data.ctrl.copy())
            tactile.append({'contact_time_s': float(data.time - model.opt.timestep), 'contacts': contacts})
    checks = {'upright': metrics['minimum_root_height_m'] > 0.6 and metrics['max_root_tilt_rad'] < 0.35, 'retained_bat': metrics['max_bat_translation_m'] < 0.02 and metrics['max_bat_rotation_rad'] < 0.1, 'planted_feet': metrics['max_foot_translation_m'] < 0.03, 'wrist_stability': metrics['max_wrist_translation_m'] < 0.02, 'joint_limits': metrics['max_joint_limit_excess_rad'] < 0.02, 'contact_geometry': max(metrics['max_handle_penetration_m'], metrics['max_robot_self_penetration_m'], metrics['max_bat_robot_penetration_m']) < 0.002, 'no_forbidden_contact': metrics['forbidden_contact_count'] == 0, 'torque_caps': max(metrics['max_body_torque_excess_nm'], metrics['max_finger_torque_excess_nm']) < 1e-10, 'loaded_fingers': metrics['minimum_loaded_fingers_after_settle'] >= 4}
    return ({**metrics, 'checks': checks, 'passed': bool(closing and all(checks.values())), 'support_residual': control.support_residual}, np.asarray(states), np.asarray(controls), tactile)

def render(model, states, path, label):
    model.vis.global_.offheight = 720
    model.vis.global_.offwidth = 600
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 19)
    with mujoco.Renderer(model, height=720, width=600) as renderer, imageio.get_writer(path, fps=50, codec='libx264', macro_block_size=1) as writer:
        for index, state in enumerate(states):
            mujoco.mj_setState(model, data, state, FULL)
            mujoco.mj_forward(model, data)
            views = []
            for lookat, distance, azimuth in (([0, 0, 0.72], 2.7, 30), ([0.03, 0, 0.85], 1.0, 140)):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = lookat
                camera.distance, camera.azimuth, camera.elevation = (distance, azimuth, -12)
                renderer.update_scene(data, camera, scene_option=option)
                views.append(renderer.render().copy())
            frame = Image.fromarray(np.concatenate(views, axis=1))
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1200, 60), fill='#152a22')
            draw.text((12, 6), f'G1 full-body grip | {label} | 1x | t={data.time:.2f}s', font=font)
            draw.text((12, 32), 'Free robot and bat; stationary PD + support feedforward; not PPO', font=font)
            writer.append_data(np.asarray(frame))
            if index == 49:
                frame.save(path.with_suffix('.png'))

def evaluate(output, grasp, capture):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'src/unilab/assets/robots/g1/g1.xml'
    assets = ROOT / 'src/unilab/assets/robots/g1_hands'
    path = output / 'scene.xml'
    build_articulated_scene(source, assets, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    bench = json.loads(grasp.read_text())
    rows, fits = ([], {})
    for hand in ('right', 'left'):
        solution = min(bench['fit_candidates'][hand], key=lambda r: r['residual_norm'])['solution']
        pose, fits[hand] = fit_stance(model, hand, solution)
        for closing in (True, False):
            for dt in (0.00025, 0.000125):
                model.opt.timestep = dt
                label = f"{hand}_{('closed' if closing else 'open')}_{round(dt * 1000000.0)}us"
                row, states, controls, tactile = run_case(model, pose, closing=closing)
                row.update(case=label, handedness=hand, closing=closing, timestep_s=dt)
                rows.append(row)
                np.savez_compressed(output / f'{label}.npz', initial_qpos=pose, states=states, controls=controls)
                (output / f'{label}_contacts.json').write_text(json.dumps(tactile) + '\n')
                if capture and dt == 0.000125:
                    render(model, states, output / f'{label}.mp4', label)
                print(json.dumps(row), flush=True)
    sources = [Path(__file__), *[ROOT / f'src/unilab/tasks/manipulation/g1_cricket/{n}.py' for n in ('articulated_batting', 'articulated_hands', 'articulated_scene', 'tracking', 'prior', 'scene')]]
    summary = {'scope': 'full floating-body stationary hold; not learned grasp, swing or boundary batting', 'input_grasp_sha256': hashlib.sha256(grasp.read_bytes()).hexdigest(), 'fits': fits, 'rows': rows, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, 'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--grasp', type=Path, default=ROOT / 'g1_cricket_results/articulated_grasp/summary.json')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    evaluate(args.output, args.grasp, args.render)
