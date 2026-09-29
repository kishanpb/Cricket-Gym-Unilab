"""Generate a bounded joint-space bowling sweep, not an executed motion."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.interpolate import BPoly, PchipInterpolator
from .twist2_cpu_probe import fingerprint

def transition(time, start, end):
    x = np.clip((np.asarray(time) - start) / (end - start), 0, 1)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)

def sweep_positions(times, original, names, hand, *, duration=0.2, cocked_pitch=-3.0, cocking_start=1.0, elbow_angle=1.3, sweep_pitch=-1.2, sweep_knot_speed=0.0, wrist_flick=0.0, shoulder_yaw_sweep=0.0, shoulder_roll=0.6, shoulder_yaw_delay=0.0, shoulder_yaw_preload=None):
    result = original.copy()
    shift = duration - 0.2
    envelope = transition(times, cocking_start, 1.5) * (1 - transition(times, 2.2 + shift, 2.7))
    elbow_envelope = transition(times, 0.0, 0.7) * (1 - transition(times, 2.2 + shift, 2.7))
    pitch = cocked_pitch + (sweep_pitch - cocked_pitch) * transition(times, 1.7, 1.9 + shift) + (0.4 - sweep_pitch) * transition(times, 1.9 + shift, 2.2 + shift)
    if sweep_knot_speed:
        knots = [1.7, 1.9 + shift, 2.2 + shift]
        continuous = BPoly.from_derivatives(knots, [[cocked_pitch, 0, 0], [sweep_pitch, sweep_knot_speed, 0], [0.4, 0, 0]])
        active = (times >= knots[0]) & (times <= knots[-1])
        pitch = np.where(active, continuous(np.clip(times, knots[0], knots[-1])), pitch)
    for index, name in enumerate(names):
        if not name.startswith(hand + '_'):
            continue
        value = 0.0
        weight = envelope
        if 'shoulder_pitch' in name:
            value = pitch
        elif 'shoulder_roll' in name:
            value = -shoulder_roll if hand == 'right' else shoulder_roll
        elif 'shoulder_yaw' in name and shoulder_yaw_sweep:
            direction = 1 if hand == 'right' else -1
            yaw_time = times - shoulder_yaw_delay
            value = direction * shoulder_yaw_sweep * (0.25 + 0.75 * transition(yaw_time, 1.7, 1.9 + shift)) * (1 - transition(yaw_time, 1.9 + shift, 2.1 + shift))
            if shoulder_yaw_preload is not None:
                value += direction * (shoulder_yaw_preload - 0.25 * shoulder_yaw_sweep) * (1 - transition(yaw_time, 1.7, 1.9 + shift)) * (1 - transition(yaw_time, 1.9 + shift, 2.1 + shift))
        elif 'elbow' in name:
            value = elbow_angle
            weight = elbow_envelope
        elif 'wrist_yaw' in name and wrist_flick:
            direction = 1 if hand == 'right' else -1
            value = direction * wrist_flick * (2 * transition(times, 1.7, 1.9 + shift) - 1)
        result[:, index] += weight * (value - original[:, index])
    return result

def generate(source, scenes, output, *, duration=0.2, cocked_pitch=-3.0, cocking_start=1.0, elbow_angle=1.3, sweep_pitch=-1.2, sweep_knot_speed=0.0, wrist_flick=0.0, shoulder_yaw_sweep=0.0, shoulder_roll=0.6, delivery_style='overarm', shoulder_yaw_delay=0.0, shoulder_yaw_preload=None):
    if delivery_style not in {'overarm', 'underarm'}:
        raise ValueError('Unknown delivery style')
    pitch_bounds = (-3.0, -2.5) if delivery_style == 'overarm' else (0.3, 1.0)
    if not 0.1 <= duration <= 0.5 or not pitch_bounds[0] <= cocked_pitch <= pitch_bounds[1]:
        raise ValueError('sweep duration/pitch outside the declared design range')
    if not 0 <= cocking_start < 1.5:
        raise ValueError('cocking must start before its fixed 1.5-second endpoint')
    if not 0.3 <= elbow_angle <= 1.3:
        raise ValueError('elbow angle outside the declared design range')
    if not -1.4 <= sweep_pitch <= -0.6:
        raise ValueError('sweep pitch outside the declared design range')
    max_knot_speed = 2.5 * min((sweep_pitch - cocked_pitch) / duration, (0.4 - sweep_pitch) / 0.3) if delivery_style == 'overarm' else 0
    if not 0 <= sweep_knot_speed <= max_knot_speed:
        raise ValueError('sweep knot speed must preserve monotone Bernstein controls')
    if not 0 <= wrist_flick <= 0.8:
        raise ValueError('wrist flick outside the declared design range')
    if not 0 <= shoulder_yaw_sweep <= 2.0:
        raise ValueError('shoulder yaw sweep outside the declared design range')
    if not 0.25 <= shoulder_roll <= 0.8:
        raise ValueError('shoulder roll outside the declared design range')
    if not 0 <= shoulder_yaw_delay <= 0.1 or (shoulder_yaw_delay and (not shoulder_yaw_sweep)):
        raise ValueError('shoulder yaw delay must be in [0, 0.1] and requires a yaw sweep')
    if shoulder_yaw_preload is not None and (not -0.5 <= shoulder_yaw_preload <= 0.5 or not shoulder_yaw_sweep):
        raise ValueError('shoulder yaw preload must be in [-0.5, 0.5] and requires a yaw sweep')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    inputs = {str(Path(__file__)): fingerprint(Path(__file__))}
    for hand in ('right', 'left'):
        path = source / f'{hand}_dense_reference.npz'
        scene = scenes / f'{hand}.xml'
        inputs.update({str(p): fingerprint(p) for p in (path, scene)})
        model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        joints = [i for i in range(model.njnt) if any((part in model.joint(i).name for part in ('shoulder', 'elbow', 'wrist')))]
        names = [model.joint(i).name for i in joints]
        q = model.jnt_qposadr[joints]
        with np.load(path) as data:
            times = data['times'].copy()
            poses = data['qpos'].copy()
        poses[:, q] = sweep_positions(times, poses[:, q], names, hand, duration=duration, cocked_pitch=cocked_pitch, cocking_start=cocking_start, elbow_angle=elbow_angle, sweep_pitch=sweep_pitch, sweep_knot_speed=sweep_knot_speed, wrist_flick=wrist_flick, shoulder_yaw_sweep=shoulder_yaw_sweep, shoulder_roll=shoulder_roll, shoulder_yaw_delay=shoulder_yaw_delay, shoulder_yaw_preload=shoulder_yaw_preload)
        interpolator = PchipInterpolator(times, poses[:, q], axis=0)
        check_times = np.arange(0, 2.7001, 0.001)
        sampled = interpolator(check_times)
        bounds = model.jnt_range[joints]
        assert (sampled >= bounds[:, 0] - 1e-09).all()
        assert (sampled <= bounds[:, 1] + 1e-09).all()
        bowling = np.array([n.startswith(hand + '_') for n in names])
        destination = output / path.name
        np.savez_compressed(destination, times=times, qpos=poses)
        rows.append(dict(hand=hand, maximum_sampled_rate_rad_s=float(np.abs(interpolator(check_times, nu=1)[:, bowling]).max()), maximum_sampled_acceleration_rad_s2=float(np.abs(interpolator(check_times, nu=2)[:, bowling]).max()), joint_range_pass=True, artifact=fingerprint(destination)))
    summary = dict(scope='Offline arm targets only; no executed dynamics, learned policy or promotion', delivery_style=delivery_style, inputs=inputs, duration_s=duration, cocked_pitch_rad=cocked_pitch, cocking_start_s=cocking_start, elbow_angle_rad=elbow_angle, sweep_pitch_rad=sweep_pitch, sweep_knot_speed_rad_s=sweep_knot_speed, wrist_flick_rad=wrist_flick, shoulder_yaw_sweep_rad=shoulder_yaw_sweep, shoulder_roll_rad=shoulder_roll, shoulder_yaw_delay_s=shoulder_yaw_delay, shoulder_yaw_preload_rad=shoulder_yaw_preload, rows=rows, promotion_allowed=False)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'scenes', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--duration', type=float, default=0.2)
    parser.add_argument('--cocked-pitch', type=float, default=-3.0)
    parser.add_argument('--cocking-start', type=float, default=1.0)
    parser.add_argument('--elbow-angle', type=float, default=1.3)
    parser.add_argument('--sweep-pitch', type=float, default=-1.2)
    parser.add_argument('--sweep-knot-speed', type=float, default=0.0)
    parser.add_argument('--wrist-flick', type=float, default=0.0)
    parser.add_argument('--shoulder-yaw-sweep', type=float, default=0.0)
    parser.add_argument('--shoulder-roll', type=float, default=0.6)
    parser.add_argument('--shoulder-yaw-delay', type=float, default=0.0)
    parser.add_argument('--shoulder-yaw-preload', type=float)
    parser.add_argument('--delivery-style', choices=('overarm', 'underarm'), default='overarm')
    args = parser.parse_args()
    generate(args.source, args.scenes, args.output, duration=args.duration, cocked_pitch=args.cocked_pitch, cocking_start=args.cocking_start, elbow_angle=args.elbow_angle, sweep_pitch=args.sweep_pitch, sweep_knot_speed=args.sweep_knot_speed, wrist_flick=args.wrist_flick, shoulder_yaw_sweep=args.shoulder_yaw_sweep, shoulder_roll=args.shoulder_roll, shoulder_yaw_delay=args.shoulder_yaw_delay, shoulder_yaw_preload=args.shoulder_yaw_preload, delivery_style=args.delivery_style)
