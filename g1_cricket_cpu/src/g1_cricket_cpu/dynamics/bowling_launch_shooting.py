"""Bounded native shooting screen; replayed support commands are not a policy."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import mujoco
import numpy as np
from .amp_bowling_probe import DeliveryMonitor, native_target_bounds
from .bowling_launch_control import LaunchController, TARGET_VELOCITY, launch_ready
from .bowling_launch_lookahead import FLIGHT_FAILURES
from .bowling_native_release_window import opportunity, window_summary, COLUMNS
from .bowling_release_window import first_bounce, projection_model
from .groot_batting_ppo import verify_inputs
from .twist2_cpu_probe import fingerprint
KNOTS = np.array([1.4, 1.5, 1.75, 1.95, 2.1])
POSITION_OFFSET_RAD = 0.35

def launch_cost(position, velocity, shoulder_z, elbow_z, gravity, height, objective):
    geometry = 20 * max(0.0, shoulder_z + 0.12 - position[2]) + 20 * max(0.0, shoulder_z - elbow_z)
    if objective == 'velocity':
        return float(np.linalg.norm(velocity - TARGET_VELOCITY) + geometry)
    if objective != 'first_bounce':
        raise ValueError('Unknown launch objective')
    if position[2] <= height:
        return float(100 + geometry + height - position[2])
    _, point = first_bounce(position, velocity, gravity, height)
    return float(abs(point[0] - 12) + 2 * abs(point[1]) + geometry + 2 * max(0.0, 6 - velocity[0]) + 2 * max(0.0, abs(velocity[1]) - 2))

def shoulder_mask(axis):
    if axis not in {'all', 'pitch', 'yaw'}:
        raise ValueError('Unknown shoulder axis')
    return np.array([axis in {'all', 'pitch'}, axis == 'all', axis in {'all', 'yaw'}])

def shoulder_schedule(phase, parameters, hand, axis='all'):
    values = np.asarray(parameters, dtype=float)
    if values.shape != (3, 3) or not np.isfinite(values).all() or np.abs(values).max() > 1:
        raise ValueError('Expected nine finite normalized shoulder commands in [-1, 1]')
    values = np.vstack((np.zeros(3), values, np.zeros(3)))
    command = np.array([np.interp(phase, KNOTS, values[:, i]) for i in range(3)])
    if hand == 'left':
        command *= [1, -1, -1]
    elif hand != 'right':
        raise ValueError('Unknown hand')
    command[~shoulder_mask(axis)] = 0
    return command

def shoulder_torque(baseline, command, caps, blend, mode, axis='all'):
    mask = shoulder_mask(axis)
    if mode == 'correction':
        torque = np.clip(baseline + command, -caps, caps)
    elif mode == 'absolute':
        torque = np.clip((1 - blend) * baseline + command, -caps, caps)
    else:
        raise ValueError('Unknown shoulder control mode')
    torque[~mask] = np.clip(baseline[~mask], -caps[~mask], caps[~mask])
    return torque

def shoulder_targets(reference, offset, bounds):
    return native_target_bounds(reference + offset, bounds)

class ShootingCase:

    def __init__(self, parent, warmstart_parent, row, *, objective='velocity', control_mode='correction', shoulder_axis='all'):
        self.row, self.hand = (row, row['hand'])
        if objective not in {'velocity', 'first_bounce'}:
            raise ValueError('Unknown launch objective')
        self.objective = objective
        if control_mode not in {'correction', 'absolute', 'position_offset'}:
            raise ValueError('Unknown shoulder control mode')
        self.control_mode = control_mode
        shoulder_mask(shoulder_axis)
        self.shoulder_axis = shoulder_axis
        self.model = mujoco.MjModel.from_xml_path(str((parent / (self.hand + '.xml')).resolve()))
        self.model.opt.timestep = row['timestep']
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.control = LaunchController(self.model, self.hand)
        self.shoulder_bounds = self.model.jnt_range[self.model.actuator_trnid[self.control.ids[self.control.selected], 0]]
        self.live, self.data = (mujoco.MjData(self.model), mujoco.MjData(self.model))
        with np.load(parent / (row['case'] + '.npz')) as saved:
            self.trace = {k: saved[k].copy() for k in ('states', 'swing_rows', 'targets', 'impedance_rows', 'feedforward_rows')}
            if saved['released'].any() or saved['residual_rows'].any():
                raise ValueError('Shooting requires unreleased zero-residual parent traces')
        phase = self.trace['swing_rows'][:, 0]
        self.start = int(np.flatnonzero(phase >= KNOTS[0])[0])
        self.end = int(np.flatnonzero(phase >= KNOTS[-1])[0])
        with np.load(warmstart_parent / (row['case'] + '.npz')) as saved:
            np.testing.assert_array_equal(saved['states'][self.start], self.trace['states'][self.start])
            warmstart = saved['launch_warmstart_rows'][self.start]
        mujoco.mj_setState(self.model, self.live, self.trace['states'][self.start], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.live.eq_active[self.model.equality('ball_holder').id] = True
        self.live.qacc_warmstart[:] = warmstart
        self.steps = round(0.02 / self.model.opt.timestep)
        if not np.isclose(self.steps * self.model.opt.timestep, 0.02, rtol=0, atol=1e-12):
            raise ValueError('Native timestep must divide the control interval')
        self.ball_q = int(self.model.joint('ball_free').qposadr[0])
        self.gravity, self.height = projection_model(self.model)

    def predict(self, parameters, *, replay=False, reference=False):
        if reference and np.any(parameters):
            raise ValueError('Reference replay requires zero parameters')
        mode = 'correction' if reference else self.control_mode
        m, d, c = (self.model, self.data, self.control)
        mujoco.mj_copyData(d, m, self.live)
        monitor = DeliveryMonitor(m, self.hand)
        state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        states, commands, samples, shoulder_target_rows = ([], [], [], [])
        trigger_samples = 0
        record_trigger = not reference and (mode == 'position_offset' or self.shoulder_axis != 'all')
        best_cost, best_state = (float('inf'), None)
        max_support = 0.0
        for index in range(self.start, self.end):
            target, velocity = np.split(self.trace['targets'][index], 2)
            kp, kd = np.split(self.trace['impedance_rows'][index], 2)
            correction = shoulder_schedule(self.trace['swing_rows'][index, 0], parameters, self.hand, self.shoulder_axis)
            correction *= POSITION_OFFSET_RAD if mode == 'position_offset' else c.caps[c.selected]
            if mode == 'position_offset':
                target = target.copy()
                target[c.selected] = shoulder_targets(target[c.selected], correction, self.shoulder_bounds)
                shoulder_target_rows.append(target[c.selected].copy())
            blend = np.interp(self.trace['swing_rows'][index, 0], KNOTS, [0, 1, 1, 1, 0])
            commands.append(correction)
            for _ in range(self.steps):
                torque = np.clip(kp * (target - d.qpos[c.q]) + kd * (velocity - d.qvel[c.v]) + self.trace['feedforward_rows'][index], -c.caps, c.caps)
                if mode != 'position_offset':
                    torque[c.selected] = shoulder_torque(torque[c.selected], correction, c.caps[c.selected], blend, mode, self.shoulder_axis)
                d.ctrl[c.ids] = d.qpos[c.q] + (torque + c.kd * d.qvel[c.v]) / c.kp
                mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                metrics = monitor.observe(d)
                max_support = max(max_support, float(metrics[8]))
                if d.warning.number.any() or not np.isfinite(np.r_[d.qpos, d.qvel, metrics]).all():
                    raise RuntimeError('Invalid shooting dynamics')
                position = d.qpos[self.ball_q:self.ball_q + 3]
                velocity_ball = d.qvel[c.ball_v:c.ball_v + 3]
                shoulder, elbow, _ = d.xpos[monitor.arm]
                if record_trigger:
                    trigger_samples += launch_ready(position, velocity_ball, shoulder, elbow, self.gravity, self.height)
                cost = launch_cost(position, velocity_ball, shoulder[2], elbow[2], self.gravity, self.height, self.objective)
                if cost < best_cost:
                    best_cost = float(cost)
                    best_state = dict(time_s=float(d.time), position_m=position.tolist(), velocity_m_s=velocity_ball.tolist())
                sample = opportunity(d.time, position, velocity_ball, shoulder[2], elbow[2], self.gravity, self.height)
                if sample is not None:
                    samples.append(sample)
            mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            if replay:
                np.testing.assert_allclose(state, self.trace['states'][index + 1], rtol=0, atol=1e-08)
            states.append(state.copy())
        physical = monitor.events.finish(complete=True)
        failures = sorted(set(physical['failures']) - FLIGHT_FAILURES)
        if max_support:
            failures.append('artificial_support')
        contact_depth = max((r['max_penetration_m'] for r in monitor.robot_contacts.values()), default=0.0)
        penalty = 100 * len(failures) + 100 * physical['maximum_joint_limit_excess_rad'] + 1000 * contact_depth
        samples = np.asarray(samples).reshape(-1, len(COLUMNS))
        result = dict(case=self.row['case'], window_physical_pass=not failures, physical_failures=failures, robot_contact_details=monitor.robot_contacts, maximum_joint_limit_excess_rad=physical['maximum_joint_limit_excess_rad'], maximum_actuator_limit_fraction=physical['maximum_actuator_limit_fraction'], max_artificial_support=max_support, minimum_pelvis_height_m=physical['minimum_pelvis_height_m'], minimum_pelvis_up=physical['minimum_pelvis_up'], objective=best_cost + penalty, **{f'best_{self.objective}_cost': best_cost}, best_state=best_state, native_steps=len(states) * self.steps, **window_summary(samples))
        command_key = {'correction': 'shoulder_correction_nm', 'absolute': 'shoulder_command_nm', 'position_offset': 'shoulder_offset_rad'}[mode]
        arrays = dict(states=np.asarray(states), **{command_key: np.asarray(commands)}, opportunities=samples)
        if record_trigger:
            result['model_launch_trigger_samples'] = trigger_samples
        if mode == 'position_offset':
            arrays['shoulder_targets_rad'] = np.asarray(shoulder_target_rows)
        return (result, arrays)

def rank_results(results):
    return max((row['objective'] for row in results))

def run(parent, warmstart_parent, output, *, seed=29, generations=3, population=16, objective='velocity', control_mode='correction', shoulder_axis='all'):
    if generations < 1 or population < 4:
        raise ValueError('Expected at least one generation and four candidates')
    saved = json.loads((parent / 'summary.json').read_text())
    warm = json.loads((warmstart_parent / 'summary.json').read_text())
    cases = [(h, dt) for h in ('right', 'left') for dt in (6.25e-05, 3.125e-05)]
    if [(r['hand'], r['timestep']) for r in saved['rows']] != cases:
        raise ValueError('Expected the complete ordered parent cohort')
    inputs, source_changes = ({}, {})
    for name, old in warm['inputs'].items():
        path = Path(name)
        current = fingerprint(path)
        if current != old:
            if path.suffix not in {'.py', '.md'}:
                raise ValueError('Non-source parent input changed: ' + name)
            source_changes[name] = dict(parent=old, current=current)
        inputs[name] = current
    for root, report in ((parent, saved), (warmstart_parent, warm)):
        for hand, dt in cases:
            for name in (hand + '.xml', f'{hand}_dt{dt:g}.npz'):
                path = root / name
                actual = fingerprint(path)
                if actual != report['artifacts'][name]:
                    raise ValueError('Parent artifact changed: ' + str(path))
                inputs[str(path)] = actual
        inputs[str(root / 'summary.json')] = fingerprint(root / 'summary.json')
    sources = list(Path('integrations/g1_dynamics').glob('*.py'))
    sources += [Path('tests/test_bowling_launch_shooting.py'), Path('runs/unilab_contact_telemetry/scripts/g1_cricket_delivery_trial.py')]
    inputs.update({str(p): fingerprint(p) for p in sources})
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(inputs):
            path = Path(name)
            if path.suffix in {'.py', '.md'}:
                archive.write(path, str(path.resolve().relative_to(Path.cwd())))
    started = time.monotonic()
    predictors = [ShootingCase(parent, warmstart_parent, row, objective=objective, control_mode=control_mode, shoulder_axis=shoulder_axis) for row in saved['rows']]
    zeros = np.zeros((3, 3))
    baseline = [case.predict(zeros, replay=True, reference=True)[0] for case in predictors]
    print(json.dumps(dict(baseline_replay=baseline)), flush=True)
    coarse = [predictors[0], predictors[2]]
    rng = np.random.default_rng(seed)
    mean, std = (zeros.copy(), np.full((3, 3), 0.35))
    history = []
    best_parameters, best_objective = (zeros.copy(), rank_results([baseline[0], baseline[2]]))
    if control_mode == 'absolute':
        best_objective = float('inf')
    for generation in range(generations):
        proposals = np.clip(rng.normal(mean, std, (population, 3, 3)), -1, 1)
        proposals[0], proposals[1] = (best_parameters, zeros)
        proposals[:, :, ~shoulder_mask(shoulder_axis)] = 0
        ranked = []
        for index, parameters in enumerate(proposals):
            results = [case.predict(parameters)[0] for case in coarse]
            score = rank_results(results)
            history.append(dict(generation=generation, index=index, parameters=parameters.tolist(), objective=score, rows=results))
            ranked.append((score, index))
            if score < best_objective:
                best_objective, best_parameters = (score, parameters.copy())
        elite = proposals[[index for _, index in sorted(ranked)[:max(2, population // 4)]]]
        mean, std = (elite.mean(axis=0), np.maximum(elite.std(axis=0), 0.05))
        print(json.dumps(dict(generation=generation, best_objective=best_objective)), flush=True)
    rows = []
    for case in predictors:
        result, arrays = case.predict(best_parameters)
        trace = output / (case.row['case'] + '.npz')
        np.savez_compressed(trace, **arrays)
        rows.append(dict(**result, trace=dict(name=trace.name, **fingerprint(trace))))
    verify_inputs(inputs)
    report = dict(scope='Native launch-window shooting with frozen parent support/reference commands; not closed-loop execution or learning', limitation='Held ball, replayed support commands and optimistic ballistic projection; no actual flight or full-episode recovery qualification', parent=str(parent), warmstart_parent=str(warmstart_parent), inputs=inputs, source_changes_since_parent=source_changes, seed=seed, generations=generations, population=population, knots=KNOTS.tolist(), launch_objective=objective, control_mode=control_mode, shoulder_axis=shoulder_axis, active_parameter_count=9 if shoulder_axis == 'all' else 3, **dict(position_offset_max_rad=POSITION_OFFSET_RAD, position_offset_velocity_rule='Preserve reference velocity targets; position bias only') if control_mode == 'position_offset' else {}, baseline_scope='Unmodified reference control, independent of search control mode', baseline=baseline, history=history, parameters=best_parameters.tolist(), rows=rows, decision='requires_full_closed_loop_release_trial' if all((r['window_physical_pass'] and r['projected_zone_samples'] and (control_mode != 'position_offset' and shoulder_axis == 'all' or r['model_launch_trigger_samples']) for r in rows)) else 'no_complete_cohort_launch_candidate', elapsed_s=time.monotonic() - started, promotion_allowed=False, artifacts={p.name: fingerprint(p) for p in output.iterdir() if p.is_file()})
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(report['decision'], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent', 'warmstart_parent', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--population', type=int, default=16)
    parser.add_argument('--objective', choices=('velocity', 'first_bounce'), default='velocity')
    parser.add_argument('--control-mode', choices=('correction', 'absolute', 'position_offset'), default='correction')
    parser.add_argument('--shoulder-axis', choices=('all', 'pitch', 'yaw'), default='all')
    args = parser.parse_args()
    run(args.parent, args.warmstart_parent, args.output, seed=args.seed, generations=args.generations, population=args.population, objective=args.objective, control_mode=args.control_mode, shoulder_axis=args.shoulder_axis)
