"""Native one-control-interval counterfactuals, not executed bowling outcomes."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from .bowling_launch_control import LaunchController, TARGET_VELOCITY
from .twist2_cpu_probe import fingerprint
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
FLIGHT_FAILURES = {'no_release', 'first_bounce_outside_delivery_zone', 'not_exactly_one_bounce_before_target', 'target_corridor_missed'}

class LaunchLookahead:

    def __init__(self, model, hand):
        self.model, self.hand = (model, hand)
        self.data = mujoco.MjData(model)
        self.control = LaunchController(model, hand)
        self.steps = round(0.02 / model.opt.timestep)
        if not np.isclose(self.steps * model.opt.timestep, 0.02, rtol=0, atol=1e-12):
            raise ValueError('Native timestep must divide the 20 ms control interval')

    def predict(self, live, targets, impedance, feedforward, launch_torque, fraction):
        from .amp_bowling_probe import DeliveryMonitor
        if not 0 <= fraction <= 1:
            raise ValueError('Launch torque fraction must be in [0, 1]')
        m, d, c = (self.model, self.data, self.control)
        mujoco.mj_copyData(d, m, live)
        target, velocity = np.split(targets, 2)
        kp, kd = np.split(impedance, 2)
        monitor = DeliveryMonitor(m, self.hand)
        for _ in range(self.steps):
            torque = np.clip(kp * (target - d.qpos[c.q]) + kd * (velocity - d.qvel[c.v]) + feedforward, -c.caps, c.caps)
            if fraction == 1:
                torque[c.selected] = launch_torque
            elif fraction:
                torque[c.selected] += fraction * (launch_torque - torque[c.selected])
            d.ctrl[c.ids] = d.qpos[c.q] + (torque + c.kd * d.qvel[c.v]) / c.kp
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            metrics = monitor.observe(d)
            if d.warning.number.any() or not np.isfinite(np.r_[d.qpos, d.qvel, metrics]).all():
                raise RuntimeError('Invalid native lookahead dynamics')
        physical = monitor.events.finish(complete=True)
        failures = sorted(set(physical['failures']) - FLIGHT_FAILURES)
        state = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        ball_velocity = d.qvel[c.ball_v:c.ball_v + 3]
        return (state, dict(fraction=fraction, physics_steps=self.steps, physical_failures=failures, interval_physical_pass=not failures, terminal_ball_velocity_m_s=ball_velocity.tolist(), target_velocity_error_m_s=float(np.linalg.norm(ball_velocity - TARGET_VELOCITY)), maximum_joint_limit_excess_rad=physical['maximum_joint_limit_excess_rad'], maximum_actuator_limit_fraction=physical['maximum_actuator_limit_fraction'], minimum_pelvis_height_m=physical['minimum_pelvis_height_m'], minimum_pelvis_up=physical['minimum_pelvis_up'], robot_contact_details=monitor.robot_contacts))

    def choose(self, live, targets, impedance, feedforward, launch_torque):
        predictions = [self.predict(live, targets, impedance, feedforward, launch_torque, f) for f in FRACTIONS]
        safe = [i for i, (_, r) in enumerate(predictions) if r['interval_physical_pass']]
        best = min(safe, key=lambda i: predictions[i][1]['target_velocity_error_m_s']) if safe else 0
        diagnostics = np.array([[r['interval_physical_pass'], r['target_velocity_error_m_s'], *r['terminal_ball_velocity_m_s'], r['maximum_joint_limit_excess_rad'], max((c['max_penetration_m'] for c in r['robot_contact_details'].values()), default=0)] for _, r in predictions])
        return (np.array([FRACTIONS[best], len(safe)]), diagnostics, predictions[best][0])

def verify_lookahead_commands(trace, model, hand):
    predictor = LaunchLookahead(model, hand)
    live = mujoco.MjData(model)
    count = len(trace['states']) - 1
    assert trace['launch_selection_rows'].shape == (count, 2)
    assert trace['launch_lookahead_rows'].shape == (count, 5, 7)
    assert trace['launch_prediction_rows'].shape == (count, trace['states'].shape[1])
    active_count = 0
    for i, state in enumerate(trace['states'][:-1]):
        if not trace['launch_rows'][i, 0]:
            for key in ('launch_selection_rows', 'launch_lookahead_rows', 'launch_prediction_rows'):
                assert not np.any(trace[key][i])
            continue
        mujoco.mj_setState(model, live, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        live.eq_active[model.equality('ball_holder').id] = True
        live.qacc_warmstart[:] = trace['launch_warmstart_rows'][i]
        selection, diagnostics, predicted = predictor.choose(live, trace['targets'][i], trace['impedance_rows'][i], trace['feedforward_rows'][i], trace['launch_rows'][i, 1:4])
        np.testing.assert_array_equal(trace['launch_selection_rows'][i], selection)
        np.testing.assert_allclose(trace['launch_lookahead_rows'][i], diagnostics, rtol=0, atol=1e-09)
        np.testing.assert_allclose(trace['launch_prediction_rows'][i], predicted, rtol=0, atol=1e-09)
        np.testing.assert_allclose(trace['states'][i + 1], predicted, rtol=0, atol=1e-08)
        active_count += 1
    return active_count

def evaluate(folder):
    summary = json.loads((folder / 'summary.json').read_text())
    expected = {(h, dt) for h in ('right', 'left') for dt in (6.25e-05, 3.125e-05)}
    if not summary.get('model_launch') or len(summary['rows']) != 4 or {(r['hand'], r['timestep']) for r in summary['rows']} != expected:
        raise ValueError('Lookahead requires the complete four-case model-launch cohort')
    inputs = {'summary.json': fingerprint(folder / 'summary.json')}
    rows = []
    for result in summary['rows']:
        scene = folder / (result['hand'] + '.xml')
        trace_path = folder / (result['case'] + '.npz')
        for path in (scene, trace_path):
            value = fingerprint(path)
            if value != summary['artifacts'][path.name]:
                raise ValueError('Retained input fingerprint mismatch: ' + path.name)
            inputs[path.name] = value
        model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        model.opt.timestep = result['timestep']
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        predictor = LaunchLookahead(model, result['hand'])
        live = mujoco.MjData(model)
        intervals = []
        with np.load(trace_path) as trace:
            if np.any(trace['residual_rows']) or np.any(trace['released']):
                raise ValueError('This diagnostic requires held-ball, zero-residual traces')
            for index in np.flatnonzero(trace['launch_rows'][:, 0]):
                mujoco.mj_setState(model, live, trace['states'][index], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                live.eq_active[model.equality('ball_holder').id] = True
                live.qacc_warmstart[:] = trace['launch_warmstart_rows'][index]
                candidates = []
                for fraction in FRACTIONS:
                    terminal, candidate = predictor.predict(live, trace['targets'][index], trace['impedance_rows'][index], trace['feedforward_rows'][index], trace['launch_rows'][index, 1:4], fraction)
                    if fraction == 1:
                        error = float(np.max(np.abs(terminal - trace['states'][index + 1])))
                        np.testing.assert_allclose(terminal, trace['states'][index + 1], rtol=0, atol=1e-08)
                        candidate['retained_terminal_state_max_error'] = error
                    candidates.append(candidate)
                safe = [c for c in candidates if c['interval_physical_pass']]
                best = min(safe, key=lambda c: c['target_velocity_error_m_s']) if safe else None
                intervals.append(dict(control_index=int(index), time_s=float(trace['states'][index, 0]), candidates=candidates, best_interval_fraction=None if best is None else best['fraction']))
        rows.append(dict(case=result['case'], intervals=intervals))
    return dict(scope='All active launch intervals in all four retained cases; independent 20 ms counterfactuals, not a sequential policy rollout', promotion_allowed=False, learned_policy=False, limitations='No future control updates or release prediction; an interval pass does not prove later safety, legal flight or a recoverable trajectory', source_fingerprints={name: fingerprint(Path(__file__).with_name(name)) for name in ('bowling_launch_lookahead.py', 'bowling_launch_control.py', 'amp_bowling_probe.py')}, input_fingerprints=inputs, mujoco_version=mujoco.__version__, rows=rows)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.folder)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
