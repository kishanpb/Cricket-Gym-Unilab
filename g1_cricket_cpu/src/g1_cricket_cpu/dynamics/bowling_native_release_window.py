"""Read-only native-step release opportunities in the retained held-ball run."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import multiprocessing
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from .amp_bowling_probe import load_actor, rollout_case
from .bowling_release_window import first_bounce, projection_model
from .groot_batting_ppo import verify_inputs
from .twist2_cpu_probe import fingerprint
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import RETURN_Y, TARGET_POPPING_X
COLUMNS = ('time_s', 'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s', 'shoulder_z_m', 'elbow_z_m', 'flight_time_s', 'bounce_x_m', 'bounce_y_m', 'bounce_z_m')

def opportunity(time_s, position, velocity, shoulder_z, elbow_z, gravity, height):
    if not (position[2] > shoulder_z + 0.12 and elbow_z > shoulder_z and (velocity[0] > 6) and (abs(velocity[1]) < 2) and (position[2] > height)):
        return None
    duration, point = first_bounce(position, velocity, gravity, height)
    return np.r_[time_s, position, velocity, shoulder_z, elbow_z, duration, point]

def window_summary(samples):
    if samples.ndim != 2 or samples.shape[1] != len(COLUMNS) or (not np.isfinite(samples).all()):
        raise ValueError('Expected finite native opportunity rows')
    if len(samples) and (not np.all(np.diff(samples[:, 0]) > 0)):
        raise ValueError('Opportunity times must strictly increase')
    in_zone = (samples[:, 10] > 4) & (samples[:, 10] < TARGET_POPPING_X) & (abs(samples[:, 11]) < RETURN_Y)
    return dict(eligible_samples=len(samples), projected_zone_samples=int(in_zone.sum()), maximum_projected_bounce_x_m=float(samples[:, 10].max()) if len(samples) else None, positive_vertical_samples=int((samples[:, 6] > 0).sum()), first_zone_time_s=float(samples[in_zone, 0][0]) if in_zone.any() else None)

class NativeWindow:

    def __init__(self, model, monitor):
        self.monitor = monitor
        joint = model.joint('ball_free')
        self.q, self.v = (int(joint.qposadr[0]), int(joint.dofadr[0]))
        self.gravity, self.height = projection_model(model)
        self.original_observe = monitor.observe
        self.inspected = 0
        self.rows = []

    def observe(self, data):
        result = self.original_observe(data)
        self.inspected += 1
        shoulder, elbow, _ = data.xpos[self.monitor.arm]
        row = opportunity(data.time, data.qpos[self.q:self.q + 3], data.qvel[self.v:self.v + 3], shoulder[2], elbow[2], self.gravity, self.height)
        if row is not None:
            self.rows.append(row)
        return result

def replay(job):
    upstream, unilab, parent, output, row, options = job
    torch.set_num_threads(1)
    actor = load_actor(upstream)
    generator = rollout_case(actor, unilab, output, row['hand'], row['timestep'], scene_file=parent / (row['hand'] + '.xml'), retain=False, **options)
    model, _, monitor = next(generator)
    observer = NativeWindow(model, monitor)
    monitor.observe = observer.observe
    while True:
        try:
            next(generator)
        except StopIteration as finished:
            result, records = finished.value
            break
    assert result == {k: v for k, v in row.items() if k != 'trace'}
    original = parent / (row['case'] + '.npz')
    assert fingerprint(original) == row['trace']
    with np.load(original) as saved:
        assert set(records) == set(saved.files)
        for key, value in records.items():
            np.testing.assert_array_equal(value, saved[key], err_msg=key)
        assert not saved['released'].any()
        assert observer.inspected == len(saved['metrics'])
    samples = np.asarray(observer.rows).reshape(-1, len(COLUMNS))
    target = output / (row['case'] + '_window.npz')
    np.savez_compressed(target, opportunities=samples)
    report = dict(case=row['case'], native_samples_inspected=observer.inspected, arrays_exact=len(records), result_exact=True, opportunities=dict(path=str(target), **fingerprint(target)), **window_summary(samples))
    print(json.dumps(report), flush=True)
    return report

def run(upstream, unilab, parent, source_run, output):
    source = json.loads((source_run / 'summary.json').read_text())
    verify_inputs(source['inputs'])
    saved = json.loads((parent / 'summary.json').read_text())
    expected = {(h, dt) for h in ('right', 'left') for dt in (6.25e-05, 3.125e-05)}
    assert len(saved['rows']) == 4 and {(r['hand'], r['timestep']) for r in saved['rows']} == expected
    assert saved['minimum_release_speed'] == 100 and all((r['release'] is None for r in saved['rows']))
    inputs = dict(source['inputs'])
    for hand in ('right', 'left'):
        name = hand + '.xml'
        assert fingerprint(parent / name) == saved['artifacts'][name]
    for row in saved['rows']:
        assert fingerprint(parent / (row['case'] + '.npz')) == row['trace']
    extra = [Path(__file__), Path('tests/test_bowling_native_release_window.py'), Path('reports/g1_bowling_native_release_window_plan.md'), parent / 'summary.json', source_run / 'summary.json', source_run / 'source_bundle.zip']
    extra += [parent / (hand + '.xml') for hand in ('right', 'left')]
    extra += [parent / (r['case'] + '.npz') for r in saved['rows']]
    with zipfile.ZipFile(source_run / 'source_bundle.zip') as archive:
        sources = {name: archive.read(name) for name in archive.namelist()}
    for path in extra:
        name = str(path.resolve().relative_to(Path.cwd()))
        inputs[name] = fingerprint(path)
        if path.suffix in ('.py', '.md'):
            sources[name] = path.read_bytes()
    recorded = {Path(n).resolve() for n in inputs}
    assert (upstream / 'checkpoints/model_6200.pt').resolve() in recorded
    assert (unilab / 'scripts/g1_cricket_delivery_trial.py').resolve() in recorded
    options = {name: saved[name] for name, parameter in inspect.signature(rollout_case).parameters.items() if parameter.kind == inspect.Parameter.KEYWORD_ONLY and name in saved}
    for name in ('groot_gather', 'arm_reference_directory'):
        options[name] = Path(options[name])
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(sources.items()):
            archive.writestr(name, raw)
    started = time.monotonic()
    jobs = [(upstream, unilab, parent, output, row, options) for row in saved['rows']]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(replay, jobs))
    verify_inputs(inputs)
    result = dict(scope='Native-step necessary first-bounce screen; no release, trajectory change or performance promotion', optimistic_exclusions='No descending-velocity or swing-clock filter; release-induced contacts and foot legality are not checked by projection', columns=COLUMNS, rows=rows, inputs=inputs, source_archive=fingerprint(output / 'sources.zip'), wall_seconds=time.monotonic() - started, decision='needs_actual_release_evaluation' if any((r['projected_zone_samples'] for r in rows)) else 'no_native_sampled_timing_only_first_bounce_solution', promotion_allowed=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['decision'], flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'unilab', 'parent', 'source_run', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.upstream, args.unilab, args.parent, args.source_run, args.output)
