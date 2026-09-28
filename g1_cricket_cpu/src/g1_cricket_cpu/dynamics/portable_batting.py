"""Run the packaged PPO/reference batting controller with explicit external assets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
from g1_cricket_cpu.dynamics.scene_distribution import checked_file, digest, read_contract

def verify_bundle(bundle, upstream, scenes):
    manifest = json.loads((bundle / 'runtime_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        checked_file(bundle, name, expected)
    for name, expected in manifest['upstream_files'].items():
        checked_file(upstream, name, expected)
    for name, expected in manifest['runtime_versions'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'Runtime version mismatch: {name}')
    scene_manifest, _ = read_contract(scenes)
    checked_file(scenes, 'batting.xml', manifest['scene'])
    for name, expected in scene_manifest['assets'].items():
        checked_file(scenes, name, expected)
    return manifest

def verify_imports(bundle, upstream, manifest):
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name not in manifest['module_files'] and (not name.startswith(('integrations.', 'unilab', 'g1_cricket_cpu.scripts.', 'decoupled_wbc'))):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        root = upstream if name.startswith('decoupled_wbc') else bundle
        expected = manifest['upstream_files'] if root == upstream else manifest['files']
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f'Runtime imported source outside its package: {name}')
        relative = path.relative_to(root.resolve()).as_posix()
        if relative not in expected:
            raise ValueError(f'Unrecorded imported source: {name}')
        checked_file(root, relative, expected[relative])
        imported[name] = relative
    return imported

def evaluate_job(job):
    bundle, upstream, scenes, output, hand, feed, options, manifest = job
    from g1_cricket_cpu.dynamics.groot_batting_evaluate import evaluate_case
    row = evaluate_case((upstream, bundle / 'data', scenes, bundle / 'data/final.pt', output, hand, feed, True, options), scene_file=scenes / 'batting.xml')
    return (row, verify_imports(bundle, upstream, manifest))

def run(bundle, upstream, scenes, output, *, backend='mujoco', case=None, workers=2):
    if workers not in (1, 2):
        raise ValueError('Use one or two CPU workers')
    manifest = verify_bundle(bundle, upstream, scenes)
    from g1_cricket_cpu.dynamics.groot_batting_evaluate import case_keys, cohort_metrics
    verify_imports(bundle, upstream, manifest)
    selected = case_keys() if case is None else [case]
    if any((key not in case_keys() for key in selected)):
        raise ValueError('Case must belong to the complete two-hand feed cohort')
    output.mkdir(parents=True, exist_ok=False)
    options = dict(manifest['environment_options'], physics_backend=backend)
    options['reference_files'] = [bundle / path for path in manifest['references']]
    jobs = [(bundle, upstream, scenes, output, hand, feed, options, manifest) for hand, feed in selected]
    rows, imported = ([], {})

    def collect(results):
        for row, sources in results:
            rows.append(row)
            imported.update(sources)
            print(f"Finished {row['case']}: qualified_hit={row['qualified_hit']}", flush=True)
    if workers == 1:
        collect(map(evaluate_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            collect(pool.map(evaluate_job, jobs))
    report = dict(scope='complete_cohort' if case is None else 'relocation_smoke_not_cohort', backend=backend, workers=workers, rows=rows, imported_sources=imported, runtime_manifest=digest((bundle / 'runtime_manifest.json').read_bytes()), metrics=cohort_metrics(rows) if case is None else None, provenance='Fixed PPO residual, frozen GR00T balance, reference swing; not new training')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('scenes', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--backend', choices=('mujoco', 'mjbatch'), default='mujoco')
    parser.add_argument('--case', nargs=2, metavar=('HAND', 'FEED'))
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.bundle.resolve(), args.upstream.resolve(), args.scenes.resolve(), args.output.resolve(), backend=args.backend, workers=args.workers, case=None if args.case is None else (args.case[0], int(args.case[1])))
