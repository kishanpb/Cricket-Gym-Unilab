"""Relocatable assets for a retained bowling checkpoint, without success claims."""
import argparse
import inspect
import json
from pathlib import Path
import shutil
from .amp_bowling_probe import rollout_case
from .portable_scenes import package_scenes
from .scene_distribution import checked_file, digest, read_contract
ENV_OPTIONS = ('release_curriculum', 'learn_release', 'allow_upward_release', 'flight_curriculum', 'one_bounce_flight', 'post_saturation_residual', 'bowling_arm_actions', 'freeze_approach', 'prepare_swing_residuals')
CASES = {(h, dt) for h in ('right', 'left') for dt in (6.25e-05, 3.125e-05)}

def export(checkpoint, scene_source, output, *, runtime_bundle):
    saved = json.loads((checkpoint / 'summary.json').read_text())
    parent = Path(saved['parent'])
    physical = json.loads((parent / 'summary.json').read_text())
    checked_file(parent, 'summary.json', saved['inputs'][str(parent / 'summary.json')])
    checked_file(checkpoint, 'final.pt', saved['checkpoint'])
    if len(saved['rows']) != 4 or {(r['hand'], r['timestep']) for r in saved['rows']} != CASES:
        raise ValueError('Checkpoint must retain the complete hand/resolution cohort')
    if physical['delivery_style'] != 'overarm':
        raise ValueError('This checkpoint task supports overarm delivery only')
    _, provenance = read_contract(scene_source)
    scenes = []
    references = []
    for hand in ('right', 'left'):
        scene = parent / f'{hand}.xml'
        checked_file(parent, scene.name, physical['artifacts'][scene.name])
        scenes.append((hand, scene))
        ref = Path(physical['arm_reference_directory']) / f'{hand}_dense_reference.npz'
        checked_file(ref.parent, ref.name, saved['inputs'][str(ref)])
        references.append(ref)
    packaged = package_scenes(scenes, output)
    assets = {}
    for name, expected in packaged['assets'].items():
        checked_file(scene_source, name, expected)
        source = provenance['assets'][name]
        if {key: source[key] for key in expected} != expected:
            raise ValueError('Checkpoint mesh differs from verified Unitree source')
        assets[name] = source
    provenance = dict(provenance, assets=assets)
    (output / 'unitree_assets.json').write_text(json.dumps(provenance, indent=2) + '\n')
    for name in ('UNITREE_LICENSE', 'UNILAB_LICENSE'):
        shutil.copyfile(scene_source / name, output / name)
    packaged.update(mesh_source_verified=True, mesh_source_provenance=digest((output / 'unitree_assets.json').read_bytes()))
    (output / 'manifest.json').write_text(json.dumps(packaged, indent=2) + '\n')
    for path in [*references, checkpoint / 'final.pt']:
        shutil.copyfile(path, output / path.name)
    excluded = {'groot_gather', 'arm_reference_directory', 'scene_file', 'retain', 'physics_backend', 'learn_release', 'allow_upward_release', 'post_saturation_residual'}
    options = {name: physical.get(name, param.default) for name, param in inspect.signature(rollout_case).parameters.items() if param.kind == inspect.Parameter.KEYWORD_ONLY and name not in excluded}
    amp_name, = [name for name in saved['inputs'] if name.endswith('/checkpoints/model_6200.pt')]
    groot = Path(physical['groot_gather'])
    runtime = json.loads((runtime_bundle / 'runtime_manifest.json').read_text())
    groot_files = runtime['upstream_files']
    for name, expected in groot_files.items():
        if saved['inputs'][str(groot / name)] != expected:
            raise ValueError('Portable controller differs from checkpoint training input')
    report = dict(schema_version=1, delivery_style='overarm', rollout_options=options, environment_options={name: saved.get(name, False) for name in ENV_OPTIONS}, amp_checkpoint=saved['inputs'][amp_name], groot_files=groot_files, checkpoint_source=digest((checkpoint / 'summary.json').read_bytes()), physical_parent=digest((parent / 'summary.json').read_bytes()), steps=saved['steps'], seed=saved['seed'], qualification=[{k: row[k] for k in ('case', 'hand', 'timestep', 'passed', 'failures')} for row in saved['rows']], scope='Retained SKRL PPO residual and release policy over frozen priors; not UniLab-trained skill', promotion_allowed=False, files={path.relative_to(output).as_posix(): digest(path.read_bytes()) for path in sorted(output.rglob('*')) if path.is_file()})
    (output / 'policy_manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report

def load(bundle, groot, amp):
    manifest = json.loads((bundle / 'policy_manifest.json').read_text())
    if manifest['schema_version'] != 1 or manifest['delivery_style'] != 'overarm':
        raise ValueError('Unsupported bowling checkpoint profile')
    flags = manifest['environment_options']
    if set(flags) != set(ENV_OPTIONS) or any((type(value) is not bool for value in flags.values())):
        raise ValueError('Invalid bowling environment options')
    for name, expected in manifest['files'].items():
        checked_file(bundle, name, expected)
    checked_file(amp, 'checkpoints/model_6200.pt', manifest['amp_checkpoint'])
    for name, expected in manifest['groot_files'].items():
        checked_file(groot, name, expected)
    scenes, _ = read_contract(bundle)
    for name, expected in scenes['assets'].items():
        checked_file(bundle, name, expected)
    for name in ('final.pt', 'right_dense_reference.npz', 'left_dense_reference.npz'):
        checked_file(bundle, name, manifest['files'][name])
    options = dict(manifest['rollout_options'], groot_gather=groot, arm_reference_directory=bundle)
    return (options, {hand: bundle / f'{hand}.xml' for hand in ('right', 'left')}, flags)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'scene_source', 'output', 'runtime_bundle'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = export(args.checkpoint, args.scene_source, args.output, runtime_bundle=args.runtime_bundle)
    print(f"Packaged all {len(report['qualification'])} checkpoint cases; promotion remains blocked")
