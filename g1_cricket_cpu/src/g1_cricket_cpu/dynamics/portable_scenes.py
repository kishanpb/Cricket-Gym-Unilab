"""Package flattened STL-based G1 scenes without changing their compiled physics."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .twist2_cpu_probe import fingerprint

def compare_models(original, relocated):
    arrays = scalars = 0
    for name in dir(original):
        if name.startswith('_') or name.endswith('_pathadr') or name in ('npaths', 'nbuffer'):
            continue
        value = getattr(original, name)
        other = getattr(relocated, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=name)
            arrays += 1
        elif isinstance(value, (int, float)):
            assert value == other, name
            scalars += 1
    assert original.names == relocated.names
    for group in ('opt', 'stat'):
        for name in dir(getattr(original, group)):
            if name.startswith('_'):
                continue
            value = getattr(getattr(original, group), name)
            other = getattr(getattr(relocated, group), name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other, err_msg=f'{group}.{name}')
                arrays += 1
            elif isinstance(value, (int, float)):
                assert value == other, f'{group}.{name}'
                scalars += 1
    return dict(arrays_exact=arrays, scalars_exact=scalars, names_exact=True)

def scene_inputs(source):
    tree = ET.parse(source)
    root = tree.getroot()
    if root.findall('.//include'):
        raise ValueError('Flatten scene includes before packaging')
    compiler = root.find('compiler')
    options = {} if compiler is None else compiler.attrib
    meshdir = source.parent / options.get('meshdir', options.get('assetdir', '.'))
    assets = []
    for element in root.iter():
        if 'file' not in element.attrib:
            continue
        path = Path(element.get('file'))
        if element.tag != 'mesh' or path.suffix.lower() != '.stl':
            raise ValueError('Only flattened scenes with STL file assets are supported')
        if options.get('strippath', 'false') == 'true':
            path = Path(path.name)
        resolved = (meshdir / path).resolve()
        assets.append((element, resolved, fingerprint(resolved)))
    return (tree, assets)

def package_scenes(scenes, output):
    if len({name for name, _ in scenes}) != len(scenes):
        raise ValueError('Scene labels must be unique')
    if any((not name or Path(name).name != name or name in ('.', '..') for name, _ in scenes)):
        raise ValueError('Scene labels must be plain filenames')
    prepared = [(name, source, *scene_inputs(source)) for name, source in scenes]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'assets').mkdir()
    manifest = dict(scope='Relocatable scene/mesh packaging only; not controller or release validation', mujoco_version=mujoco.__version__, scenes={}, assets={}, generator=fingerprint(Path(__file__)), mesh_source_verified=False)
    for name, source, tree, assets in prepared:
        root = tree.getroot()
        compiler = root.find('compiler')
        if compiler is not None:
            for key in ('meshdir', 'assetdir', 'texturedir', 'strippath'):
                compiler.attrib.pop(key, None)
        for element, path, digest in assets:
            element.set('name', element.get('name', Path(element.get('file')).stem))
            relative = f"assets/{digest['sha256']}.stl"
            if relative not in manifest['assets']:
                shutil.copyfile(path, output / relative)
                assert fingerprint(output / relative) == digest
                manifest['assets'][relative] = digest
            element.set('file', relative)
        target = output / f'{name}.xml'
        ET.indent(tree, space='  ')
        tree.write(target, encoding='utf-8', xml_declaration=True)
        proof = compare_models(mujoco.MjModel.from_xml_path(str(source.resolve())), mujoco.MjModel.from_xml_path(str(target.resolve())))
        manifest['scenes'][target.name] = dict(source=fingerprint(source), packaged=fingerprint(target), asset_references=len(assets), compiled_equivalence=proof)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--scene', action='append', required=True, metavar='LABEL=XML')
    args = parser.parse_args()
    sources = [(label, Path(path)) for label, path in (s.split('=', 1) for s in args.scene)]
    report = package_scenes(sources, args.output)
    print(json.dumps(dict(scenes=len(report['scenes']), unique_assets=len(report['assets']))))
