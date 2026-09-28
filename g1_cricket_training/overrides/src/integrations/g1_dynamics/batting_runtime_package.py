"""Snapshot the imported live batting runtime without external models or meshes."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys

from integrations.g1_dynamics.scene_distribution import digest


def imported_sources(root, checkout):
    roots = (root / "integrations", checkout / "src/unilab", checkout / "scripts")
    targets = (Path("src/integrations"), Path("src/unilab"), Path("src/scripts"))
    sources = {}
    for name, module in tuple(sys.modules.items()):
        if name == __name__:
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        for base, target in zip(roots, targets, strict=True):
            if path.is_relative_to(base.resolve()) and path.is_file():
                sources[target / path.relative_to(base.resolve())] = path
    return sources


def bowling_inputs(root, amp_upstream):
    importlib.import_module("integrations.g1_dynamics.portable_bowling")
    importlib.import_module("integrations.g1_dynamics.portable_bowling_ppo")
    from integrations.g1_dynamics.amp_bowling_probe import rollout_case
    from integrations.g1_dynamics.amp_running_probe import load_actor

    load_actor(amp_upstream)
    sources, styles = {}, {}
    parents = dict(overarm="amp_bowling_synchronized_arms_v23", underarm="amp_bowling_underarm_loft_v26")
    parameters = inspect.signature(rollout_case).parameters
    excluded = {"groot_gather", "arm_reference_directory", "physics_backend", "retain", "scene_file", "learn_release"}
    for style, folder in parents.items():
        parent_path = root / "runs/external_g1" / folder / "summary.json"
        parent = json.loads(parent_path.read_text())
        keys = [(r["hand"], r["timestep"]) for r in parent["rows"]]
        if len(keys) != 4 or set(keys) != {(h, dt) for h in ("right", "left") for dt in (0.0000625, 0.00003125)}:
            raise ValueError("Bowling parent must contain the complete bilateral resolution cohort")
        options = {name: parent.get(name, parameter.default) for name, parameter in parameters.items()
                   if parameter.kind == inspect.Parameter.KEYWORD_ONLY and name not in excluded}
        if options["delivery_style"] != style:
            raise ValueError("Bowling parent style differs from the requested controller")
        styles[style] = dict(options=options, parent_summary=digest(parent_path.read_bytes()))
        for hand in ("right", "left"):
            name = f"{hand}_dense_reference.npz"
            reference = Path(parent["arm_reference_directory"]) / name
            if digest((root / reference).read_bytes()) != parent["inputs"][str(reference)]:
                raise ValueError("Bowling reference differs from retained parent")
            sources[Path("data") / style / name] = root / reference
    return sources, styles


def export(root, checkout, upstream, output, *, amp_upstream=None):
    importlib.import_module("integrations.g1_dynamics.portable_batting")
    from integrations.g1_dynamics.groot_articulated_probe import CONFIG, load_policy
    from integrations.g1_dynamics.groot_batting_evaluate import case_keys

    load_policy(upstream)
    extra, bowling = ({}, {}) if amp_upstream is None else bowling_inputs(root, amp_upstream)
    sources = imported_sources(root, checkout)
    source_names = {path.resolve(): str(name) for name, path in sources.items()}
    modules = {name: source_names[Path(module.__file__).resolve()]
               for name, module in tuple(sys.modules.items())
               if getattr(module, "__file__", None)
               and Path(module.__file__).resolve() in source_names}
    refs = []
    for hand in ("right", "left"):
        original = f"g1_cricket_results/articulated_swing_compact/{hand}_reference.npz"
        sources[Path("data") / original] = checkout / original
        ref = f"data/{hand}_height-0.04.npz"
        sources[Path(ref)] = root / f"runs/external_g1/groot_height_screen_v1/{hand}_height-0.04.npz"
        refs.append(ref)
    sources[Path("data/final.pt")] = root / "runs/external_g1/groot_whole_body_ppo_long_v1/final.pt"
    sources[Path("UNILAB_LICENSE")] = checkout / "LICENSE"
    sources[Path("GYM_CRICKET_LICENSE")] = root / "LICENSE"
    sources.update(extra)
    external = {Path(CONFIG): upstream / CONFIG}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("decoupled_wbc") and filename:
            path = Path(filename).resolve()
            external[path.relative_to(upstream.resolve())] = path
    for name in ("Balance", "Walk"):
        path = Path(CONFIG).parent / "policy" / f"GR00T-WholeBodyControl-{name}.onnx"
        external[path] = upstream / path
    versions = {name: importlib.metadata.version(name) for name in
                ("mujoco", "numpy", "torch", "onnxruntime", "scipy", "skrl", "gymnasium",
                 "unisim-core", "hydra-core", "omegaconf", "tensordict", "rsl-rl-lib")}
    output.mkdir(parents=True, exist_ok=False)
    for relative, source in sources.items():
        (output / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output / relative)
    manifest = dict(
        scope="Batting runtime snapshot; not full fork integration or physical success",
        files={str(name): digest(path.read_bytes()) for name, path in sorted(sources.items())},
        module_files=modules,
        upstream_files={str(name): digest(path.read_bytes()) for name, path in sorted(external.items())},
        runtime_versions=versions, references=refs,
        scene=digest((root / "release/g1_scene_source/batting.xml").read_bytes()),
        environment_options=dict(whole_body=True, condition_timing=True, swing_power=0.65),
        case_keys=case_keys(),
    )
    if amp_upstream is not None:
        manifest.update(bowling=bowling,
                        amp_checkpoint=digest((amp_upstream / "checkpoints/model_6200.pt").read_bytes()),
                        bowling_scenes={f"bowling_{hand}.xml": digest(
                            (root / f"release/g1_scene_source/bowling_{hand}.xml").read_bytes())
                            for hand in ("right", "left")})
    (output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = Path(__file__).parent / "batting_environment"
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(environment / name, output / name)
    shutil.copyfile(Path(__file__).with_name("controller_installation.py"), output / "install_controllers.py")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--amp-checkout", type=Path)
    args = parser.parse_args()
    report = export(args.root.resolve(), args.checkout.resolve(), args.upstream.resolve(), args.output,
                    amp_upstream=None if args.amp_checkout is None else args.amp_checkout.resolve())
    print(f"Packaged {len(report['files'])} runtime files; external models remain separate.")
