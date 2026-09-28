"""Source-bound CPU bowling PPO from portable assets, with full-cohort preflight."""

import argparse
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .amp_bowling_env import AmpBowlingEnv, CASES
from .amp_bowling_probe import rollout_case
from .amp_bowling_ppo import evaluate, optimize
from .groot_batting_ppo import make_agent, verify_inputs
from .mjbatch_stepper import backend_inputs
from .portable_batting import verify_bundle, verify_imports
from .scene_distribution import checked_file, read_contract
from .twist2_cpu_probe import fingerprint


def make_env(bundle, groot, amp, scenes, manifest, style, backend):
    options = dict(manifest["bowling"][style]["options"],
                   groot_gather=groot, arm_reference_directory=bundle / "data" / style)
    return AmpBowlingEnv(
        amp, bundle / "data", bundle / "data", physics_backend=backend,
        rollout_options=options,
        scene_files={hand: scenes / f"bowling_{hand}.xml" for hand in ("right", "left")},
    )


def preflight(raw):
    rows = []
    try:
        for hand, dt in CASES:
            reference = rollout_case(
                raw.actor, raw.unilab, raw.parent, hand, dt, retain=False,
                scene_file=raw.scene_files[hand], physics_backend="mujoco", **raw.options,
            )
            while True:
                try:
                    next(reference)
                except StopIteration as finished:
                    result, expected = finished.value
                    break
            raw.reset(options={"hand": hand, "timestep": dt})
            terminal = False
            while not terminal:
                observation, reward, terminal, truncated, _ = raw.step(np.zeros(29))
                assert raw.observation_space.contains(observation) and np.isfinite(reward)
                assert not truncated
            assert raw.result == result
            for name, value in expected.items():
                assert np.isfinite(value).all(), name
                np.testing.assert_array_equal(raw.records[name], value, err_msg=name)
            assert raw.records["metrics"][:, 7].max() <= 1 + 1e-9
            assert raw.records["metrics"][:, 8].max() == 0
            rows.append(dict(result=result, compared_keys=list(expected), states_exact=True,
                             physics_samples=len(expected["metrics"])))
            print(f"Preflight exact: {result['case']}", flush=True)
    finally:
        raw.close()
    return rows


def input_files(bundle, groot, amp, scenes, manifest, backend):
    scene_manifest, _ = read_contract(scenes)
    groups = {
        "runtime": (bundle, [*manifest["files"], "runtime_manifest.json", "pyproject.toml", "uv.lock"]),
        "groot": (groot, manifest["upstream_files"]),
        "amp": (amp, ["checkpoints/model_6200.pt"]),
        "scenes": (scenes, [*scene_manifest["assets"], *manifest["bowling_scenes"],
                            "manifest.json", "unitree_assets.json", "UNITREE_LICENSE"]),
    }
    files = {f"{label}/{name}": folder / name
             for label, (folder, names) in groups.items() for name in names}
    if backend == "mjbatch":
        files.update({f"backend/{i}_{path.name}": path for i, path in enumerate(backend_inputs())})
    return files


def run(bundle, groot, amp, scenes, output, *, style="overarm", backend="mujoco", steps=400, seed=0):
    if style not in ("overarm", "underarm") or backend not in ("mujoco", "mjbatch"):
        raise ValueError("Expected overarm/underarm and mujoco/mjbatch")
    if steps <= 0 or steps % 400:
        raise ValueError("Budget must be a positive multiple of 400 PPO transitions")
    manifest = verify_bundle(bundle, groot, scenes)
    checked_file(amp, "checkpoints/model_6200.pt", manifest["amp_checkpoint"])
    for name, expected in manifest["bowling_scenes"].items():
        checked_file(scenes, name, expected)
    verify_imports(bundle, groot, manifest)
    files = input_files(bundle, groot, amp, scenes, manifest, backend)
    inputs = {str(path): fingerprint(path) for path in files.values()}
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            if path.suffix in {".py", ".xml", ".md", ".json", ".toml", ".lock", ".cpp", ".h"}:
                archive.write(path, arcname=name)
    torch.set_num_threads(1)
    started = time.monotonic()
    rows = preflight(make_env(bundle, groot, amp, scenes, manifest, style, backend))
    verify_inputs(inputs)
    verify_imports(bundle, groot, manifest)
    proof = dict(status="zero_residual_full_cohort_parity_passed", backend=backend,
                 delivery_style=style, rows=rows, promotion_allowed=False)
    (output / "preflight.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")

    raw = make_env(bundle, groot, amp, scenes, manifest, style, backend)
    try:
        agent, training, probes = optimize(raw, steps, seed)
        agent.save(str(output / "final.pt"))
        restored, _ = make_agent(raw, steps, log_std=-2.0)
        restored.load(str(output / "final.pt"))
        with torch.no_grad():
            torch.testing.assert_close(
                agent.policy.compute({"observations": probes})[0],
                restored.policy.compute({"observations": probes})[0], rtol=0, atol=0,
            )
        evaluated = evaluate(restored, raw, output)
    finally:
        raw.close()
    verify_inputs(inputs)
    imported = verify_imports(bundle, groot, manifest)
    report = dict(
        scope="Portable CPU PPO connection; full delivery qualification remains required",
        backend=backend, delivery_style=style, steps=steps, seed=seed, inputs=inputs,
        source_paths={name: str(path) for name, path in files.items()}, imported_sources=imported,
        training=training, rows=evaluated, checkpoint_action_roundtrip_exact=True,
        elapsed_s=time.monotonic() - started, promotion_allowed=False,
        provenance="PPO torque residual, frozen AMP/GR00T, reference arms and mechanical ball holder",
        artifacts={p.name: fingerprint(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "groot", "amp", "scenes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--style", choices=("overarm", "underarm"), default="overarm")
    parser.add_argument("--backend", choices=("mujoco", "mjbatch"), default="mujoco")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(*(getattr(args, name).resolve() for name in ("bundle", "groot", "amp", "scenes", "output")),
        style=args.style, backend=args.backend, steps=args.steps, seed=args.seed)
