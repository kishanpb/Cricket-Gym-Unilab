"""Run retained overarm/underarm controllers from the portable cricket package."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from integrations.g1_dynamics.portable_batting import verify_bundle, verify_imports
from integrations.g1_dynamics.scene_distribution import checked_file, digest


def case_keys():
    return [(style, hand, dt) for style in ("overarm", "underarm")
            for hand in ("right", "left") for dt in (0.0000625, 0.00003125)]


def evaluate_job(job):
    bundle, groot, amp, scenes, output, style, hand, dt, manifest, backend = job
    import torch
    from integrations.g1_dynamics.amp_bowling_probe import run_case
    from integrations.g1_dynamics.amp_running_probe import load_actor

    torch.set_num_threads(1)
    options = dict(manifest["bowling"][style]["options"])
    options.update(groot_gather=groot, physics_backend=backend,
                   arm_reference_directory=bundle / "data" / style,
                   scene_file=scenes / f"bowling_{hand}.xml")
    row = run_case(load_actor(amp), bundle / "data", output / style, hand, dt, **options)
    return style, row, verify_imports(bundle, groot, manifest)


def run(bundle, groot, amp, scenes, output, *, case=None, workers=2, backend="mujoco"):
    if workers not in (1, 2):
        raise ValueError("Use one or two CPU workers")
    selected = case_keys() if case is None else [case]
    if any(key not in case_keys() for key in selected):
        raise ValueError("Case must belong to the complete two-style bowling cohort")
    manifest = verify_bundle(bundle, groot, scenes)
    checked_file(amp, "checkpoints/model_6200.pt", manifest["amp_checkpoint"])
    for name, expected in manifest["bowling_scenes"].items():
        checked_file(scenes, name, expected)
    output.mkdir(parents=True, exist_ok=False)
    for style in {key[0] for key in selected}:
        (output / style).mkdir()
    jobs = [(bundle, groot, amp, scenes, output, style, hand, dt, manifest, backend)
            for style, hand, dt in selected]
    rows, imported = {style: [] for style in ("overarm", "underarm")}, {}

    def collect(results):
        for style, row, sources in results:
            rows[style].append(row)
            imported.update(sources)
            print(f"Finished {style}/{row['case']}", flush=True)

    if workers == 1:
        collect(map(evaluate_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            collect(pool.map(evaluate_job, jobs))
    report = dict(scope="complete_cohort" if case is None else "relocation_smoke_not_cohort",
                  backend=backend, workers=workers, styles=rows, imported_sources=imported,
                  runtime_manifest=digest((bundle / "runtime_manifest.json").read_bytes()),
                  provenance="Frozen AMP/GR00T locomotion, reference arms, mechanical holder; not learned bowling arms",
                  promotion_allowed=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "groot", "amp", "scenes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--backend", choices=("mujoco", "mjbatch"), default="mujoco")
    parser.add_argument("--case", nargs=3, metavar=("STYLE", "HAND", "DT"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.bundle.resolve(), args.groot.resolve(), args.amp.resolve(), args.scenes.resolve(),
        args.output.resolve(), workers=args.workers, backend=args.backend,
        case=None if args.case is None else (args.case[0], args.case[1], float(args.case[2])))
