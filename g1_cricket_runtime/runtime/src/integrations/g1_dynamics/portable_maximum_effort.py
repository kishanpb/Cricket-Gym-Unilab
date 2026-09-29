"""Reproduce the bounded maximum-effort checks with packaged cricket assets."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from .portable_batting import verify_bundle, verify_imports
from .scene_distribution import checked_file, digest


def evaluate_job(job):
    bundle, groot, amp, scenes, folder, key, maximum, backend, manifest, shifts = job
    import torch

    torch.set_num_threads(1)
    if key[0] == "batting":
        from .batting_maximum_effort import MaximumEffortBattingEnv
        from .groot_batting_evaluate import evaluate_case

        _, hand, feed = key
        options = dict(manifest["environment_options"], physics_backend=backend,
                       maximum_arm_effort=maximum)
        options["reference_files"] = [bundle / name for name in manifest["references"]]
        row = evaluate_case(
            (groot, bundle / "data", scenes, bundle / "data/final.pt", folder,
             hand, feed, True, options), scene_file=scenes / "batting.xml",
            env_class=MaximumEffortBattingEnv,
        )
    else:
        from .amp_bowling_probe import run_case
        from .amp_running_probe import load_actor

        style, hand, dt = key
        options = dict(manifest["bowling"][style]["options"])
        options["start_x"] += shifts[hand]
        options.update(groot_gather=groot, physics_backend=backend,
                       arm_reference_directory=bundle / "data" / style,
                       scene_file=scenes / f"bowling_{hand}.xml",
                       maximum_arm_effort=maximum)
        row = run_case(load_actor(amp), bundle / "data", folder, hand, dt, **options)
    return row, verify_imports(bundle, groot, manifest)


def run(bundle, groot, amp, scenes, output, *, task="all", maximum=True,
        workers=2, backend="mujoco", case=None):
    manifest = verify_bundle(bundle, groot, scenes)
    experiment = json.loads((bundle / "data/maximum_effort.json").read_text())
    checked_file(amp, "checkpoints/model_6200.pt", manifest["amp_checkpoint"])
    for name, expected in manifest["bowling_scenes"].items():
        checked_file(scenes, name, expected)
    batting = [("batting", hand, feed) for hand in ("right", "left") for feed in range(7)]
    bowling = [(style, hand, dt) for style in ("overarm", "underarm")
               for hand in ("right", "left") for dt in (0.0000625, 0.00003125)]
    selected = batting + bowling if task == "all" else batting if task == "batting" else bowling
    if case is not None:
        if case not in selected:
            raise ValueError("Diagnostic case must belong to the selected complete cohort")
        selected = [case]
    output.mkdir(parents=True, exist_ok=False)
    for style in {key[0] for key in selected}:
        (output / style).mkdir()
    jobs = [(bundle, groot, amp, scenes, output / key[0], key, maximum, backend,
             manifest, experiment["approach_shifts_m"]) for key in selected]
    rows, imported = [], {}
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for key, (row, sources) in zip(selected, pool.map(evaluate_job, jobs)):
            rows.append(dict(task=key[0], result=row))
            imported.update(sources)
            print(f"Finished {key[0]}/{row['case']}", flush=True)
    result = dict(scope="complete_cohort" if case is None else "diagnostic_not_cohort",
                  task=task, maximum_arm_effort=maximum, backend=backend, rows=rows,
                  imported_sources=imported, approach_shifts_m=experiment["approach_shifts_m"],
                  runtime_manifest=digest((bundle / "runtime_manifest.json").read_bytes()),
                  promotion_allowed=False, hardware_ceiling_claim=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "groot", "amp", "scenes", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--task", choices=("all", "batting", "bowling"), default="all")
    parser.add_argument("--original-effort", action="store_true")
    parser.add_argument("--backend", choices=("mujoco", "mjbatch"), default="mujoco")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--case", nargs=3, metavar=("STYLE", "HAND", "FEED_OR_DT"))
    args = parser.parse_args()
    case = None if args.case is None else (
        args.case[0], args.case[1],
        int(args.case[2]) if args.case[0] == "batting" else float(args.case[2]))
    run(*(getattr(args, name).resolve() for name in ("bundle", "groot", "amp", "scenes", "output")),
        task=args.task, maximum=not args.original_effort, backend=args.backend,
        workers=args.workers, case=case)
