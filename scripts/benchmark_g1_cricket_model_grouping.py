"""Check frozen G1 control replay and time exact-model grouping, not policy quality."""

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import mujoco
import numpy as np
from evaluate_g1_cricket_first_step import ROOT, evaluation_override
from mjbatch.held_control import HeldControlRollout
from mujoco.rollout import Rollout
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.tasks.manipulation.g1_cricket.bowling import HOLDER_SENSORS


def held_inputs(model, trace, indices, nstep):
    spec = int(mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE)
    values = np.concatenate(
        (trace["controls"][indices], np.ones((len(indices), model.neq))), axis=1
    )
    return (
        trace["state"][indices],
        np.broadcast_to(values[:, None], (len(indices), nstep, values.shape[1])),
        spec,
    )


def compare(recorder, models, data, initial, control, spec, nstep, sensor_indices=None):
    options = {} if sensor_indices is None else {"sensor_indices": sensor_indices}
    return recorder.rollout(
        models, [data], initial, control, control_spec=spec, nstep=nstep, **options
    )


def check_arrays(actual, expected):
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.shape == reference.shape and observed.dtype == reference.dtype
        for row in range(len(observed)):
            assert np.isfinite(observed[row]).all()
            assert np.array_equal(observed[row], reference[row])


def benchmark(source, output, compact=False):
    registry.ensure_registries()
    hashes, rows = {}, []
    inputs = [Path(__file__), ROOT / "src/unilab/base/mujoco_substeps.py"]
    import mjbatch.held_control as held_module

    held_path = Path(held_module.__file__)
    for hand in ("right", "left"):
        directory = source / f"ppo_{hand}"
        config = directory / "run_config.json"
        inputs.append(config)
        owner = OmegaConf.create(json.loads(config.read_text())["config"])
        for dt in (0.0000625, 0.00003125):
            env = registry.make(
                owner.training.task_name,
                num_envs=1,
                sim_backend="mujoco",
                env_cfg_override=evaluation_override(owner, dt),
            )
            try:
                model = env.get_playback_model()
                nstep = env.cfg.sim_substeps
                compiled = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
                mujoco.mj_saveModel(model, buffer=compiled)
                models = [copy.copy(model) for _ in range(8)]
                data = mujoco.MjData(model)
                columns = np.concatenate(
                    [
                        np.arange(
                            model.sensor(name).adr[0],
                            model.sensor(name).adr[0] + model.sensor(name).dim[0],
                        )
                        for name in HOLDER_SENSORS
                    ]
                )
                group_modes = (True, True) if compact else (False, True)
                sensor_modes = (None, columns) if compact else (None, None)
                for controller in ("reference_only", "ppo"):
                    path = directory / "evaluation" / f"{controller}_{dt * 1e6:g}us.npz"
                    inputs.append(path)
                    with np.load(path) as saved:
                        trace = {key: saved[key] for key in ("state", "controls")}
                    count = len(trace["controls"])
                    print(
                        hand,
                        controller,
                        dt,
                        "intervals",
                        count,
                        "sensor_width",
                        model.nsensordata,
                        flush=True,
                    )
                    replay_models = models[:2]
                    recorders = [
                        HeldControlRollout(
                            replay_models, num_threads=8, group_identical_models=grouped
                        )
                        for grouped in group_modes
                    ]
                    official = Rollout(nthread=1)
                    try:
                        assert [len(r.groups) for r in recorders] == ([1, 1] if compact else [2, 1])
                        for start in range(0, count, 2):
                            indices = np.minimum(np.arange(start, start + 2), count - 1)
                            initial, control, spec = held_inputs(model, trace, indices, nstep)
                            expected = compare(
                                official, replay_models, data, initial, control, spec, nstep
                            )
                            np.testing.assert_array_equal(
                                expected[0][:, -1].astype(trace["state"].dtype),
                                trace["state"][indices + 1],
                            )
                            for recorder, selected_columns in zip(
                                recorders, sensor_modes, strict=True
                            ):
                                actual = compare(
                                    recorder,
                                    replay_models,
                                    data,
                                    initial,
                                    control,
                                    spec,
                                    nstep,
                                    selected_columns,
                                )
                                selected = (
                                    expected
                                    if selected_columns is None
                                    else (expected[0], expected[1][:, :, selected_columns])
                                )
                                check_arrays(actual, selected)
                                check_arrays((recorder.final_sensors,), (expected[1][:, -1],))
                                del selected
                                del actual
                            del expected
                            if start % 64 == 0:
                                print("checked", min(start + 2, count), "of", count, flush=True)

                        for recorder in recorders:
                            recorder.close()
                        recorders = [
                            HeldControlRollout(
                                models, num_threads=8, group_identical_models=grouped
                            )
                            for grouped in group_modes
                        ]
                        assert [len(r.groups) for r in recorders] == ([1, 1] if compact else [8, 1])
                        selected = np.linspace(0, count - 1, 8, dtype=int)
                        initial, control, spec = held_inputs(model, trace, selected, nstep)
                        for _ in range(2):
                            for recorder, selected_columns in zip(
                                recorders, sensor_modes, strict=True
                            ):
                                compare(
                                    recorder,
                                    models,
                                    data,
                                    initial,
                                    control,
                                    spec,
                                    nstep,
                                    selected_columns,
                                )
                        samples = [[], []]
                        for repetition in range(6):
                            for mode in (0, 1) if repetition % 2 == 0 else (1, 0):
                                begin = time.perf_counter()
                                compare(
                                    recorders[mode],
                                    models,
                                    data,
                                    initial,
                                    control,
                                    spec,
                                    nstep,
                                    sensor_modes[mode],
                                )
                                samples[mode].append(time.perf_counter() - begin)
                        row = dict(
                            hand=hand,
                            controller=controller,
                            physics_dt_s=dt,
                            model_sha256=hashlib.sha256(compiled.tobytes()).hexdigest(),
                            model_size_bytes=len(compiled),
                            sensor_width=model.nsensordata,
                            full_replay_batch_size=2,
                            timed_batch_size=8,
                            complete_control_intervals=count,
                            complete_physical_substeps=count * nstep,
                            exact_official_all_substep_state_sensor_parity=not compact,
                            exact_official_recorded_substep_state_sensor_parity=True,
                            exact_full_final_sensor_parity=True,
                            exact_frozen_endpoint_parity=True,
                            selected_interval_indices=selected.tolist(),
                            mode_labels=["grouped_full", "grouped_compact"]
                            if compact
                            else ["identity", "compiled"],
                            seconds=samples,
                            recorded_sensor_width=len(columns) if compact else model.nsensordata,
                            full_sensor_trajectory_bytes=8
                            * nstep
                            * model.nsensordata
                            * np.dtype(mujoco.MJTNUM_DTYPE).itemsize,
                            compact_sensor_trajectory_bytes=8
                            * nstep
                            * len(columns)
                            * np.dtype(mujoco.MJTNUM_DTYPE).itemsize
                            if compact
                            else None,
                            full_final_sensor_bytes=8
                            * model.nsensordata
                            * np.dtype(mujoco.MJTNUM_DTYPE).itemsize,
                            median_speedup=float(np.median(samples[0]) / np.median(samples[1])),
                        )
                        rows.append(row)
                        print(hand, controller, dt, row["median_speedup"], flush=True)
                    finally:
                        official.close()
                        for recorder in recorders:
                            recorder.close()
            finally:
                env.close()
    for path in inputs:
        hashes[str(path.resolve().relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    report = dict(
        scope="two-model complete frozen replay and eight-model CPU held-control timing; not end-to-end training",
        python_version=platform.python_version(),
        machine=platform.machine(),
        mujoco_version=mujoco.__version__,
        compact_comparison=compact,
        recorded_sensor_names=list(HOLDER_SENSORS) if compact else "all",
        input_sha256=hashes,
        held_control_sha256=hashlib.sha256(held_path.read_bytes()).hexdigest(),
        protocol="two warm-ups per mode; six samples per mode in alternating order; eight evenly spaced intervals per trace; eight threads for both modes; setup and parity excluded from timing",
        rows=rows,
        guard="Runtime parity and local recorder timing only. No changed policy, physical gate, learning gain or qualified cricket video.",
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    benchmark(args.source, args.output, args.compact)
