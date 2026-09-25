# G1 CPU Recorder

The cricket owner supplied distinct model objects with identical compiled
content. The native recorder consequently used separate singleton batches.
`HeldControlRollout(..., group_identical_models=True)` now groups exact MJB
byte sequences, not just matching dimensions or selected physical properties.
Identity grouping remains the default. Models must remain immutable after
construction; reset-time model randomization remains unsupported by this adapter.

UniLab exposes the opt-in as `env.mujoco_group_identical_models`. For an owner
without that YAML key, append `+env.mujoco_group_identical_models=true` to the
existing training command. It requires the native mjbatch substep engine and
this fork's updated recorder. No task, reward, policy or physical limit changes.
Recorder source: mjbatch commit `556475bc83700f0b2f51383f53fad760223cff65`.

## Complete Replay

The [retained benchmark](../g1_cricket_results/model_grouping_benchmark.json)
uses both hands, reference-only and final foot-reward PPO controls, and both
62.5/31.25 microsecond timesteps. All 2,432 recorded control intervals and
1,167,360 unique physical substeps are covered, including failed episodes.
For each interval, both grouping modes reproduce every official MuJoCo rollout
state and sensor value exactly; terminal states also match the saved float32
endpoints exactly. This replays frozen controls, not newly trained actors.

Full replay uses two model copies at a time; an odd final interval is padded
with a duplicate and counted only once. Timing uses eight copies. Every one
of the 115,305 sensor values remains present at every substep; no contact or
body sensor is dropped. The original eight-row verifier was interrupted after
profiling showed roughly 11 GB footprint and excessive NumPy validation work.
A row-wise verifier still retained large arrays, so that attempt was also
stopped before producing a result. The completed run uses two-row replay and
row-wise equality checks, retaining the full original interval pool.

## Fixed Timing Sample

Local arm64, Python 3.13, MuJoCo 3.11.0. Both modes request eight threads;
each singleton batch necessarily uses only one. Each trace contributes eight
evenly spaced control intervals, selected before timing. Two warm-ups per mode
precede six samples per mode, alternating which mode runs first. Timings include
trajectory allocation and copying, but exclude model construction, parity
checks, environment logic, PPO updates and rendering. No project training or
test job ran concurrently with the completed benchmark. Raw samples are retained.

| Hand | Control | Step (microseconds) | Identity median (s) | Compiled median (s) | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Reference | 62.5 | 0.3413 | 0.1642 | 2.078 |
| Right | PPO | 62.5 | 0.3127 | 0.1517 | 2.062 |
| Right | Reference | 31.25 | 1.0381 | 0.6734 | 1.542 |
| Right | PPO | 31.25 | 1.0130 | 0.6774 | 1.495 |
| Left | Reference | 62.5 | 0.3186 | 0.1545 | 2.062 |
| Left | PPO | 62.5 | 0.3201 | 0.1582 | 2.024 |
| Left | Reference | 31.25 | 1.0839 | 0.7088 | 1.529 |
| Left | PPO | 31.25 | 1.0666 | 0.7138 | 1.494 |

This is a 1.49-2.08x local recorder speedup, not an end-to-end training claim.
The report pins 12 local inputs plus the installed recorder source hash and
compiled model hashes. Reproduce in the configured fork environment:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 \
  python scripts/benchmark_g1_cricket_model_grouping.py \
  g1_cricket_results/first_step_foot_reward_v1 \
  g1_cricket_results/model_grouping_benchmark.json
```

Tests cover noncontiguous groups, different masses/timesteps/contact/actuator
parameters, partial resets, held wrenches, releases, divergence and recovery,
and actual G1 phase/reset/native-replay behavior with grouping on and off.
The focused UniLab suite passes 207 tests and the native Batch/recorder suite
passes 60 tests, both with warnings as errors; this is not full upstream CI.
The grouping change cannot qualify a failed policy or create a new video result.

## Cricket Scope

The requested end state remains a humanoid replacing the earlier body while
preserving the cricket sequence: a one-bounce incoming delivery and two-handed
batting swing, plus a running bowler, legal release and recovery, in both hands.
The current lofted soft toss is not the requested incoming one-bounce delivery.
Neither that diagnostic nor a quiet hold or single step qualifies the final
showcase. This opt-in enables further CPU learning work without changing physics.

## Compact Sensor Recording

`HeldControlRollout.rollout(..., sensor_indices=columns)` optionally records
only the requested sensor columns at each substep. `columns` must be a
one-dimensional integer array; order and duplicates are preserved, and an empty
integer array is supported. The default `None` still records all channels.
Every physical sensor is computed regardless of selection. All FULLPHYSICS
states remain recorded at native precision, and `final_sensors` exposes every
sensor channel at the final substep. This cache is cleared before each call,
including failed calls, and on close. No stale or partial cache is returned.
Recorder source: mjbatch `91422674cb73328c6ba0c1b2d65749fcebcdf5f4`.

The UniLab opt-in is `env.mujoco_compact_substeps`, default false. It requires
the mjbatch substep engine, records the registered observer's channels, and
retains the complete final sensor cache used by task observations and rewards.
Without an observer, it records no intermediate sensor columns. State and
root-velocity trajectories, read-only observer inputs, held wrenches, releases,
partial resets and failure handling retain their existing contracts. Enable
grouping and compact recording together on an owner without those YAML keys:

```sh
+env.mujoco_group_identical_models=true +env.mujoco_compact_substeps=true
```

The [compact comparison](../g1_cricket_results/compact_substeps_benchmark.json)
replays the same eight retained cases: 2,432 control intervals and 1,167,360
unique physical substeps. Both modes use exact-model grouping. Every state,
every requested substep sensor and the complete final sensor array match
official MuJoCo rollout exactly; float32 endpoints match the frozen traces.
The requested `holder_force`, `holder_quat` and `ball_hand` sensors total 75
columns. Compact mode does not retain all 115,305 channels at every substep.
The earlier full-recording grouping benchmark above remains unchanged.

Two-copy complete replay bounds verification memory. Eight-copy timings use
the same fixed eight evenly spaced intervals per trace, two warm-ups per mode,
six samples per mode, alternating order, and eight requested threads. No
project training/tests overlapped timing. All samples, including the slower
left-PPO full-recording sample, remain in the report. Platform and exclusions
are the same as the grouping study; this compares recorder calls, not complete
PPO training or evaluation.

| Hand | Control | Step (microseconds) | Grouped full median (s) | Grouped compact median (s) | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Reference | 62.5 | 0.1547 | 0.0870 | 1.779 |
| Right | PPO | 62.5 | 0.1503 | 0.0870 | 1.729 |
| Right | Reference | 31.25 | 0.6852 | 0.1751 | 3.914 |
| Right | PPO | 31.25 | 0.6963 | 0.1751 | 3.977 |
| Left | Reference | 62.5 | 0.1574 | 0.0902 | 1.744 |
| Left | PPO | 62.5 | 0.1566 | 0.0905 | 1.730 |
| Left | Reference | 31.25 | 0.6850 | 0.1752 | 3.910 |
| Left | PPO | 31.25 | 0.7313 | 0.1789 | 4.089 |

At 62.5 microseconds, an eight-row, 320-substep sensor trajectory falls from
2,361,446,400 bytes to 1,536,000 bytes, plus a 7,379,520-byte complete final
sensor cache in either mode. At 31.25 microseconds the trajectories double;
the final cache does not. These are array allocation sizes, not measured total
RSS: model/data storage, full sensor computation and state trajectories remain.
The measured incremental recorder speedup is 1.73-4.09x over grouped full
recording. Do not multiply separate study ratios into an end-to-end claim.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 \
  python scripts/benchmark_g1_cricket_model_grouping.py \
  g1_cricket_results/first_step_foot_reward_v1 \
  g1_cricket_results/compact_substeps_benchmark.json --compact
```

All 12 input hashes plus the recorder source hash verify. The focused suites
pass 235 UniLab tests and 89 native Batch/recorder tests with warnings as errors.
They cover selected/duplicate/empty columns, invalid selections, cleared failure
caches, native release/reset parity and actual both-handed G1 task behavior.
Ruff and diff hygiene checks pass; this is not full upstream CI. This runtime change
does not alter a policy, reward, robot, physical gate or existing failure result,
and no new video or learned cricket capability is claimed.
