# G1 CPU Recorder Grouping

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
