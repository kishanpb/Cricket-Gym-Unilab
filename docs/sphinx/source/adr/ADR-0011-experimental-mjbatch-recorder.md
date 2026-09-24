---
orphan: true
---

# ADR-0011 Experimental mjbatch Recorder

- Status: Experimental (fork only)
- Date: 2026-09-24
- Owners: Cricket-Gym-UniLab and mjbatch cricket forks
- Supersedes: None

## Decision

Extend ADR-0010 with an explicit `mujoco_substep_engine: mjbatch` selector.
The default remains official MuJoCo Rollout. The native executor is the public
`mjbatch.held_control.HeldControlRollout` from the companion fork, imported only
when selected. This is native mjbatch execution of the UniLab G1 task, not an
equivalence claim for mjbatch's earlier cricket scene or a new upstream backend.
No installed third-party code, task reward, policy I/O or physical limit changes.

Use the existing fully materialized models, including injected tracking sensors.
Batch copies models at construction; init-time variants are supported, but model
randomization at reset is explicitly unsupported. Root/joint/ball state resets,
partial resets and external body wrenches retain the existing owner behavior.
Direct caller mutation of a materialized model is outside this fixed-model path.

Every held-control interval starts from FULLPHYSICS with reset auxiliary state.
The recorder restores public INTEGRATION states into Batch and calls `step()`
once per physics substep. It preserves warmstart inside the interval, records
solved-phase sensordata without extra forward calls, and returns native-dtype
trajectories. Existing UniLab endpoint caches retain their float32 boundary.
Models with different identities use separate Batch groups, stepped sequentially;
this first implementation makes no speedup or production-scalability claim.
Reset native warning counters at each interval and check them after every step.
Any warning raises before returning a trajectory, including divergence that
MuJoCo might otherwise auto-reset into finite-looking output. This fail-fast
boundary differs from official Rollout's frozen-divergence output; exact parity
claims apply only to warning-free trajectories.

## Evidence Boundary

Require exact full substep state/sensor parity, not just rendered poses, through
actual bat/ball contact. Verify both hands and timesteps, raw actions beyond the
old clipping threshold, prior/action history, rewards, separation latches,
external wrenches, partial resets and control-boundary warmstart reset.
Unsupported combinations fail explicitly. Default-engine regressions remain tests.

Keep old experiment JSON, checkpoints and hashes immutable. Historical source
checks may resolve only explicitly listed path/digest pairs at a fixed verified
Git revision; data/checkpoint hashes and live evaluator checks remain strict.
New native-mjbatch evidence must record current source, executor and runtime
provenance. Parity is not new training, physical calibration, policy promotion
or permission for a new upstream PR or social/video release.
