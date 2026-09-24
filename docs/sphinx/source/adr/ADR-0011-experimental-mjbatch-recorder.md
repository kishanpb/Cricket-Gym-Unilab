---
orphan: true
---

# ADR-0011 Experimental mjbatch Recorder

- Status: Experimental (fork only)
- Date: 2026-09-24
- Owners: Cricket-Gym-UniLab and mjbatch cricket forks
- Supersedes: None
- Superseded by: None

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

## Equality Activation Extension

The fork-only `EqualityConstraintBackend` capability extends the declared
`SimBackend` boundary for explicit, persistent per-environment activation.
Tasks access it through `NpEnv.equality_constraints`, never backend-private data.
The cold-path name order covers every model equality, including inactive ones.
Setters accept a boolean matrix for selected environments; getters copy it.
Changing activation does not change qpos, qvel or the previous solved sensor
cache. The next held interval applies it in MuJoCo state-field order after CTRL
and any XFRC_APPLIED input. This path also runs without a substep observer.

`set_state` resets only selected environments to their model defaults after
a successful state reset. Task code owns release latches and cannot treat a
constraint as a learned grasp. FULLPHYSICS snapshots still omit equality state;
replay must retain the activation matrix separately. The low-level native
recorder remains stateless across intervals: omitting EQ_ACTIVE intentionally
restores model defaults, while the UniLab owner supplies its persistent state
every interval. No-equality models retain the existing fast path and results.

Release tests cover activation together with controls and external wrenches,
observer on/off, initially inactive constraints, copied getter state, selective
resets and exact native/direct/official-rollout trajectories. The separate G1
holder scene tests actual left/right wrist geometry, non-overlap, continuity at
release, gravity-only free flight and zero released-ball constraint force.
They are physical infrastructure tests, not bowling performance or learning.
Frozen batting results remain tied to their original source revisions; this
extension does not regenerate or relabel them as current-head evidence.

## Alternatives Considered

- Endpoint-only contact sampling misses short impacts and cannot validate release loads.
- Encoding activation in FULLPHYSICS would misrepresent the MuJoCo state contract.
- Changing free-ball velocity on release would inject an unearned bowling action.

## Evidence In Repo

- `tests/base/test_mujoco_substeps.py`: existing solved-phase recording and reset parity.
- `tests/base/test_mujoco_constraints.py`: public equality capability and reset lifecycle.
- `tests/envs/test_g1_cricket_holder.py`: both-hand G1 clearance and physical release.
- Companion mjbatch `tests/test_held_control.py`: full interval parity and explicit defaults.

## Related Documents

- {doc}`/adr/ADR-0010-experimental-substep-observation`
- {doc}`/adr/ADR-0007-unisim-extraction-boundary`
