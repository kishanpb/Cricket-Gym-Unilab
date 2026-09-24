# Matched-resolution humanoid contact-model transfer

The isolated compliance study halves penetration but doubles peak loads at a
2 ms time constant, with little change in rebound ratio. Only its two finest
timesteps agree for that candidate. This follow-up tests the articulated,
moving blade; no isolated-fixture result is treated as robot evidence.

## Frozen Contract

Compare the current **4 ms control** against an opt-in **2 ms candidate** at
both **0.0625 and 0.03125 ms** physics steps. Pair both settings at the same
resolutions, keeping 20 ms control, geometry, masses, prior, gains, actuator
limits, reward, observations, reset and full shot gates unchanged. Only the
bat-ball positive solref time constant changes; damping ratio stays one.
The existing task and all retained policy checkpoints remain unchanged.

The new `G1CricketImpactV2` owner inherits the old scene builder, replacing
only that pair parameter. It rejects physics steps above 0.0625 ms. Hydra
owners `g1_cricket_compliance_v2/mujoco` and `/mjbatch` inherit tanh-v1;
the latter selects the existing native Batch recorder. The task name versions
the physical contract, not the policy architecture. This is a scoped task
extension using the existing ADR-0010/0011 recorder contracts, not a new backend.

Run right AND left hands, seed 4301, center lane; both zero residual and the
unchanged `pmppppp_s10` schedule. The latter is normalized magnitude 0.95,
signs `[1,-1,1,1,1,1,1]`, reversal at tick 10, zero from tick 20 onward. Left
uses those same arm-channel commands: **untrained command transfer, not a
mirrored or learned skill**. No command selection or optimization occurs here.
Both MuJoCo rollout and native mjbatch run every combination: **32 full
100-control-tick / 2-second rows**, or retained native early failures.

Require blade-first-only contact, no forbidden ball or bat contacts, pelvis
height >=0.48 m and up-z >=0.65, joint/actuator limits, first exit vx strictly
>1 m/s, maximum blade penetration <=6 mm over all contacts, and native full
episode completion. Keep all negatives, contact episodes and simulated
blade/fixture peak loads. No external forces, pose writes outside reset/replay,
stronger motors, gate changes or new training. Preserve exact executor outcome
and first-impact evidence plus zero independent replay endpoint errors.

Use the existing resolution comparator (5% or absolute floors: 0.1 mm depth,
0.05 m/s exit velocity, 1 N blade peak force), identical contact/failure/duration
states. Also compare fixture peaks at max(5%, 1 N force / 0.1 N m torque).
A model/hand/controller witness requires all four executor/resolution cases to
pass, all comparison checks, and exact executor evidence. Increased model loads
must be reported; passing actuator limits does not establish fixture strength
or hardware safety. This uncalibrated contact model cannot prove material fidelity.

A passing scripted witness only supports a later learning design. Both-hand
full lane/seed policy-versus-zero evaluation remains required before any
learned/generalization claim or showcase. Results at different physical models
must not be merged into a policy-improvement curve. Existing videos remain intact.

## Reproduce

With the previously pinned Unitree prior and runtime installed:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_model_transfer.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_model_transfer.py
```

Preflight freezes the full parent chain, isolated study, new owner, runner,
tests, this contract and installed native executor. Outputs are
`g1_cricket_results/model_transfer_v1/preflight.json` and `evaluation.json`;
the runner refuses to overwrite retained evidence. Reproduce in a clean
checkout without that output directory. No new upstream PR or social publication.
