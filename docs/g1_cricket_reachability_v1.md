# G1 bounded residual reversal probe

This is a scripted simulation controllability diagnostic, **not RL training,
policy promotion, a showcase, or physical contact calibration**. It asks whether
one fixed finite motion family can produce a fully gated forward strike before
spending another PPO training budget. A negative result does not prove that the
robot, task or seven-dimensional residual space is incapable of batting.

## Fixed contract

Keep the tanh-v1 task, frozen locomotion prior, 0.70 kg rigid wrist bat fixture,
reset distribution, reward and physical limits unchanged. Only the commanded
seven arm residuals vary. Search right-handed seed 4301, center toss only; this
is deliberately not generalization or a tournament result.

1. Evaluate zero residual plus every one of the 128 seven-joint sign vectors.
   Command normalized `u = 0.95 * signs`, passing `atanh(u)` through the existing
   tanh action owner. Hold `u` for ticks 0-9, `-u` for ticks 10-19, then zero.
   The control interval remains 0.02 s. The prior still runs closed-loop.
2. Screen 25 control ticks (0.50 s) at physics dt 0.00025 s without changing the
   native 2 s episode horizon. Retain every row, including all failures.
3. Rank nonzero, prefix-safe trials with an observed first blade separation by
   first-exit ball vx descending, breaking ties by candidate ID. Take at most
   eight, and test switch ticks 6, 8, 12 and 14, always stopping the residual at
   tick 20. Empty eligibility stops refinement; do not broaden the search.
   Total prefix budget: at most 161, including zero.
4. Rank the combined prefixes by the same rule. Up to eight prefix-safe trials
   with first-exit vx strictly greater than 1 m/s receive fresh complete 100-tick
   replays in both executors (MuJoCo rollout and native mjbatch), at physics dt
   0.00025 and 0.000125 s. Retain every replay, including failures.

A prefix can only be eligible when it lasts exactly 0.50 s, has not terminated
or truncated, and passes every physical/contact gate except full-episode
completion and outgoing speed. Its `passed` field stays false. Prefix results
never constitute a valid shot, even when the measured exit speed exceeds 1 m/s.

A scripted witness requires **every** full replay to pass the unchanged gate,
exact cross-executor outcome rows, and the existing impact-resolution comparator
(including its absolute/5%-relative tolerances). Full gates require native 2 s
truncation without termination, first contact blade-only, no pitch before blade,
no ball/robot/handle/wicket contact, first exit vx >1 m/s, no guarded contact,
pelvis height >=0.48 m and up-z >=0.65, joint-limit excess <=1e-6 rad, actuator
force fraction <=1+1e-6, and maximum blade penetration <=6 mm over all impacts.
Do not substitute a faster second exit or overlook a later forbidden contact.

## Evidence and reproduction

The new trial collector leaves all pinned historical evaluators unchanged. Before
search it must exactly reproduce eight complete zero-residual rows: both hands,
both timesteps, both executors, seed 4301/center toss. Equality covers the entire
row, including returns, events, guard counts and force/torque extrema. Tests
separately exercise simultaneous blade/pitch contact, forbidden later contact,
recontact penetration, missing first separation and every physical gate.

Every native control interval is independently replayed at physics rate with
exact endpoint and named-sensor equality. A second serial pass records signed
ball-blade impulse and contact-point motion, with another exact endpoint check.
Full-shot safety uses **all-geometry** contacts, not this blade-only diagnostic.
Contact geometry/velocity/force fields are pre-integration solved quantities;
the separation velocity is measured after integration on the first absent step.
Simulated force/tactile values are uncalibrated, not real robot measurements.

Run from this fork using the existing project runtime and two CPU threads:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_reachability.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_reachability.py
```

The preflight freezes plan, all inherited input hashes, new collector/search/test
sources, package versions and installed native executor hashes before compute.
External locomotion assets retain their pinned checks and are not redistributed.
The report is `g1_cricket_results/reachability_v1/evaluation.json`; no checkpoint
is trained. Existing reports/checkpoints/videos are not replaced. Any witness is
an optimization clue for a subsequent predeclared learned-control experiment,
not a learned G1 video or a claim about left-handed batting or bowling.
