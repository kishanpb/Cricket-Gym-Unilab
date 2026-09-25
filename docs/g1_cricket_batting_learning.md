# Ball-Observed G1 Batting

This versioned learning task extends the two-handed G1 scene with measured
bat-position error, ball state and simulated contact/tactile observations.
It is not a relabeling of the frozen dry-swing actors or learned finger grasping.
Mechanical palm grips, original robot geometry/inertias/motor limits and
the reset-only one-bounce delivery remain unchanged.

## Learning Contract

Actor and critic both receive the existing ball-relative-to-bat observation,
contact/tactile snapshots and the measured bat-center error against the current
motion frame. World targets include each environment's origin; the reference
phase is not shifted to make the error smaller. Bat-reference FK is shared with
the evaluator. Ball position/velocity are privileged simulator state, not
camera estimates; contact features are uncalibrated solve-phase snapshots,
not a high-frequency tactile-history encoder or hardware measurements.

The inherited whole-body motion reward remains. Add bat tracking with weight
2 and `exp(-||error||^2 / 0.08^2)`, plus the existing approach/first-exit shaping
with weight 1, now requiring **positive normal force on a present blade
contact**. Margin-only contacts and stale unused sensor slots cannot trigger
the loaded-hit history. The first force-free exit is acquired at physics
substeps and rewarded once; partial resets clear only selected environments.
Reward is shaping, not the physical acceptance gate or proof of cricket success.

Use the retained supported reference, zero motor lead, residual scale 0.05,
ankle/waist/root gains 4/1/4, and the same `(4, 0, 1.3)` m, `(-3, 0, 4)` m/s
reset-only delivery. Native gravity/contact governs subsequent motion. Physics
is 31.25 microseconds and control is 20 ms. This is a fixed nominal practice feed,
not a regulation-speed or held-out delivery pool.

## Declared CPU Pilot

Train fresh, independent right/left PPO actors: seed 1, eight native mjbatch
environments, 256 updates, 24 controls per environment/update, **49,152
transitions per hand**, with final checkpoint `model_255.pt`. Use two 128-unit
ELU layers, initial action noise 0.08, the inherited adaptive PPO optimizer and
enabled learner/environment finite checks. No imported actor weights, mid-run
reward changes or best-checkpoint search. Retain all scalars and both final
outcomes even on failure.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_batting_learning_v1/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/bimanual_batting_learning_v1/ppo_right
```

Repeat with `env.handedness=left` and `ppo_left`. The declared outcome pool is
both hands x reference/PPO x 31.25/15.625-microsecond evaluation, complete
three-second episodes from the first frame, seed 1. Keep every failure.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python scripts/evaluate_g1_cricket_tracking.py \
  g1_cricket_results/bimanual_batting_learning_v1/ppo_right \
  --output g1_cricket_results/bimanual_batting_learning_v1/right_fine \
  --contact-dt 0.00003125 --bounced-delivery --compact-substeps
```

Repeat for both resolutions and hands, using new directories. Compare new
reference/PPO returns only within this reward version, never against earlier
dry-swing returns. Report raw bat errors, contact forces/penetration, forward
exit, bounce ordering, motor/joint/grip/stability checks and resolution
agreement. A completed training run alone cannot clear the original 8 cm
bat-path gate, any physical gate or the running-bowling requirement. Nominal-feed
success would still need perturbed-delivery evaluation before a robust showcase.
