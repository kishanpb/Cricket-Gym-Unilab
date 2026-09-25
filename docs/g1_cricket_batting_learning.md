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

Add `--render` to each 31.25-microsecond evaluation to retain both reference and
PPO videos. Export the native scalar logs with
`scripts/retain_g1_training_diagnostics.py` for each completed training directory,
then build the complete report and the right-then-left video:

```sh
PYTHONPATH=src:scripts uv run python scripts/report_g1_cricket_batting_learning.py \
  g1_cricket_results/bimanual_batting_learning_v1 --media
```

The report checks both final checkpoints, the declared training budget and
observations, all four evaluation files and their source fingerprints. Video
assembly requires both full 150-control PPO episodes plus their terminal hold;
it rejects missing or blank frames and retains 0.5x playback. Qualification
remains separate from video assembly. Fixed review frames are 0.02, 1.06, 1.40,
1.80 and 3.00 seconds, independent of which actor performs better.

## Complete Results

Task source `c969ca49`. Both seed-1 runs completed all 49,152 transitions;
all 256 iteration rows and 53 native scalar series per hand are finite. Final
actor/critic tensors are finite and changed since the first update. The
[complete report](../g1_cricket_results/bimanual_batting_learning_v1/summary.json)
retains both final checkpoints and all eight outcomes, without checkpoint selection.

| Hand | Control | Physics step (microseconds) | Peak bat error (cm) | First forward exit (m/s) | Peak blade penetration (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Reference only | 31.25 | 13.6990 | 2.60307 | 4.0900 |
| Right | PPO | 31.25 | 14.8104 | 2.54977 | 3.9685 |
| Right | Reference only | 15.625 | 13.6891 | 2.62044 | 4.0998 |
| Right | PPO | 15.625 | 14.7939 | 2.56792 | 3.9965 |
| Left | Reference only | 31.25 | 13.6591 | 2.59972 | 4.0653 |
| Left | PPO | 31.25 | 13.5967 | 2.62646 | 4.0830 |
| Left | Reference only | 15.625 | 13.6537 | 2.61686 | 4.0697 |
| Left | PPO | 15.625 | 13.5804 | 2.63794 | 4.0812 |

Every episode completes three seconds, hits after exactly one incoming bounce,
and passes height, grip, joint/motor limits, unintended-contact, root/joint
tracking and ball-contact gates. All four force/penetration/exit-velocity
resolution comparisons pass. **Every episode still fails the unchanged 8 cm
bat-path limit**, with its error peak at 1.66 s during follow-through. Right
PPO worsens reference tracking; left PPO improves it only slightly. Adding
ball/contact observations and training does not establish learned interception:
reference-only control also hits, and no perturbed delivery was tested.

The [full right-then-left slow-motion video](../g1_cricket_results/bimanual_batting_learning_v1/two_hand_ppo_learned_batting.mp4)
shows the final PPO actors with mechanical grips, not learned finger grasping.
The [fixed-frame review sheet](../g1_cricket_results/bimanual_batting_learning_v1/learned_batting_contact_sheet.png)
was visually inspected. Full reference/PPO source clips remain in
[right fine evaluation](../g1_cricket_results/bimanual_batting_learning_v1/right_fine/)
and [left fine evaluation](../g1_cricket_results/bimanual_batting_learning_v1/left_fine/).
All 1,050 source/combined frames decode nonblank at 960 x 540. This is a
development diagnostic, not an advertising qualification or running-bowling video.

All 1,152,000 evaluated physical substeps pass independent native endpoint/sensor
replay. Each evaluation's 75 input hashes and recorder hash verify; every
reference-control physical trace exactly matches the prior zero-lead study.
The complete summary reproduces, and 154 focused tests pass with warnings as
errors. Final weights, configs, scalar CSVs and complete reports remain;
redundant initial weights, events and unrelated generated runtime diffs were
removed after verification. Neither actor is promoted as the accuracy solution.
