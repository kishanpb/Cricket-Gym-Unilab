# G1 World-Frame Foot-Placement Reward

The [uniform-start pilot](g1_cricket_first_step_uniform.md) improves left-hand
landing and regresses the right hand. Both still fail the complete first-step
gate. This comparison changes one reward term, not the physical success criteria.

Add `motion_world_foot_pos` with weight 1 and scale 0.02 m. Its unweighted value
is `exp(-max(left_error_squared, right_error_squared) / 0.02**2)`.
Errors compare actual world-frame ankle-roll body origins to the same planned
`foot_targets` used by the native physical audit, including environment origins.
Use planned targets, not the slightly approximate exported IK body positions.
The worse foot determines the reward; errors are neither root-aligned nor
averaged away. Reward is computed before advancing the motion command, matching
the reference frame applied to the measured motor interval. It is a control-rate
learning signal, not a replacement for the all-substep physical audit.

Keep uniform sampling, all existing rewards, reference/torque files, observations,
robot geometry, joint/motor limits, residual scale, PPO network and budget fixed.
Train each hand from scratch with seed 1, eight CPU environments, 256 updates /
49,152 transitions, retaining both final checkpoints regardless of performance.
No selected checkpoint, changed denominator or training-return increase is
evidence that the robot has learned a complete step or cricket delivery.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 \
  python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_first_step_foot_reward/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/first_step_foot_reward_v1/ppo_right
```

Repeat sequentially for `left` / `ppo_left`. In the configured fork environment:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 \
  python scripts/evaluate_g1_cricket_first_step.py \
  g1_cricket_results/first_step_foot_reward_v1/ppo_right --render
```

Evaluate both directories: final deterministic PPO
and zero residual, explicitly from frame zero, at 62.5 and 31.25 microseconds.
Retain all eight outcomes, complete diagnostic videos and native replay traces.
All 10.5 s completion, quiet hold, settling, support, foot-path, joint/motor,
contact and holder gates remain unchanged. This is an intermediate experiment
toward full running bowling, not a smaller substitute for the requested videos.

## Complete Results

Source `a3ac93083b6e9357e3af8dc81402d4098b4a79d9`. Both runs completed
256 updates / 49,152 transitions with seed 1. Saved training configurations
differ from the uniform-start parents only in this reward term and output
directory. Each retained scalar CSV contains all 256 finite iteration rows.

| Hand | Control | Physics step (microseconds) | Episode end (s) | Longest airborne interval (s) | First loaded recontact (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Zero residual | 62.5 | 6.70 | 1.9383 | 5.8705 |
| Right | Foot-reward PPO | 62.5 | 5.52 | 0.8561 | 3.7137 |
| Right | Zero residual | 31.25 | 6.70 | 1.9381 | 5.8703 |
| Right | Foot-reward PPO | 31.25 | 5.52 | 0.8684 | 3.7137 |
| Left | Zero residual | 62.5 | 6.68 | 1.9349 | 5.8661 |
| Left | Foot-reward PPO | 62.5 | 5.42 | 1.5362 | None |
| Left | Zero residual | 31.25 | 6.68 | 1.9348 | 5.8659 |
| Left | Foot-reward PPO | 31.25 | 5.42 | 1.5361 | None |

All eight fail the unchanged 10.5 s gate and terminate on tracked-body-height
deviation. Right PPO regresses from 6.52 to 5.52 s and fails initial quiet hold
again. Its first recontact occurs before the planned forward step starts at
4.5 s: the detector reports recontact after any qualifying airborne interval,
not necessarily landing from the intended step. Do not count this preparation-
phase event as successful forward-step landing. Left PPO regresses from 6.84
to 5.42 s and no longer lands. It retains the initial quiet-hold pass.

Every episode fails orientation, settling/support, stance-foot displacement,
foot-target error and final step distance/lateral drift. No joint or motor
limit violation, unintended loaded contact, ball penetration or release occurs.
This single-seed reward candidate is not promoted. A changed training return
cannot be compared directly to the parent's different reward definition.

The [right report](../g1_cricket_results/first_step_foot_reward_v1/ppo_right/evaluation/evaluation.json)
and [left report](../g1_cricket_results/first_step_foot_reward_v1/ppo_left/evaluation/evaluation.json)
retain all eight outcomes and 1,167,360 finite physical substeps. All intervals
pass exact independent native endpoint/sensor replay, all 44 input hashes per
report verify, and both zero-residual traces match every uniform-parent array
bit-for-bit at both resolutions. The
[right PPO video](../g1_cricket_results/first_step_foot_reward_v1/ppo_right/evaluation/ppo.mp4)
and [left PPO video](../g1_cricket_results/first_step_foot_reward_v1/ppo_left/evaluation/ppo.mp4)
show complete terminated episodes at 0.5x, not running-bowling highlights.
Baseline videos remain alongside them. All 1,220 frames decode nonblank at
960 x 540; both fixed six-frame-per-condition review sheets were inspected.

All 130 focused tests pass with warnings as errors, including world-frame
worst-foot reward semantics, environment offsets, applied-frame timing, no
live state writes and unchanged native replay. Final weights and scalar logs
remain; redundant initial weights, events and unrelated runtime diffs were
removed after scalar validation. Full running, delivery/recovery, ball-aware
batting and independently trained Menagerie policies remain unfinished.

## Runtime Follow-Up

The current CPU recorder groups models by object identity. Inspection of the
eight-environment owner found eight distinct model objects and eight singleton
groups, although all eight compiled MJB byte streams were identical (152,565
bytes each). This prevents this recorder from batching those environments
together. It is a structural finding, not a measured speedup.

After this paired policy comparison finishes, test opt-in grouping of immutable,
byte-identical compiled models. Keep genuinely different models separate and
preserve row order, partial resets, release constraints, warm starts and every
substep sensor. Require exact native replay and unchanged frozen-policy outcomes
before enabling it. Benchmark fixed inputs with warm-up and repeated samples;
do not change the runtime midway through either training/evaluation pair.
