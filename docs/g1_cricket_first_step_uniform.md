# G1 Uniform-Start First-Step Comparison

This completed comparison tests whether broader training-state coverage improves
the failed [from-rest PPO pilot](g1_cricket_first_step_learning.md). Neither
hand passes the full step gate; this is not a running-bowling demonstration.

The only training change is `sampling_mode: uniform`, inherited through
`task=g1_cricket_first_step_uniform/mjbatch`. The existing motion sampler
uniformly draws a reference frame on each reset. Robot and held-ball position
and velocity are initialized together only at reset. No live state writes,
external support forces, geometry changes or actuator-limit increases are
allowed. Clip-end truncation stays enabled; short late-phase episodes are
training data, never evidence of a complete step.

Both hands train independently from scratch, seed 1, eight CPU environments,
256 updates / 49,152 transitions each. Reference poses, support torques,
reward, termination thresholds, observations, residual scale, PPO architecture
and training budget are unchanged. Retain the final actor, complete finite
scalar logs and every evaluation outcome, including regressions.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_first_step_uniform/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/first_step_uniform_v1/ppo_right
```

Repeat with `left` and `ppo_left`, sequentially to avoid local contention.
Use the existing `evaluate_g1_cricket_first_step.py` on both run directories.
It explicitly overrides sampling to `start`, resets motion and timeout counters
to zero and leaves the saved training configuration unchanged. Evaluate zero
residual and final PPO at 62.5 and 31.25 microseconds, with the unchanged full
10.5 s physical gate and exact native state/sensor replay. Retain all eight
outcomes and complete failure-inclusive diagnostic videos; partial resets,
larger returns and selected landings cannot qualify a full run-up or delivery.

The evaluator source change does not rewrite old results: their source hashes
refer to the retained evaluator revision recorded in each historical report.

## Complete Results

Source `d52ff90112ea0c769045d5afb6f98a0f446b9684`. Both sequential runs
completed all 256 updates / 49,152 transitions, retaining final `model_255.pt`
checkpoints. Both saved training configurations differ from their parents only
in sampling mode and output directory; all 256 scalar rows per hand are finite.
The evaluator reports training sampling as `uniform` and evaluation as `start`.

| Hand | Control | Physics step (microseconds) | Episode end (s) | Longest airborne interval (s) | Loaded landing (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Zero residual | 62.5 | 6.70 | 1.9383 | 5.8705 |
| Right | Uniform-start PPO | 62.5 | 6.52 | 0.5677 | 5.7566 |
| Right | Zero residual | 31.25 | 6.70 | 1.9381 | 5.8703 |
| Right | Uniform-start PPO | 31.25 | 6.52 | 0.5601 | 5.7564 |
| Left | Zero residual | 62.5 | 6.68 | 1.9349 | 5.8661 |
| Left | Uniform-start PPO | 62.5 | 6.84 | 1.7572 | 5.7716 |
| Left | Zero residual | 31.25 | 6.68 | 1.9348 | 5.8659 |
| Left | Uniform-start PPO | 31.25 | 6.84 | 1.7618 | 5.7732 |

All eight fail the full 10.5 s gate and terminate on tracked-body-height
deviation. Every episode fails pelvis orientation, settling/support, planted-
foot displacement, foot-target error and final step distance/lateral drift.
No episode violates joint or motor limits, has an unintended loaded contact,
penetrates the ball or releases it. Both actors now pass initial quiet hold,
but neither completes and settles its step.

Relative to from-rest training, right PPO regresses from 7.22/7.28 s to 6.52 s;
left improves from 5.00/4.98 s without landing to 6.84 s with landing. These
single-seed, mixed outcomes do not establish a generally better method.
Do not select one hand as promotion evidence or compare training returns
across different reset distributions as if they measured complete episodes.

The [right report](../g1_cricket_results/first_step_uniform_v1/ppo_right/evaluation/evaluation.json)
and [left report](../g1_cricket_results/first_step_uniform_v1/ppo_left/evaluation/evaluation.json)
retain all 1,283,520 finite physical substeps, actions, controls and states.
Every interval passes exact independent native state/sensor replay. Both
zero-residual traces match every parent array bit-for-bit at both resolutions.
All 44 input hashes per report verify. The
[right PPO video](../g1_cricket_results/first_step_uniform_v1/ppo_right/evaluation/ppo.mp4)
and [left PPO video](../g1_cricket_results/first_step_uniform_v1/ppo_left/evaluation/ppo.mp4)
show complete terminated episodes at 0.5x, not qualified highlights. Baseline
videos remain alongside them. All 1,341 frames decode nonblank at 960 x 540;
both fixed six-frame-per-condition review sheets were inspected.

All 126 focused tests pass with warnings as errors, including reset wrist/ball
kinematics, phase coverage, partial-reset isolation, unchanged physical startup
traces and forced from-rest evaluation without mutating training config.
Final weights and scalar CSVs remain; redundant initial weights, TensorBoard
events and unrelated installed-runtime diffs were removed after validation.

## Next Comparison

Do not extend the same runs unchanged. The current whole-body position reward
uses a 0.3 m scale and reference-relative body positions, while qualification
requires world-frame foot-path error at most 0.02 m and planted-foot motion at
most 0.005 m. Removing balance correction from the unloaded ankle was already tested
in the parent controller study and also failed; do not repeat it unchanged.

Next, add one world-frame foot-link tracking reward using the same ankle-roll
body origins as the existing foot-target audit: weight 1 and exponential
squared-error scale 0.02 m, with maximum position error across both feet.
Keep uniform sampling,
all other rewards, reference, observations, residual scale, robot and budget
fixed; train both hands from scratch. This is a proposed reward-axis experiment,
not implemented or computed here. Keep every from-rest evaluation and the full
physical gates unchanged; reward improvement cannot qualify a step, run-up,
delivery or showcase. Ball-aware batting, repeated strides, delivery/recovery
and independent Menagerie-model learning remain unfinished.
