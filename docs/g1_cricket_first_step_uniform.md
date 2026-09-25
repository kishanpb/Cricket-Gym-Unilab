# G1 Uniform-Start First-Step Comparison

This tests whether broader training-state coverage improves the failed
[from-rest PPO pilot](g1_cricket_first_step_learning.md). It is not yet a
completed experiment or a running-bowling demonstration.

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
