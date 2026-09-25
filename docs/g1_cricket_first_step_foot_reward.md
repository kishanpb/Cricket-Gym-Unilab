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
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_first_step_foot_reward/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/first_step_foot_reward_v1/ppo_right
```

Repeat sequentially for `left` / `ppo_left`. Evaluate each directory with
`scripts/evaluate_g1_cricket_first_step.py --render`: final deterministic PPO
and zero residual, explicitly from frame zero, at 62.5 and 31.25 microseconds.
Retain all eight outcomes, complete diagnostic videos and native replay traces.
All 10.5 s completion, quiet hold, settling, support, foot-path, joint/motor,
contact and holder gates remain unchanged. This is an intermediate experiment
toward full running bowling, not a smaller substitute for the requested videos.
