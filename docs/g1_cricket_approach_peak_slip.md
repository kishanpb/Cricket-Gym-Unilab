# Peak-Slip Learning Comparison

The closed-loop residual completes its approach but retains fast touchdown
sliding. Its original slip penalty averages squared contact speed across a
20 ms control interval. In the retained fine-grid PPO traces, replacing that
average with each foot's interval peak changes the full-episode slip penalty
from 1.016 to 4.719 (right) and 1.028 to 5.394 (left), at the same -5 weight.
Those are counterfactual reward calculations, not new policy outcomes.

This experiment changes only the `loaded_slip` reward function to the sum of
squared per-foot interval peak speeds. Contact points, >1 N load threshold,
sensor timing, physical model, zero-initialized 29-joint actor, observations,
forward command, other rewards and all PPO settings remain unchanged. It does
not include the separate lane-feedback trial. The external locomotion prior
remains explicitly external and frozen; no weights are redistributed.

Fresh right/left seed-1 actors each train for 256 updates x 24 steps x 8 CPU
environments = 49,152 transitions, matching the mean-slip parent. Use only the
final checkpoint. Evaluate both hands, zero residual and final PPO at both
62.5/31.25-us physics, seed 1 from rest; retain all eight outcomes and both
finest PPO videos, including falls. No checkpoint or favorable-seed selection.

The target remains raw peak loaded-foot slip <0.2 m/s, with all original lane,
stance slip, stability, joint/motor, contact, holder and resolution gates.
Higher training reward or smaller mean slip cannot replace peak-slip success.
Do not extend the pilot just because training improves. A failed final pool
closes this reward-only pilot as a full solution; diagnose the actual landing
controller before scaling compute. Walking/carry remains distinct from the
requested continuous running gather, legal overarm release and recovery.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_approach_peak_slip/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/approach_peak_slip_v1/ppo_right
```

Repeat for `left`/`ppo_left`; retain iteration diagnostics and evaluate with
`evaluate_g1_cricket_approach_learning.py`, the new result directory and
`--action-name residual --render`. Finish with the existing approach reporter.
Both physical scene and controller commands must reproduce the parent under
identical actions; only the reward may differ.
