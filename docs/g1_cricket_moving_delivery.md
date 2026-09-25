# Continuous Moving Delivery Prototype

This development experiment reconnects the earlier cricket sequence to the
G1's live locomotion controller. It is not a trained cricket policy or a
qualified running-bowling video. The previous batting videos are unchanged.

## Controller

The verified external Unitree locomotion policy continues to observe the live
robot and command its legs and trunk throughout the episode. Both arms blend
into the retained G1-scaled bowling reference, including the non-bowling arm;
the reference's root and leg poses are never written to the simulator. Original
motor gains, force limits, collision geometry and joint limits remain unchanged.

The eight-second episode begins from rest. Forward command ramps to 1 m/s,
continues through gather at 4.00 s and scheduled release at 4.62 s, then ramps
down from 4.80 to 5.80 s. Arm targets blend back to the live prior by 6.10 s,
leaving recovery time. A 1 m/s command is not proof of a running gait. Release
disables the existing holder equality using the integrated ball state; it does
not inject ball velocity. Mechanical holding is not a learned finger grasp.

Local residual actions are zero. This prototype uses reference arm motion plus
an externally learned locomotion controller, not the earlier local carry-PPO
checkpoints and not a new local cricket-learning result. External weights are
hash-checked locally and are not redistributed.

## Evaluation

Every right/left rollout at both 62.5 and 31.25 microsecond physics steps is
retained, starting at seed 1 and ending at the first physical termination or
eight-second timeout. Independent native MuJoCo replay checks each integrated
endpoint and sensor buffer exactly. The existing full-delivery gate checks
feet, overarm geometry, elbow extension, ball speed/contact/penetration, motor
and joint limits, stability, bounce and target crossing. Slip and lane drift
remain explicit diagnostics, not silently waived by a delivery result.

Videos retain the complete start-to-terminal sequence at 0.5x, including
failure. They must not be cropped into successful-looking advertising footage.

```bash
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python python \
  scripts/evaluate_g1_cricket_moving_delivery.py \
  g1_cricket_results/moving_delivery_v1 --render
```

Use a fresh output directory; the command refuses to overwrite an experiment.
