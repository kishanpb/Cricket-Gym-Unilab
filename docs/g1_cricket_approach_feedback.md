# Closed-Loop Whole-Body Approach Residual

The fixed motor-tape pilot lost balance before delivery, and its zero-residual
baseline was timestep-sensitive. This follow-up retains the pinned external
Unitree locomotion controller's five-frame feedback rather than copying its
commands along one trajectory. It is not independently trained locomotion or
running bowling. The earlier batting videos remain unchanged.

## Controller and Learning Contract

`task=g1_cricket_approach_feedback/mjbatch` uses the same robot, holder, scene,
physics, gains, limits and eight-second command profile as the approach teacher.
The original seven-arm/release action is held at zero. A local PPO actor adds
`0.1 * clip(action, -1, 1)` radians to all 29 SDK-order motor targets, including
legs and trunk. It cannot release the ball or overwrite physical poses.
The local actor's output layer is initialized to zero; inference therefore starts
at the original closed-loop controller, not a random deterministic correction.
Gaussian exploration starts at 0.02 action units, or 0.002 target radians.

The external controller still uses its original 480-input history, original
joint mapping, defaults and previous **external** actions. The local actor and
critic each receive 148 current values: existing bowling proprioception, ball
and holder state, prior and residual actions, time, plus per-foot RMS contact
slip, normal load and lateral lane error. Only the local residual is trained;
external ONNX/deployment files remain locally obtained, hash-checked assets and
are not redistributed in checkpoints.

Reward terms are upright (1), failure (-5), residual action-rate (-0.01),
loaded-contact mean-square slip (-5), lane tracking (1), and commanded planar
velocity tracking (1). Velocity reward is `exp(-sum((error / 0.25 m/s)^2))`;
lane reward is `exp(-(error / 0.1 m)^2)`. The obsolete release reward is removed.
These are training signals, not replacements for the original physical gates.
One substep observer samples holder and foot-frame/contact signals together.

## Fixed Protocol

Train fresh independent right/left residual actors, seed 1, eight CPU environments,
24 steps/update and 256 updates: 49,152 transitions per hand. Episodes start from
rest, with the teacher's existing reset jitter; no mid-trajectory resets or
checkpoint selection. Actor and critic are 128 x 128 ELU networks. Save the final
checkpoint, full configuration and all iteration-indexed diagnostics.

Evaluate final PPO and zero residual for both hands at 62.5 and 31.25 us, seed 1,
from rest: all eight cases. Unlike the motor-tape pilot, zero residual recomputes
the same closed-loop policy at each timestep. Retain every failure and both
finest PPO videos. Require independent native substep replay, original gait,
stability, contact and holder checks, and resolution comparisons. Agreement
between two failed episodes is not qualification. Seed 1 is development, not
held-out robustness.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_approach_feedback/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/approach_feedback_v1/ppo_right
```

Repeat for `left`/`ppo_left`, then use
`scripts/evaluate_g1_cricket_approach_learning.py` with the experiment directory,
`--action-name residual --render`, followed by
`scripts/report_g1_cricket_approach_learning.py`.

No gather, overarm release or recovery after delivery is implemented by this
approach task. Those must be integrated continuously from a moving state before
the requested running-bowling showcase is complete.
