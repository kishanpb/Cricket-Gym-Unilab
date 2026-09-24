# G1 Running-Delivery Learning Pilot

This native UniLab task uses CPU mjbatch to control all 29 G1 joints against
the repaired front-raise running reference. It includes approach, gather,
back/front plant, overarm motion and recovery, not an arm policy over a frozen
walking prior. Original links, inertias, hard joint limits, motor-force caps,
foot collisions and the finite-compliance ball holder are unchanged.

The first curriculum learns motor residuals, not release timing or grasping.
At reference time 1.82 s, the holder is disabled once; no ball velocity or pose
is assigned. At reset only, the ball starts at the wrist offset with the hand's
linear/angular velocity. Partial reset restores the holder for that environment
without touching other episodes. Initial robot-pose noise is disabled so the
held-ball reset is exact; unsupported reset perturbations fail explicitly.

Controls are reference position/velocity, reference-pose joint gravity bias,
the existing ankle/root/waist feedback, and bounded PPO residuals (scale 0.25).
This differs from the standalone PD comparison's current-state bias calculation.
There is no root support, pose overwrite, inverse-kinematics solve during a
policy step, or rejected stance-load compensation. Observations include the
reference, robot proprioception, peak holder force, world force impulse, hand
contact fraction and release state. These are uncalibrated simulated loads and
contact indicators, not hardware tactile taxels.

## Fixed Pilot

Train right and left independently, seed 1, eight environments, 128 updates,
24 steps/update: 24,576 transitions per hand. Use native PPO, 128x128 actor/critic,
initial action noise 0.2, strict learner finite checks, .0625 ms physics and
20 ms held controls. Keep the final checkpoint for both hands regardless of
outcome; do not select an earlier checkpoint or change the budget mid-run.
Training progress is not rollout qualification.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_running_tracking/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/running_tracking_v1/ppo_right
```

Use `left` and `ppo_left` for the other independent run. Evaluate each final run
directory with `scripts/evaluate_g1_cricket_running.py`. It retains complete
reference-only and deterministic PPO episodes from frame zero and seed 1,
including all failures and full physics states. Every held-control interval must
match independent native state and sensor replay exactly. The unchanged delivery
gate checks actual stride/foot geometry, elbow, overarm release, stability,
joint/actuator limits, ball contacts, bounce and target corridor. Scheduled
release, tracking rewards or a short clip cannot replace those checks.

This 2.7-second initial tracking curriculum may end before downfield ball flight
can be qualified. It is not a full-match, held-out, hardware, independently
retrained Menagerie-model, or learned-release claim. A usable showcase still
requires successful full start-to-recovery motion and the complete delivery
evaluation, including longer ball-flight observation and resolution checks.
