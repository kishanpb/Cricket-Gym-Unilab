# G1 Running-Delivery Learning Pilot

The policies/configurations below remain the frozen original pilots. The newer
[COM and root-rotation reference repair](g1_cricket_running_reference.md#airborne-rotation-repair)
has not been substituted into their training or claimed as new PPO evidence.
Its physical PD trials still fail at the stance ankle before takeoff; ground
support/control remains unresolved before a learned running-bowling showcase.

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
directory with `scripts/evaluate_g1_cricket_running.py --render`. It retains complete
reference-only and deterministic PPO episodes from frame zero and seed 1,
including all failures and full physics states. Every held-control interval must
match independent native state and sensor replay exactly. The unchanged delivery
gate checks actual stride/foot geometry, elbow, overarm release, stability,
joint/actuator limits, ball contacts, bounce and target corridor. Scheduled
release, tracking rewards or a short clip cannot replace those checks.
The videos replay those saved physics states at 0.5x, including terminal falls,
with simulated holder-load/contact readouts and a fixed-frame review per control.
They are not offline target animations or selected successful intervals.

This 2.7-second initial tracking curriculum may end before downfield ball flight
can be qualified. It is not a full-match, held-out, hardware, independently
retrained Menagerie-model, or learned-release claim. A usable showcase still
requires successful full start-to-recovery motion and the complete delivery
evaluation, including longer ball-flight observation and resolution checks.

## Complete Pilot Results

Both runs finished all 128 updates, 24,576 transitions per hand, with final
checkpoint `model_127.pt` (zero-based iteration index). All retained native
training scalars are finite. Left training was temporarily suspended while right
finished because concurrent runs caused local contention; wall-clock throughput
is therefore not an algorithm or handedness comparison.

| Hand | Control | Duration (s) | Minimum pelvis (m) | Joint excess (rad) | Peak holder load (N) |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Reference only | 0.56 | 0.6026 | 0.006269 | 15.758 |
| Right | Final PPO | 0.52 | 0.6831 | 0 | 34.399 |
| Left | Reference only | 0.56 | 0.6029 | 0.006115 | 15.661 |
| Left | Final PPO | 0.58 | 0.6860 | 0 | 40.692 |

**All four fail the unchanged full delivery gate; none reaches release.**
PPO remains more upright at its early stop, but both learned policies terminate
on the unintended-contact guard. The reference controls stop on tracked-body
deviation and also have substep self/wicket contacts and joint-stop excursions.
Every interval passes exact independent native endpoint and sensor replay.
Scheduled release is implemented, not demonstrated in these episodes.

[Right complete report](../g1_cricket_results/running_tracking_v1/ppo_right/evaluation.json)
and [left complete report](../g1_cricket_results/running_tracking_v1/ppo_left/evaluation.json)
retain both controls, all failures, contact loads and full state arrays.
[Right final PPO video](../g1_cricket_results/running_tracking_v1/ppo_right/ppo.mp4)
and [left final PPO video](../g1_cricket_results/running_tracking_v1/ppo_left/ppo.mp4)
show the entire start-to-terminal episode at 0.5x, not selected successful motion.
Reference-only videos and fixed-frame reviews are alongside them.

The right/left PPO recorded terminal endpoints show the hand intersecting the
bowler-end wicket by 10.404 / 3.291 mm. This is endpoint geometry, not a reconstructed
substep force. The early approach needs clearance and dynamic balance work;
simply increasing this pilot's training budget is not the next experiment.
The next bounded comparison should shift the complete reference lane outward
by 0.20 m for both hands, keep the frozen actors and all physical/cricket gates,
and retain both controls from frame zero. Any collision-free partial approach
still fails unless it proceeds through actual gather, plant, release and recovery.

Playback restores rounded free-body origins from the native model without
changing physics, fixed links, masses or state layout. The right baseline's
recorded states are byte-identical before and after this playback repair.
Final checkpoints, configuration/summary, complete scalar CSVs and finite-value
summaries are retained; redundant initial checkpoints, TensorBoard events and
an unrelated installed-runtime git snapshot are pruned after scalar retention.

## Frozen-Policy Lane Clearance Comparison

Predeclared next comparison: outward lane offset 0.20 m, changing the approach
from y=+/-0.50 to +/-0.70 m. Translate the robot and held-ball reference paths
together; do not shift the pitch, wickets, crease lines or world-body origin.
Preserve every joint angle, velocity, phase, gravity/balance term and scheduled
release time. No retraining or checkpoint selection is permitted in this comparison.
Evaluate both final actors and both reference-only controls from frame zero and
seed 1 with the same full delivery gate. Save all four full-start trajectories,
including early terminations; lane clearance alone is not bowling success.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/evaluate_g1_cricket_running.py \
  g1_cricket_results/running_tracking_v1/ppo_right --render --lane-offset 0.20 \
  --output g1_cricket_results/running_lane_clearance_v1/right
```

Use `ppo_left` and the `left` output directory for the other hand. The separate
output includes translated reference/tracking arrays and hashes of both their
original sources and the derived inputs. Parent evaluations remain immutable.

### Lane Results And Dynamic Reference Defect

The fixed pair (source `019ec088`) is retained in
[`running_lane_clearance_v1`](../g1_cricket_results/running_lane_clearance_v1).
Right/left PPO now reach 0.66 / 0.72 s, versus 0.52 / 0.58 s in the original
lane. Both stop on tracked-body deviation before release. Right has no forbidden
contact; left still records substep self/wicket contact. Both have zero hard
joint-limit excursion and reach the original motor-force cap. Reference-only
controls still stop at 0.56 s with self/wicket contact and joint-limit excursions.
All four fail full delivery qualification. All 129 video frames decode nonblank,
and every physical endpoint/sensor matches independent native substep replay.

An offline [centre-of-mass audit](../g1_cricket_results/running_lane_clearance_v1/reference_flight_audit.json)
then exposed why geometric retargeting alone is insufficient. At 0.26, 0.56,
0.86 and 1.16 s, three consecutive reference frames have both complete foot
collision shapes above 2 mm, with no sampled world contact. Their total-system
COM accelerates **upward** at 1.94-4.38 m/s2. For the 33.497 kg robot/ball system,
the discrete trajectory would require an extra upward force of 393.6-475.5 N,
rather than gravity-only flight. Horizontal residuals are also retained.

These are inferred reference-consistency residuals, not measured contact forces,
not applied support, and not certification that unsampled contact is absent.
The checker uses all system mass, original gravity, central position differences
and only complete aerial three-frame stencils before gather. Tests verify zero
residual for analytic ballistic motion and exclude stencils with grounded frames.
Reproduce with `scripts/audit_g1_cricket_running_flight.py` and the lane directory.

Next: repair the run-up's vertical phase so its airborne COM follows gravity,
with continuous stance/flight transitions; check the horizontal residual too.
Revalidate both full G1 references for geometry and dynamics, then rerun native
controllers before additional training. Do not add a root force, weaken contact
or foot-legality checks, or treat this lane comparison as a qualified teacher.
