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

## Baseline Results

Source `4e921b84` reaches the scheduled physical release in all four cases,
but none passes. Right-hand recovery falls; left-hand recovery ends on an
unintended contact. The arm never reaches the upward shoulder-level crossing
before release, and forward ball speed stays below the unchanged 6 m/s gate.

| Hand | Physics step (us) | Episode end (s) | Release speed forward (m/s) | Peak loaded-foot slip (m/s) |
| --- | ---: | ---: | ---: | ---: |
| Right | 62.5 | 6.78 | 0.759 | 6.435 |
| Right | 31.25 | 6.54 | 0.765 | 7.748 |
| Left | 62.5 | 5.32 | 1.000 | 2.914 |
| Left | 31.25 | 5.36 | 0.992 | 2.882 |

These are measured failures, not a visible policy improvement. Right-hand
ball penetration reaches 16.6-16.8 mm and joint limits are exceeded; both hands
contact the ball with the hand collision geometry after release. The left
delivery stride is also out of order. All failed checks remain in the
[complete baseline report](../g1_cricket_results/moving_delivery_v1/summary.json).

Baseline finest-step videos: [right](../g1_cricket_results/moving_delivery_v1/right_finest/live_prior_reference_arms.mp4)
and [left](../g1_cricket_results/moving_delivery_v1/left_finest/live_prior_reference_arms.mp4).
The coarse-step videos and complete state/control traces are retained alongside
them; this is no timestep-convergence claim.

## Arm Servo Follow-Up

The next fixed comparison changes only arm velocity feedforward. Add
`--arm-velocity-feedforward` and use `g1_cricket_results/moving_delivery_velocity_v1`
as a fresh output directory. Desired arm velocity is multiplied by the
unchanged native motor damping/stiffness ratio and added to position targets,
which remain clipped to original joint ranges. Motor forces remain capped.
This compensates velocity damping; it does not increase motor authority or
turn the reference into a learned policy. Release timing, forward command,
physics, seed, four-case pool and all gates remain unchanged.

Source `e1be14f3` clears the overarm position and upward-crossing checks in all
four cases, and removes the baseline's post-release ball/hand contacts. Both
right-hand episodes finish eight seconds upright. Neither hand delivers a
valid ball: forward velocity is negative, and left recovery contacts the
bowler's wicket at 5.20 s. Terminal geometry inspection identifies the right
foot against `bowler_wicket_0`; the episode is not continued through it.

| Hand | Physics step (us) | Episode end (s) | Release speed forward (m/s) | Peak joint excess (rad) |
| --- | ---: | ---: | ---: | ---: |
| Right | 62.5 | 8.00 | -0.416 | 0.0551 |
| Right | 31.25 | 8.00 | -0.427 | 0.0364 |
| Left | 62.5 | 5.20 | -0.055 | 0.0000 |
| Left | 31.25 | 5.20 | -0.043 | 0.0000 |

Right recovery remains timestep-sensitive: total forward travel is 6.01 versus
4.57 m, and lateral drift is 39.9 versus 29.8 cm. Left drift is 60.6-61.0 cm;
its delivery footfalls remain in the wrong order. Peak loaded-foot slip remains
4.49-5.43 m/s. This improves a component of the motion, not the whole task.

[Complete follow-up report](../g1_cricket_results/moving_delivery_velocity_v1/summary.json)
and full finest-step videos: [right](../g1_cricket_results/moving_delivery_velocity_v1/right_finest/live_prior_reference_arms.mp4),
[left](../g1_cricket_results/moving_delivery_velocity_v1/left_finest/live_prior_reference_arms.mp4).
All eight baseline/candidate outcomes, four videos per controller, native state
and target traces, and 32 output fingerprints are retained. Both 48-entry
input manifests verify against their frozen source commits. All 2,528 video
frames decode nonblank at 960x540/25fps, and the motion review sheets were
inspected. The independent replay covers 1,208,000 physics substeps exactly.
43 distinct focused tests pass, including prior-preservation, release-state,
bilateral-resolution-pool and overwrite-refusal checks.

Next work must coordinate forward-swing release with the real arm state and
keep the moving robot clear of the wicket through recovery. The original
motor/joint limits and complete-episode gate stay in force. This is shared
UniLab G1 work on native CPU Batch, not independently trained Menagerie G1.
