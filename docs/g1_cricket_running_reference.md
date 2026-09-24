# Whole-Body Running Delivery Reference

The requested G1 bowling video needs an approach, gather, planted overarm
delivery and recovery. The historical G1 arm policy over a frozen 0.4 m/s
walking prior does not satisfy that requirement. This separate first reference
retargets the earlier Gym-Cricket `DeliveryStrideMotion` choreography to the
original G1 links and joints; it does not copy its pose-overwrite runtime or
assign a release velocity.

Fixed initial design: both hands, 2.70 s at 50 reference frames/s, 0.55 horizontal
stride scale, 2.20 m total forward travel, y=+/-0.5 m approach, gather 1.20 s,
back landing 1.42 s, front landing 1.65 s and intended release 1.82 s. The
planned front ankle is behind the popping crease; actual loaded footprint and
stride legality still require the existing full delivery audit. Root/foot/arm
targets and original side-swapped stride timing are solved through bounded
offline IK on all 29 joints. No link scaling, collision disabling, joint-limit
relaxation or added root support is permitted. The existing finite-compliance
ball holder and corrected pitch response remain explicit.

Keep every reference frame and its optimization, foot-error, arm-error and
penetration diagnostics. Target animation is labeled **OFFLINE IK TARGET, NOT
PHYSICS**; it is not a learned demonstration, contact proof or valid teacher
merely because the rendered motion looks convincing.

Then test exactly one fixed PD baseline per hand, original motor-force limits,
20 ms held targets and .0625 ms physics: position/velocity reference plus
joint-side bias compensation, existing gain-4 ankle/root feedback and gain-1
waist feedback. Set initial position/velocity once. Thereafter use native
physics only; at the intended release disable the holder without altering ball
position or velocity. Stop on pelvis below .48 m or any solver/nonfinite error.
Retain complete-or-failed trajectories, every substep joint/force/contact peak,
and actual release state if reached. Rendering replays recorded poses in a
separate MjData after physics; it must not alter the physical trajectory.

No training or new showcase is authorized by this reference diagnostic. Before
learning, inspect the generated motions, resolve reach/collision defects, and
integrate the whole-body reference into the native task with the existing full
delivery gates: physical stride, all-geometry foot legality, geometric elbow,
overarm release, stability, ball flight and contact-resolution checks. Do not
replace those gates with planned ankle positions, intended release time or
offline inverse-kinematics success.

Reproduce the initial bounded pair with:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_reference_v1 --render
```

Original batting highlights and the current two-hand soft-toss diagnostic remain
unchanged. This is the missing bowling path, not a replacement for batting.

## First Reference And Repair

The initial pair (`running_reference_v1`, frozen source `13d46a5c`) solves all
136 IK frames per hand, with foot error below 1.83 mm, but contains 16 frames
per hand with hand/hip penetration. Arm-segment error reaches 0.16021 m and a
shoulder-yaw branch switch demands 75.02 rad/s. Both physical PD baselines fall
at 0.62 s, before release, with hand/hip and inter-foot contacts. No teacher,
running policy or showcase is accepted from that pair.

The bounded second revision keeps the same spatial/phase targets and physical
controller. It uses shoulder-roll/elbow/wrist-roll origins for anatomical limb
directions, leaving distal wrist orientation as a softer target. It adds the
observed hand/hip clearance pairs, allows torso yaw redistribution, and limits
adjacent joint-reference changes to 12 rad/s. This is a reference smoothness
bound, not a claim about hardware-certified velocity limits. Original hard
joint limits, inertias and force caps remain unchanged. Evaluate both hands
once in `running_reference_v2`, retaining every failed frame and complete-or-
fallen physical episode; do not train through geometric defects.

## Second Reference Results

The complete [second pair](../g1_cricket_results/running_reference_v2/evaluation.json)
uses frozen source `ebdb057a`. Both 136-frame targets have zero unexpected
penetrations deeper than the 1 mm audit threshold and a maximum joint-reference
speed of 12 rad/s. Foot error stays below 1.83 mm and intended-release arm error
is 4.61 mm. However, follow-through arm error reaches 122.23 mm as the IK solution
reaches shoulder and waist bounds. One frame per hand (1.16 s) also reaches the
optimizer iteration limit. This is not an accepted training reference.

Both physical episodes still fall at 0.62 s, before the gather or release.
They have no loaded unexpected contacts or joint-stop excursions, but do reach
the motor-force cap during the episode. Final pelvis heights are 0.4723 / 0.4730 m
and root errors are 0.3198 / 0.3198 m (right / left). The scene has no floating-base
support or pose correction. Removing the reference intersections did not solve
dynamic balance; the original pose-scripted human trajectory cannot simply be
replayed by the robot's motors.

The [fixed-frame review](../g1_cricket_results/running_reference_v2/running_motion_review.png)
includes approach, gather, plant, intended release and recovery targets above
both complete failed PD rollouts, including their terminal frames. The target
videos are explicitly offline animation, not physical or learned bowling.
Full pose arrays, source hashes and every error/physical trace remain available
for both revisions. Superseded v1 videos/tracking exports were pruned; its
fixed-frame review remains, and v2 retains all four videos and tracking exports.
Next work must resolve the shoulder
return path and dynamically feasible foot support before running-delivery RL
or a bowling showcase can be accepted.
