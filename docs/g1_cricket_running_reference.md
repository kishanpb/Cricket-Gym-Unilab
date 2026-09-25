# Whole-Body Running Delivery Reference

## Airborne Rotation Repair

Current [complete two-hand results](../g1_cricket_results/running_momentum_v1/evaluation.json)
and [fixed-frame review](../g1_cricket_results/running_momentum_v1/running_motion_review.png)
use `278e09f6`. All 16 flight intervals per hand conserve discrete midpoint
angular momentum within 3.5e-12 Nms; COM remains exact. Arm error is below
5.46 mm, foot error below 1.75 mm, and all 136 poses per hand have no audited
intersection. One optimizer frame per hand (0.78 s) still reaches its evaluation
limit. Peak root orientation changes by 0.667 rad; the original model and its
joint/force limits are unchanged, and recovery is level again by 1.42 s.
The [maximum-root-rotation review](../g1_cricket_results/running_momentum_v1/maximum_root_rotation_review.png)
shows frame 63 for each hand; it is an offline target, not achieved balance.

The [independent fixed-curve refinement](../g1_cricket_results/running_momentum_v1/reference_rotation_refinement.json)
does not claim exact continuous conservation: finest pitch residuals are
7.72-8.02 Nm rather than the parent's 93.94-97.41 Nm in the first three flights.
All 96 rows remain, including the fourth flight. This is a reference repair,
not achieved physical motion or measured torque.

Both unchanged PD baselines still fall at 0.68 s, before release. At the recorded
0.16 s endpoint, the stance ankle pitch is -0.9102/-0.9100 rad (right/left trial),
past its original -0.87267 rad stop; the reference is approximately -0.7079 rad.
The violating joint is left/right ankle pitch respectively, before the first
takeoff. Full traces retain subsequent contacts and up to 0.090 rad excursions.
The next actual-control repair should inspect first-stance support loads and
ankle/hip control allocation, not spend more PPO on unchanged failed support
control or present the target animation as learned bowling. No new PPO was run.

All 342 new video frames decode nonblank; source hashes and the full review
are checked. All 50 focused running/reference/delivery tests pass with warnings
treated as errors; Ruff checks pass. Existing batting demonstrations and earlier failed-reference
comparisons are preserved. The full two-fork learned-bowling goal remains open.

The [fixed-curve refinement](../g1_cricket_results/running_ballistic_com_v3/reference_rotation_refinement.json)
(`0bf43b25`) retains both hands, four flight centers, six torque intervals from
20 ms to 0.625 ms, and both 0.1/0.01 ms velocity differences: 96 rows. Joint and
orientation splines preserve the original knots, with the same ballistic COM
and held ball. Finest pitch residuals remain 93.94-97.41 Nm in the first three
flights and 48.93-48.96 Nm in the fourth, with over 11 mm sampled foot clearance
and no sampled world contact. This confirms a defect of the reconstructed
reference curve, not a measured torque, new IK solution or physical rollout.
The finest two torque intervals differ by at most 0.48% in vector norm;
changing the velocity-difference interval changes torque by at most 0.0023 Nm.

This fixed comparison changes only offline root-orientation construction.
Native angular-momentum and wrist Jacobians include the mechanically held ball.
During each declared flight interval, an implicit-midpoint solve chooses root
angular velocity to preserve the preceding takeoff momentum while the original
29 joints solve the unchanged foot and arm targets. The same exact COM path
is retained. Stance uses a smooth relative-rotation recovery to level orientation
with matched landing angular velocity; no root forces/torques or pose overwrites
are added to the actual motor-only PD run. Original joint limits, force caps,
phase timings and full delivery structure remain unchanged.

Run both complete 136-frame references and unchanged PD baselines, retaining all
failed frames and episodes, in `g1_cricket_results/running_momentum_v1`:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_momentum_v1 --render --lane-offset 0.20 \
  --ballistic-parent g1_cricket_results/running_front_raise_v1 --conserve-momentum
```

Check discrete momentum residuals and independently refined derivatives,
geometry/foot/arm errors, continuity and actual physics. Momentum-conserving
target construction alone cannot qualify a running teacher or bowling showcase.

## Ballistic COM Repair Comparison

Latest complete pair: [exact-COM results](../g1_cricket_results/running_ballistic_com_v3/evaluation.json),
[flight/momentum audit](../g1_cricket_results/running_ballistic_com_v3/reference_flight_audit.json),
and [fixed-frame review](../g1_cricket_results/running_ballistic_com_v3/running_motion_review.png).
Source `f8c5af17` enforces COM to machine precision and reduces all sampled aerial
force components below 1.2e-10 N. This is a reference-consistency repair, not an
achieved physical flight. Both 136-frame references have no audited intersections
and arm error below 5.49 mm, but foot error reaches 2.043 mm and six IK frames
per hand hit 120 evaluations. Both unchanged physical PD episodes fall at 0.68 s,
before release, with joint excursions and unintended contact. No new PPO was trained.

The momentum extension (`ed001fe9`) uses
[MuJoCo angular momentum about subtree COM](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h)
for the entire robot and ball. Coarse airborne pitch-torque residuals reach
93.30 Nm; these are inferred from centered pose velocities and momentum
differences, not actual/applied loads. Their five-pose velocity windows can
include stance/flight boundaries, so temporal refinement is required. Next:
check angular-momentum consistency at finer time spacing and repair whole-body
orientation/counter-motion before more PPO; do not add root torque, weaken the
physical gate or treat exact COM alone as a valid running teacher.

All 1,016 generated video frames across the three candidates decoded nonblank
and each fixed-frame review was inspected. Historical v1/v2 duplicate videos
and tracking exports were pruned after verification; their complete reports,
pose arrays, flight audits and reviews remain. v3 retains both target videos
and both complete failed PD videos. Existing learned batting highlights are unchanged.
All 36 focused running/reference/flight/lane/delivery tests pass, including native
COM-frame angular-momentum semantics, analytic ballistic/torque cases, original
robot invariants and exact motor-only replay boundaries. Ruff checks pass.

Full-COM weighted IK (`dc1ad3e7`, `running_ballistic_com_v2`) brings each aerial
force component below 0.09 N, with no unexpected intersections, arm error below
5.49 mm and foot error 2.043 mm. Five/right and four/left IK frames still hit the
iteration limit. Both physical PD runs fall at 0.68 s with hand/thigh contact and
up to 0.090 rad joint excursion, before release. This is not qualified bowling.

The solver repair in `running_ballistic_com_v3` keeps those exact COM/limb
targets, but analytically eliminates root translation rather than fitting three
extra coordinates with a large COM penalty. Only the original 29 joints remain
optimization variables; original joint bounds, continuity, objective weights,
iteration budget and physical controller remain fixed. Record optimizer effort
and every geometry/physical failure for both hands. Exact COM projection is
offline reference construction only, never a simulation action or root force.

The vertical-only candidate (`c6f348f8`, `running_ballistic_com_v1`) reduces
inferred aerial vertical support from 393.6-475.5 N to below 0.10 N, but leaves
115.6-204.1 N of forward residual. All 136 poses per hand have no unexpected
intersection; maximum arm error is 5.27 mm, foot error 2.27 mm and COM height
error 6.04 micrometers. Four/right and five/left IK frames hit the evaluation
limit. Both PD baselines fall at 0.58 s, with joint excursions up to 0.0761 rad
and hand/hip contact. This candidate is not an accepted teacher or showcase.

The next bounded candidate extends the same COM repair to all three coordinates:
constant horizontal COM velocity through the run-up, matching the parent's
initial/gather COM positions, with the same ballistic vertical phases and C1
gather transition. Pelvis translation and original joints are solved together;
all foot/arm targets remain fixed. Run both hands into
`g1_cricket_results/running_ballistic_com_v2` using the command below with that
output name. The same geometry and native-physics checks apply. Original default
retargeting and frozen PPO actors remain unchanged; no training on failed targets.

### Vertical-Only Design (Historical)

The lane-clearance diagnostic exposed upward COM acceleration during aerial
run-up frames. The next fixed comparison changes vertical COM timing only,
using the same outward y=+/-0.70 m lane, original foot placements, arm targets,
joint/force limits and full 2.70 s delivery. Each 0.30 s run-up cycle has a
0.22 s stance and 0.08 s ballistic flight, with matching +/-0.3924 m/s vertical
COM velocities. Cubic stance and gather transitions preserve position/velocity
continuity. Whole-system COM includes the held ball; bounded offline IK may
adjust pelvis height and joints to achieve the COM and unchanged limb targets.
This is not root support or pose control during physics.

Evaluate both complete references and the unchanged PD baseline in
`g1_cricket_results/running_ballistic_com_v1`. Retain every optimizer failure,
intersection, tracking error and fallen rollout. Require COM flight residual
inspection, arm error below 6 mm, foot error below 2 mm and no unexpected
penetration above 1 mm before training. A feasible vertical reference alone
does not establish horizontal flight consistency, dynamic balance or bowling.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_ballistic_com_v1 --render --lane-offset 0.20 \
  --ballistic-parent g1_cricket_results/running_front_raise_v1
```

## Retained Reference

The requested G1 bowling video needs an approach, gather, planted overarm
delivery and recovery. The historical G1 arm policy over a frozen 0.4 m/s
walking prior does not satisfy that requirement. This separate first reference
retargets the earlier Gym-Cricket `DeliveryStrideMotion` choreography to the
original G1 links and joints; it does not copy its pose-overwrite runtime or
assign a release velocity.

Current reference: [front-raise targets and complete baseline](../g1_cricket_results/running_front_raise_v1/evaluation.json),
with [fixed-frame review](../g1_cricket_results/running_front_raise_v1/running_motion_review.png).
Both hands now solve the full motion with maximum arm error 5.25 mm and no
audited intersections. Physical balance remains failed; there is no learned
running-delivery demonstration yet. Earlier comparisons below are historical.

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

Reproduce the current bounded pair into a new output directory with:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_front_raise_reproduction --render
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
fixed-frame review remains. v2 videos/tracking exports were subsequently pruned
when the front-raise reference replaced it. At this stage, the shoulder
return path and dynamically feasible foot support still needed repair before
running-delivery RL or a bowling showcase could be accepted.

## Reverse Continuation Comparison

One bounded comparison keeps the v2 targets, constraints, weights, motor gains
and physical baseline unchanged, but solves the IK frames backward from the
recovery pose. The initial default seed is therefore attached to recovery
instead of approach. Outputs are restored to increasing physical time before
velocity construction or simulation. The same 12 rad/s continuity bound applies
in either solve direction. This tests whether the forward warm start trapped
the shoulder in a poor solution branch; it cannot itself establish balance.
The frozen `c0ac67f1` runner used `--reverse-ik --render` into
`g1_cricket_results/running_reference_reverse_v1`, retaining all errors/falls.
This rejected option has been removed from the current runner; use the frozen
commit for historical reproduction.

Backward continuation (source `c0ac67f1`) fails: maximum arm error is 110.06 mm,
13 frames per hand have unexpected penetration, and both physical episodes fall
at 0.36 s. No frame anywhere joins the forward and backward solutions within
2.61 rad maximum joint difference. Reversing the warm-start order is not a repair.

## G1-Adapted Front Raise

The next bounded revision changes only the bowling-arm wind-up path. It raises
the arm in front (2.4 to 1.6 to 0.18 rad from vertical) before the forward overarm
delivery; it does not force a continuous rearward circle through the shoulder
stop. The same full run-up, gather, plant, delivery and recovery phases remain.
Elbow flexion stays at the 8-degree target through release, and release-angle
rate remains +10 rad/s. All other targets, original robot limits, forward IK
settings, physical PD control and failure checks stay fixed. Test both hands
in `g1_cricket_results/running_front_raise_v1`; reject intersections and excessive
tracking error before using it for whole-body learning. This adaptation is not
an exact copy of human shoulder circumduction.

The complete front-raise pair (source `68a0c4c3`) converges at all 136 frames per
hand, has no unexpected penetration above 1 mm, and reduces maximum arm error
from 122.23 to 5.25 mm. The physical runs still fall at 0.62 s; the arm-path
repair is not a balance result.

## Stance-Load Motor Compensation

A separate fixed comparison adds a joint-torque feedforward term for estimated
weight-bearing at the reference feet. At every reference pose, use only foot
capsule endpoints whose bottom is within 2 mm of the pitch; solve nonnegative
normal loads against the floating-base gravity wrench, and map those loads
back to joint torque. Add only `-J.T @ normal_load / kp` to motor setpoints,
alongside the unchanged joint bias compensation and PD controller. No estimated
load is applied directly to the feet, root or environment. Existing motor-force
caps and joint-target limits still apply.

This uses the force/Jacobian mapping in the
[MuJoCo equations of motion](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework),
but is an approximate static load allocation, not full inverse dynamics or
predicted contact evidence. It excludes ball-holder load transfer, lateral
friction and acceleration. Frames without near-ground feet get zero load offset;
nonzero root residuals remain recorded rather than claiming a support solution.
The frozen `a929ddaa` runner tested the unchanged front-raise motion for both
hands with `--stance-feedforward --render` in
`g1_cricket_results/running_stance_feedforward_v1`.

Both physical episodes last 0.82 s rather than 0.62 s, but fail with foot/foot
and arm/wicket contacts and joint-limit excess of 0.01044 / 0.01115 rad. The
static solver also emitted an invalid-multiply warning, despite finite returned
values. This comparison is rejected, not a balance improvement or a certified
support solution. Its approximate load allocation and CLI option were removed
from the current runner; the frozen source, complete results and failure review
remain available. No further training or rollout uses that compensation.

Current media/tracking exports belong to `running_front_raise_v1`. Rejected
comparisons retain complete evaluation/pose arrays and fixed-frame reviews,
with duplicate videos/tracking exports pruned. The feedforward comparison uses
bit-identical reference poses, so its duplicate reference/exports were also
removed; its physical poses and complete estimated-load traces remain. Next:
use the repaired reference in a whole-body dynamic-balance curriculum, while
keeping actual stride, foot legality, elbow, release and recovery gates intact.
