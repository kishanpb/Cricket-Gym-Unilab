# Native G1 cricket foundation

This experimental task uses UniLab's floating-base, 29-DoF Unitree G1 model
with its original joint limits, inertias and motor force limits. A 0.70 kg bat
is rigidly mounted to the selected wrist: this is a declared mechanical
fixture, not a dexterous grasp. A 0.156 kg, 36 mm-radius free ball, pitch,
creases, wickets and visual practice-net background are added at construction.
The source robot XML and existing demonstrations are unchanged.

![Untrained G1 with the rigid wrist fixture in the practice scene](g1_cricket_results/initial_stance.png)

The current geometry is a **practice drill**, not a regulation-match pitch:
visual practice lines are at x=0 and x=18 m, the retained wicket is at x=-0.6 m,
and the incoming ball starts at x=2.5 m. Regulation wicket separation, delivery
length and no-ball/crease semantics remain unverified. The robot stands 0.30 m
to the side of the ball/wicket line instead of on the stumps.

`G1CricketBatting` uses the native Manager-Based environment and CPU MuJoCo
backend. `task=g1_cricket_batting/mujoco` selects its owner configuration;
`env.handedness=left` selects the mirrored wrist fixture and lateral stance.
All 29 actions are joint-position target offsets. Initial standing pose and
incoming ball velocity are reset conditions; no robot root/joint pose is
overwritten during a policy step. The incoming ball is a bowling-machine
curriculum, not learned bowling.

## Signals

Named public sensor views expose ball/blade, ball/pitch and ball/wicket contact
records, both feet's full collision-body support records, and bat-fixture force/torque.
Contact records use MuJoCo contact-frame forces in N and torques in Nm; world
positions, normals and tangents are also retained. Fixture wrenches are in the
attachment-site frame and include rigid-body inertial/gravitational loads;
they are not finger pressure or hardware tactile data. Policy observations
include geometry-level simulated touch flags, normal/shear loads scaled by
100 N, and fixture wrench components scaled by 100 (N or Nm respectively).

These are **end-of-control-step sensor snapshots** from the native backend,
with `post_step_forward_sensor=false`. They are not time-aligned final-pose
force solves, all-substep force peaks or integrated impulses. Decimation is
five physics steps per policy step. A public post-substep backend contract is
needed before making transient impact-load claims. Contact slot overflow
raises an error rather than silently truncating evidence.

## Incomplete

This is a construction/reset/control foundation, not a trained cricket result.
The current upright/action-rate reward is only a curriculum diagnostic, not
a validated batting objective. Ball-strike attribution, hit quality and
cricket legality gates, impact convergence, long-horizon stability, trained
PPO/A2C evaluation and showcase videos remain to be implemented and validated.
Bowling and constraint-switch release are not implemented; no launch impulse
is disguised as a learned throw. No robot-learning claim should be made from
zero-action tests or the previous non-G1 videos.

## Foundation checks

`PYTHONPATH=src uv run python -m pytest tests/envs/test_g1_cricket.py -q`
checks both hands, original inertias/joints/actuators, two floating roots,
native observation/control dimensions, public sensor views, repeatable reset,
finite short dynamics and absence of pose writes during policy steps. Initial
feet have no floor penetration after a 4 mm cricket-only reset-height increase.

[Zero-action diagnostics](g1_cricket_results/zero_action_diagnostics.json) retain
source hashes and both hands with seed 4, two identical environments per hand.
Both runs fall at 1.39 seconds, without detected bat-ball contact at the sampled
instants. Maximum recorded joint-limit excess is about 0.026 rad during the
fall. The largest support snapshot is 279 N on one foot; the largest recorded
fixture component is 10.52 N and 2.10 Nm. These failed stability diagnostics
are retained intentionally and are not a learned-policy or force-validation
success. The next required step is a bounded balance/stance curriculum before
attempting bat-ball interception, followed by full contact and legality gates.
