# G1 Cricket Motion Retargeting

The earlier cricket demonstrations are the visual target: two hands on the bat,
guard, backlift, downswing and follow-through; running approach, gather, legal
plant and overarm delivery. The earlier bowling implementation posed the body
kinematically and initialized ball release velocity. Those shortcuts cannot
establish physically learned G1 performance.

The new batting scene retains the original G1 geometry, inertias, 29 joints and
motor force limits. The bat is mounted to the top-hand wrist (left for a
right-handed batter). A compliant three-axis connect joins the other palm to
the lower handle. The handle is non-colliding as a declared mechanical fixture;
the blade and robot collision geometry remain active. This is not finger-grasp
learning or tactile hardware validation.

Offline inverse kinematics fits the bat pose and both grips while tracking foot
poses, center of mass and selected arm/torso clearances. Its output is a motion
reference, not an achieved trajectory. Export uses the exact compiled body-ID
layout, native quaternion differentiation and world-frame body velocities for
UniLab's existing MotionLoader. Joint names remain in SDK order.

`G1CricketBimanualTracking` learns residual targets for **all 29 joints** around
that reference. It does not use the frozen walking policy. Both the standard
MuJoCo and native mjbatch owners use the same task. Episodes start at clip
frame zero, with no reset noise, and truncate at clip end instead of teleporting
to the start. This first stage is dry-swing tracking, not trained ball hitting.
The diagnostic script additionally uses motor-side gravity and ankle feedback;
that hand-written diagnostic controller is not the PPO action owner.

## Development History

Three exploratory dry-swing prototypes preceded the retained diagnostic. The
first kept the grip within 1.42 mm at control snapshots but self-collided and
fell. Adding torso clearance and center-of-mass/foot targets removed the initial
torso collisions, but both variants still fell during the swing. Cross-arm
clearance and flexed-knee targets removed the remaining early wrist collisions;
the simple PD/gravity/ankle controller still fell. These are failed development
tests, not successful batting or evidence that the motion is impossible.
Intermediate media are superseded by one complete both-hand diagnostic.

## Reproduction

Generate references and a complete diagnostic in a new output directory:

```sh
PYTHONPATH=src uv run python scripts/retarget_g1_cricket_batting.py \
  --output g1_cricket_results/bimanual_v1 --render
```

Then select `task=g1_cricket_bimanual_tracking/mjbatch` for the CPU PPO owner.
The left-hand owner selects its own reference through `env.handedness=left`;
it must be trained and evaluated independently, not called trained by mirroring
a video or transferring a right-hand checkpoint without qualification.

No showcase gate is cleared here. Final acceptance still requires physical
stability, joint/motor/contact checks, actual blade-ball contact and useful shot
motion, complete running bowling with native release, and both-handed videos.
Bowling whole-body retargeting and learned showcase validation remain open.
