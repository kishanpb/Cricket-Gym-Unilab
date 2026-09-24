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
