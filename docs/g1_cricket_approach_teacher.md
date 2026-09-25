# Physically Achieved Bowling Approach

The hand-prescribed first-step references have not survived a complete physical
rollout, even after three local PPO variants. This experiment instead records
the live, externally trained Unitree locomotion policy as a possible teacher
for the approach stage. It does not replace the required gather, legal overarm
release, recovery or locally learned whole-body cricket controller.

## Fixed Protocol

Use the existing delivery PitchV2 scene, original robot/holder/contact parameters
and Unitree deploy gains. Translate the reset robot and held ball together to
`x=-4.5 m`, `y=+/-0.7 m`; retain actual bowler and striker wickets and crease
geometry. No live pose writes or external root forces. All eight task actions
are zero: the external prior controls locomotion, with no learned arm residual
or release. Its asset hashes are pinned and verified; weights stay local-only.

At the existing 20 ms control rate, request forward velocity:

| Time (s) | Command (m/s) |
| --- | --- |
| 0-1 | 0 |
| 1-2 | Linear ramp 0 to 1 |
| 2-4 | 1 |
| 4-5 | Linear ramp 1 to 0 |
| 5-8 | 0 |

Keep every outcome from both hands, seeds 5301/5302, and 62.5/31.25 microsecond
physics: eight complete cases. No seed, checkpoint, trajectory or timestep
selection. Optional videos always show seed 5301 at the finer timestep for
both hands, including failure. A command of 1 m/s does not establish running.

## Acceptance and Export

Require all eight seconds, pelvis above 0.65 m and up component above 0.95,
original joint/motor bounds, no unintended contacts, holder error below 1 mm,
ball penetration at most 6 mm and no release. Actual foot collision footprints
must remain inside the +/-1.32 m corridor and on the declared wicket side;
root lateral drift must stay below 0.15 m. This is approach-lane checking,
not delivery-stride legality.
At foot-pitch contacts loaded above 1 N, require tangential contact-point speed
below 0.2 m/s and integrated slip below 3 cm per contiguous loaded stance.
Use solved-frame body velocity at the contact point, allowing normal foot roll;
do not equate ankle displacement with skating or reuse a planted-foot bound.

Require 2.4-3.6 m forward displacement, mean actual forward velocity between
0.7 and 1.3 m/s during seconds 3-4, and at least two measured airborne-to-loaded
landings per foot. Over both initial 0.5-1 s and final 7.5-8 s windows, require
root linear speed below 0.05 m/s, angular speed below 0.2 rad/s, maximum joint
speed below 0.5 rad/s, both feet loaded above 1 N, and mean support within 2%
of weight. These predeclared approach checks do not replace the full bowling gate.

Paired resolutions must agree on every gate and landing count, final root
position within 5 cm, peak holder force within 5% or 1 N, and peak holder error
within 0.1 mm. Simulated holder loads are uncalibrated fixture forces; geometric
touch is not hardware tactile pressure.

Every native interval is independently replayed with its actual held targets;
endpoint state and complete sensor values must match exactly. Retain full
control-boundary states, native qpos/qvel, executed motor targets, prior actions,
velocity commands, equality state, contact/support telemetry and interval
holder loads/impulses/touch. Export MotionLoader FK with the **recorded native
velocities**, not finite differences of sampled positions. The existing running
tracker adds controller terms absent from the prior, so poses alone are not a
replay contract: future teacher-target residual control must preserve executed
commands and be physically validated separately.
Integrated states carry end-of-substep times; contact/load/geometry and slip
measurements are from the solved-start frame one physics step earlier. Retain
the resolved owner, all source/mesh/compiled-model hashes, recorder hash and
dependency versions alongside these timing semantics.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python scripts/evaluate_g1_cricket_approach.py \
  g1_cricket_results/approach_teacher_v1 --render
```

Passing this approach study would yield teacher data, not a finished bowling
video, independently trained Menagerie policy or hardware demonstration.
