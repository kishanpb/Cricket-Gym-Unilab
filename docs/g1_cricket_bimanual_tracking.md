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

## Whole-Body PPO Pilot

Both hands were trained independently with CPU native mjbatch, seed 1, 16
environments, 512 updates and 196,608 transitions each. The deterministic
development evaluation starts at the first reference frame and keeps every
episode through termination. The right policy lasts 1.56 seconds; the left
lasts 1.68 seconds. Both fail the anchor-height gate during downswing. The
reference-only controller terminates at 0.32 seconds on incidental bat support
for both hands. This is not a completed swing, held-out evaluation or ball hit.

The first pilot inherited the learner's disabled finite checks; its complete
saved scalars and independent evaluation are checked for finite values, not
claimed as a strict training-time finite-state audit. Future runs enable learner
finite checks explicitly. The evaluator restores the original visual meshes
from the same generated scene because the physics compiler strips them; it
checks state-layout and body-transform compatibility before rendering, while
measurements still use the original physics model.

Predeclared continuation: resume each hand's final `model_511.pt` for exactly
512 additional updates (another 196,608 transitions per hand), using the same
seed, reference, reward, action scale, model, episode horizon and termination
gates. Enable learner finite checks and explicitly select the existing critic
observation group. Keep both outcomes regardless of improvement; no adaptive
checkpoint selection. Evaluate each final checkpoint against reference-only
control and render complete failed or completed episodes. A completed dry swing
still cannot establish contact quality, running bowling or a showcase result.

### Completed Continuation

Both continuations completed their full budgets: **393,216 cumulative
transitions per hand**, with final checkpoint `model_1022.pt`. The learner's
resume counter repeats label 511; each retained scalar CSV has 512 update rows,
so transition counts, not the last checkpoint label, define the budget.

| Hand | Initial PPO duration / return | Continued PPO duration / return | Final result |
| --- | --- | --- | --- |
| Right | 1.56 s / 6.3136 | 2.00 s / 7.7729 | Anchor-height failure |
| Left | 1.68 s / 6.7105 | 2.64 s / 10.2905 | Anchor-height failure |

Neither completes the three-second motion. Both tip backward during the swing;
control-boundary grip separation remains below 0.491 mm and 0.630 mm, while
reported configured joint-limit excess reaches 0.02467 and 0.01810 rad. These
are endpoint diagnostics, not a complete substep safety/contact qualification.
No ball-hit, running, hardware-load or policy-promotion claim follows.

[Right complete evaluation](../g1_cricket_results/bimanual_v1/ppo_right_continued/evaluation.json)
and [left complete evaluation](../g1_cricket_results/bimanual_v1/ppo_left_continued/evaluation.json)
retain all reference-only and learned traces. The initial pilot's reports,
final parent checkpoints, saved configurations, all iteration scalars and
continued final checkpoints are retained alongside them. All 49 scalar series
per run are finite. Redundant event files, intermediate checkpoints and the
learner's unrelated installed-checkout diff were removed after scalar export.

[Right failed development video](../g1_cricket_results/bimanual_v1/ppo_right_continued/ppo_diagnostic.mp4)
and [left failed development video](../g1_cricket_results/bimanual_v1/ppo_left_continued/ppo_diagnostic.mp4)
show each complete episode at 0.5x, including terminal failure. The
[contact sheet](../g1_cricket_results/bimanual_v1/ppo_diagnostic_contact_sheet.png)
uses the first, midpoint and last physical frame of both final episodes. These
are diagnostics, not replacements for the earlier showcase videos.

## Reproduction

Generate references and a complete diagnostic in a new output directory:

```sh
PYTHONPATH=src uv run --with scipy python scripts/retarget_g1_cricket_batting.py \
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
