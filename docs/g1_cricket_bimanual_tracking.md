# G1 Cricket Motion Retargeting

The [balance-feedback follow-up](g1_cricket_balance_feedback.md) now completes
both reference-only three-second motions without falling. It still misses the
bat-tracking accuracy gate; learned cricket and running bowling are not done.

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
The original owner below has no support feedback. The newer supported and
balanced owners explicitly add motor-side feedforward and ankle feedback;
these analytic controller components are not learned by PPO.

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

### Grounded Support Pilot

The next reference puts the soles on the pitch, centers the reference COM over
the foot soles, settles the initial IK pose before differentiating it, and
checks blade clearance against both hands and shins. Joint targets retain a
0.10 rad margin from original stops. The supported action adds reference
velocity and static support feedforward through the original position motors;
nonnegative sole loads balance the floating base offline, without applying
external forces or writing robot state during rollout. This is a motor-control
baseline, not learned grasping or a dynamically feasible motion certificate.

The complete serial controller comparison retains position-only,
position/velocity and supported control for static guard and swing, both hands.
Supported guard completes three seconds; supported swing falls at 2.40 seconds
on both sides. Its pelvis-height stop differs from the PPO task's termination
rules, so these durations must not be compared directly with PPO survival.
The guard has no unexpected contact or hard-limit excursion and uses at most
43.3% of a motor's force limit. The swing reaches motor saturation and has
0.00140 / 0.00130 rad hard-limit excursions (right / left), despite the larger
reference margins. Keep those failures visible in the
[grounded comparison](../g1_cricket_results/bimanual_grounded_control_v2/evaluation.json)
and [original-reference comparison](../g1_cricket_results/bimanual_control_v1/evaluation.json).

Predeclared pilot: fresh independent right/left PPO, seed 1, 16 native mjbatch
environments, 512 updates and 196,608 transitions per hand, initial action noise
0.2, strict learner finite checks. Use the grounded reference and supported
action with unchanged tracking reward, original robot limits and physical
termination gates. Keep both final checkpoints and complete deterministic
reference-only/learned evaluations regardless of result. No checkpoint search
or mid-budget changes. This is still dry-swing development, not ball hitting,
held-out validation, running bowling or advertising footage.

Evaluation now independently replays every held-control interval and requires
exact native endpoint-state and sensor agreement. Per-interval peaks cover all
physics substeps: original hard-joint-limit excess, applied motor-load fraction,
grip separation, fixture force/torque and unexpected contact force/penetration.
These are uncalibrated simulator measurements, not hardware safety limits or
proof of realistic ball contact. Earlier endpoint-only reports are unchanged.

The supported owner also needs SciPy at initialization. With the retained
grounded references available, reproduce a fresh right-hand run with:

```sh
PYTHONPATH=src uv run --with scipy python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_supported_tracking/mjbatch env.handedness=right \
  algo.num_envs=16 algo.max_iterations=512 algo.save_interval=512 \
  training.device=cpu training.no_play=true \
  training.log_dir=g1_cricket_results/supported_reproduction_right
```

Use `env.handedness=left` and a distinct output directory for the other hand.
Evaluate the final run directory with `scripts/evaluate_g1_cricket_tracking.py`
and `--render`; it evaluates both reference-only and learned control.

#### Completed Grounded Pilot

Both independent runs completed 512 updates / 196,608 transitions, with final
`model_511.pt` and all 49 scalar series finite. Neither final actor improves
its own supported reference-only baseline; do not promote this pilot.

| Hand / control | Duration | Return | Hard-limit excess | Unexpected loaded contact |
| --- | --- | --- | --- | --- |
| Right reference | 2.36 s | 12.5296 | 0.00143 rad | None |
| Right PPO | 2.10 s | 10.5382 | 0.00369 rad | None |
| Left reference | 2.36 s | 12.5246 | 0.00126 rad | None |
| Left PPO | 2.00 s | 9.4203 | 0.01560 rad | Left elbow / right hand |

All four terminate on anchor height and reach motor saturation. The left PPO
collision peaks at 66.96 N and 2.17 mm penetration. Maximum substep grip gaps
remain below 1.136 mm; grip attachment alone does not establish safe motion.
Fixture force peaks for learned right/left control are 67.63 / 64.25 N, with
torque peaks 6.82 / 9.21 Nm. These are simulator loads, not hardware readings.

[Right complete evaluation](../g1_cricket_results/bimanual_grounded_v2/ppo_right/evaluation.json)
and [left complete evaluation](../g1_cricket_results/bimanual_grounded_v2/ppo_left/evaluation.json)
retain both controllers and every interval's substep measurements, with exact
native/serial state and sensor agreement. The
[right video](../g1_cricket_results/bimanual_grounded_v2/ppo_right/ppo_diagnostic.mp4),
[left video](../g1_cricket_results/bimanual_grounded_v2/ppo_left/ppo_diagnostic.mp4)
and [first/midpoint/final contact sheet](../g1_cricket_results/bimanual_grounded_v2/ppo_diagnostic_contact_sheet.png)
show complete failed development episodes at 0.5x, not selected successful shots.
Final checkpoints, configurations, complete scalar CSVs and traces are retained;
redundant event logs, initial checkpoints and reference-only videos are removed.

The standing-control defect is addressed, but moving support and arm clearance
are not. Next work must establish physically stable full-swing tracking before
another learned ball-hit claim; the running bowling sequence still needs
whole-body retargeting, plant and physical release. Any continuation needs a
separately declared budget and a reason to expect a different outcome.

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
