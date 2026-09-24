# G1 Two-Hand Balance Feedback

The grounded reference can stand under static support compensation but falls
during the swing. Test one scalar gain on a pelvis-tilt/angular-velocity ankle
feedback term before any further training. This is a motor-reference diagnostic,
not PPO or achieved batting evidence.

Fixed study: grounded v2 references, original model, 1 ms physics, 20 ms held
control, full three-second motion or pelvis below 0.5 m. Both hands use gains
`-2, -1, 0, 0.5, 1, 2, 4`: exactly 14 deterministic cases, no adaptive search.
The feedback in the reference pelvis frame is gain times `0.7 * tilt + 0.1 *
angular_velocity`, clipped to +/-0.3 rad and added only to both ankle pitch/roll
targets. Motor targets and force limits remain bounded by the original model.
Negative gains check the sign hypothesis; gain zero must exactly reproduce
the retained supported baseline on all previously reported trace fields.

Retain every case. A feasibility candidate must finish, stay above 0.65 m at
every physics substep, have no unexpected loaded contact above 0.1 N, hard-limit
excess at most 0.0001 rad, grip gap below 6 mm, motor fraction at most 1, root
translation error below 0.15 m, joint RMS error below 0.2 rad, and bat-center
tracking error below 0.08 m throughout. Any solver warning or nonfinite aborts
the study. These are dry-swing diagnostic gates, not ball-hit, hardware or
running-bowling qualification. No learner or policy promotion follows merely
from a three-second completion.

Run `scripts/g1_cricket_tracking_control_audit.py` with `--balance-sweep`, the
retained grounded-v2 reference directory, and a new output directory.

## Completed Gain Study

All 14 cases are retained in
[the full report](../g1_cricket_results/bimanual_balance_feedback_v1/evaluation.json).
Gain zero exactly reproduces both retained baseline traces on all old fields.
Gains 2 and 4 complete the full motion on both sides; gain 4 has no unexpected
loaded contacts or hard-limit excursions, grip gap below 1.136 mm and joint
RMS error below 0.067 rad. It still fails bat tracking: maximum error is
0.15681 m right / 0.15678 m left, at 1.70 seconds. Thus zero cases pass all
declared gates. These controller trials did not train a policy.

## Feedforward Encoding Comparison

The supported action clips the sum of position, velocity and gravity offsets
to joint-position bounds. That sum is an equivalent position-motor torque
encoding, not an achieved joint position. Clipping can suppress the intended
feedforward torque even when the reference itself stays within physical limits.

Predeclared comparison: gain 4, both hands, clipped versus unclipped equivalent
motor targets, exactly four deterministic complete-or-failed episodes. Every
other input, timestep and the nine original feasibility checks stay fixed.
Physical joint limits and motor-force caps remain unchanged; no external
forces, model retuning or state overwrites. The unclipped condition is a
controller-encoding experiment, not permission for actual joint-limit excess.
The two clipped traces must reproduce the prior gain-4 cases exactly. Retain
all four physical trajectories for visual inspection. Run the same audit
script with `--feedforward-sweep` and a new output directory.

All four cases complete, but removing the encoding clip changes maximum bat
error only from 0.15681 to 0.15590 m (right) and 0.15678 to 0.15593 m (left).
Both conditions still fail the unchanged 0.08 m gate. Keep the clipped encoding;
this branch does not justify a runtime controller change. Full evidence is in
[the encoding report](../g1_cricket_results/bimanual_feedforward_encoding_v1/evaluation.json).
Its clipped traces exactly reproduce the gain-4 study. The gain study was also
replayed once after adding trajectory recording: all 14 prior traces are
identical, no additional candidates were searched, and current source hashes
are retained alongside the original script hash.

The recorded gain-4 clipped trajectories provide complete controller-only
[right](../g1_cricket_results/bimanual_feedforward_encoding_v1/right_control_diagnostic.mp4)
and [left](../g1_cricket_results/bimanual_feedforward_encoding_v1/left_control_diagnostic.mp4)
videos. They are not learned policies, ball-hit evidence or advertising footage.
The [contact sheet](../g1_cricket_results/bimanual_feedforward_encoding_v1/control_contact_sheet.png)
shows reset, the fixed 1.14 s backlift and 1.80 s drive frames, and the final
frame of each hand.

## Waist Compensation Comparison

At the common 1.70 s bat-error peak, pelvis pitch differs from the reference
by about 0.173 rad and waist pitch differs by about 0.097 rad. Translation
error is only about 0.014 m. Test counter-rotation of the waist against measured
pelvis tilt, without changing the ankle gain, references or physical gates.

Fixed follow-up: ankle gain 4, clipped motor encoding, waist compensation
gains `0, 0.5, 1, 1.5`, both hands: exactly eight deterministic cases. Subtract
gain times reference-frame pelvis rotation error from waist roll/pitch/yaw
targets; original target bounds and force limits remain. Gain zero must exactly
reproduce the prior clipped pair. Keep all trajectories and the same nine
feasibility gates. This is not a learned policy or a change to robot geometry.
Run the audit with `--waist-sweep` in a new output directory.

All eight cases are retained in
[the waist report](../g1_cricket_results/bimanual_waist_compensation_v1/evaluation.json).
Zero gain exactly reproduces both clipped baselines. Gain 0.5 adds a root
tracking failure; gains 1 and 1.5 introduce ground/self contact and other
physical failures. None clears all gates. Reject waist compensation and
retain the original waist targets; no runtime waist-feedback variant is added.

## Runtime Integration and PPO Pilot

`g1_cricket_balanced_tracking` adds only the gain-4 ankle-feedback action to the
supported owner. It retains clipped motor encoding and no waist compensation.
The action reads pelvis orientation/angular velocity through the entity API;
it never changes robot state or applies a root support force. Both standard
MuJoCo and native mjbatch owners complete the full three-second zero-residual
motion in contract tests, with independent substep replay, no unexpected
contacts, no hard-limit excursions above 0.0001 rad and grip gap below 6 mm.
Bat accuracy remains an unresolved diagnostic, not a qualified teacher label.

Fixed new learning experiment: independent fresh right/left PPO, seed 1,
16 native mjbatch environments, 512 updates / 196,608 transitions per hand.
Only the balance-feedback owner changes from the preceding supported pilot:
same grounded references, 0.25 residual scale, 0.2 initial action noise,
architecture, rewards, optimizer and strict finite checks. Keep both final
checkpoints, all scalar series, reference-only and learned complete evaluations,
and physical failure traces. No checkpoint selection or adaptive extra updates.
Learning may improve tracking but cannot establish ball hitting or running
bowling in this dry-swing task. Compare to its own balanced baseline, not a
selected older failed rollout.

The balanced pilot completes its fixed budget but fails evaluation: right PPO
terminates on incidental bat support at 1.10 s (return 5.9044), and left PPO
terminates on end-effector tracking at 1.86 s (return 8.6948). Both reference-only
controllers complete 3.00 s, with returns 17.0956 and 17.0986. Do not continue
these checkpoints or select earlier ones for presentation.

[Right evaluation](../g1_cricket_results/bimanual_balanced_v1/ppo_right/evaluation.json)
and [left evaluation](../g1_cricket_results/bimanual_balanced_v1/ppo_left/evaluation.json)
retain both controllers and every physical interval. Right PPO contacts
`wicket_1` at 43.00 N with 1.50 mm penetration; "incidental bat support" is
the termination name, not a ground-contact diagnosis. Left PPO reaches
0.01335 rad hard-limit excess. Maximum bat-path errors are 0.19379 / 0.73067 m
for learned right/left control versus 0.15650 / 0.15641 m for their baselines.
The slight baseline difference from the earlier serial control audit reflects
the native task's float32 target/state boundary; each evaluated episode itself
has exact independent substep endpoint and sensor replay agreement.

Both runs retain final checkpoints, configurations and 49 finite scalar series.
The complete [right failed video](../g1_cricket_results/bimanual_balanced_v1/ppo_right/ppo_diagnostic.mp4),
[left failed video](../g1_cricket_results/bimanual_balanced_v1/ppo_left/ppo_diagnostic.mp4)
and [first/midpoint/final sheet](../g1_cricket_results/bimanual_balanced_v1/ppo_diagnostic_contact_sheet.png)
are diagnostics, not advertising footage. Initial checkpoints, duplicate event
logs and reference-only videos were removed; reference-only traces remain.

Next fixed comparison: reduce only the residual position scale from 0.25 to
0.05 rad. Train fresh, independent right/left actors with the same seed, 16
environments, 512 updates / 196,608 transitions, initial noise, reward and
optimizer. Output is `bimanual_balanced_small_residual_v1`; this is not an
extension of the failed actors. All 29 joints remain policy-controlled around
the same physical reference controller. Retain complete final evaluations and
compare against the unchanged reference-only baseline. Completion alone cannot
qualify the bat path, ball contact, running bowling or a showcase video.
