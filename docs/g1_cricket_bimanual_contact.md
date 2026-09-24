# G1 Two-Hand Soft-Toss Diagnostic

This tests the existing whole-body two-hand swing against a real simulated ball,
not a new interception learner. Both final small-residual PPO actors remain
frozen; their observation has no ball state. Mechanical top-hand attachment and
second-hand connect constraint remain explicit, not articulated finger grasping.

The preceding root-feedback comparison retained all 20 episodes. Every positive
root gain removes the waist-induced drift without joint-stop or unexpected-load
violations. At gain 4, right/left PPO peak bat error is 0.12687 / 0.13353 m and
root error is 0.02677 / 0.02904 m. All episodes still fail the original 0.08 m
bat-path gate. Do not discard that failure or call the teacher fully qualified.

## Fixed Protocol

Use waist-position gain 1, root-position gain 4, ankle gain 4, unchanged original
joint ranges, motor force caps and two frozen actors. Compare both hands and
reference-only/PPO control at 0.0625 and 0.03125 ms physics, holding each action
for 20 ms. At each resolution run both the out-of-play dry control and one
nominal soft toss: exactly 16 complete or failed episodes. No adaptive trajectory
search, checkpoint selection, new training or regulation-bowling claim.

The toss resets the ball at `(4, 0, 1.1)` m with velocity `(-2.8, 0, 6.5)` m/s.
It is a lofted practice toss; gravity and native collisions govern everything
after reset. Bat/ball uses the existing versioned 2 ms response, and ball/pitch
uses the existing corrected 2 ms pitch response. No impact velocity assignment,
root support force, pose overwrite, robot rescaling or collision disabling.

Keep the original nine physical/tracking checks. Additionally require loaded
blade contact, first force-free blade exit vx >1 m/s, and maximum penetration
across every ball contact <=6 mm. Report misses and every unintended loaded
contact, including robot/ball contact. All held-control intervals must reproduce
native state and sensor endpoints exactly in independent serial substep replay.
Loads are uncalibrated simulated contact diagnostics, not hardware tactile data.

For matched contact-bearing resolution pairs, compare peak force, maximum
penetration and first-exit velocity against max(5% of the finer value, 1 N /
0.1 mm / 0.05 m/s respectively). These are finite-grid diagnostics, not physical
calibration or asymptotic convergence. A force-free exit is the first substep
after a positive blade normal load; later impacts remain in the complete audit.
Normal-force time integral is reported separately from peak force; it is not a
signed world impulse. No claim of learned interception or showcase readiness
follows from a nominal hit.

Run `scripts/evaluate_g1_cricket_tracking.py` on each retained parent with
`--waist-tracking-gain 1 --root-position-gain 4 --contact-dt 0.0000625`
(then `0.00003125`), adding `--soft-toss` for the toss condition. Every run must
use a distinct new `--output` under `bimanual_soft_toss_v1`. Render both complete
finest-resolution toss evaluations irrespective of success, with frozen-actor
and mechanical-grip labels. Retain all reports and one combined PPO video.

Running bowling remains a separate incomplete whole-body task: approach,
gather, legal plant, overarm release and recovery cannot be replaced by this
batting diagnostic.

## Complete Results

All 16 episodes finish 150 held controls (three seconds), with exact independent
native-state/sensor replay throughout. The float32 endpoint clock reads
2.999997854 s, so completion uses the 150 controls and timeout, not a double-
precision clock equality. Every episode passes joint stops, grip, support
height, motor-force caps, unintended contacts, root tracking and joint tracking;
every episode still fails the unchanged bat-path accuracy gate.

All eight toss rows have loaded blade contact, forward first-exit vx >1 m/s,
and all ball penetrations below 6 mm. Finest-resolution results:

| Hand / control | Exit vx | Blade penetration | Blade peak force | Peak bat error |
| --- | --- | --- | --- | --- |
| Right reference | 1.91764 m/s | 3.260 mm | 683.49 N | 0.13838 m |
| Right PPO | 1.45590 m/s | 2.849 mm | 621.36 N | 0.12987 m |
| Left reference | 1.91256 m/s | 3.224 mm | 681.79 N | 0.13806 m |
| Left PPO | 1.70166 m/s | 3.125 mm | 664.84 N | 0.13902 m |

These are forward deflections followed by a pitch bounce, not powerful drives
or held-out batting performance. Reference-only control hits too; this does
not establish that PPO learned interception or improves shot quality.

Three of four complete resolution comparisons pass. Left reference-only pitch
peak force differs by 5.115% (1477.64 versus 1405.74 N), failing the declared
5% bound; all blade-load, penetration and exit-velocity comparisons pass.
Do not omit the reference failure or call the full contact model qualified.
No additional training or parameter search followed this fixed experiment.

[All 16 summaries and four comparisons](../g1_cricket_results/bimanual_soft_toss_v1/summary.json)
link back to eight complete evaluation files. Input hashes pin frozen source
`45b067a1`, both checkpoints, references, original robot XML and meshes.
`scripts/report_g1_cricket_bimanual_contact.py` reproduces the summary and
optionally assembles media with `--media`; it verifies evaluation inputs first.

The [two-hand slow-motion diagnostic](../g1_cricket_results/bimanual_soft_toss_v1/two_hand_ppo_soft_toss.mp4)
contains both complete finest-resolution PPO episodes, right then left, including
terminal labels: 350 decoded nonblank frames at 25 fps (0.5x). The
[fixed-frame sheet](../g1_cricket_results/bimanual_soft_toss_v1/soft_toss_contact_sheet.png)
shows guard, backlift, contact-time, follow-through and recovery without selecting
different times per hand. These remain developmental videos, not a final launch
or running-bowling result. Individual render files are reproducible from each
evaluation command; duplicate videos need not be retained with the combined clip.
