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
