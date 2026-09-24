# Impact events v1: physics-rate reward acquisition

This version changes one axis: acquisition of the first blade separation.
The old impact-v1 reward samples contacts every 20 ms and missed 9 of 40
contact-bearing trials in the complete frozen-policy audit. The new owner is
`g1_cricket_impact_events_v1/mujoco`; existing owners remain unchanged.

## Contract

Keep the 4 ms contact time constant, all geometry/inertia/friction/impedance,
collision exclusions, reset distribution, action limits, 115 observations,
7 residual actions, frozen prior, 20 ms control and shot gates unchanged.
Keep approach shaping and the nonnegative scalar separation curve
`5 * (1 + tanh(vx - 1))` unchanged. Acquire blade occupancy at every physics
solve and use the post-integration ball vx at its first subsequent absence,
even when contact and separation occur inside one control interval. Pay once
per episode at the containing interval's normal reward call. A later recontact
cannot replace the first exit or pay again. Partial reset clears only its rows.
This does not add contact observations to the policy or claim hardware tactile sensing.

The experimental adapter uses official MuJoCo Rollout once as the authoritative
physics trajectory, with the actual pool-owned models. It keeps warm-start
within each control interval and casts only final caches, matching the original
held-control path. The existing control callback is not suitable because it
clears warm-start every substep. Full trajectory capture costs more temporary
memory and copies; this is not an equal-compute claim. The local compatibility
boundary is recorded in ADR-0010, not offered as approved upstream support.

## Preflight

Freeze the final impact-v1 checkpoint, both hands, PPO/zero residual, three lanes,
seeds 4301-4308 and all 96 two-second rows at 0.125 ms. Require every original
non-return row field to match exactly, including native/serial state and sensors,
contact events, forces, penetration and safety/shot decisions. This verifies
physics and policy I/O, not successful shots. Reconcile each new native return
with the old return minus its actual paid separation bonus plus the unchanged
formula evaluated at the first physical separation. Allow absolute error <=1e-5
for float32 native reward accumulation, with zero relative tolerance. No missing
or selected rows, denominator changes, extra retraining or favorable checkpoints.

Separately test 1/80/160 physics steps, multi-environment models, pending forces
and torques, partial resets, solved versus integrated phase, contacts wholly
between samples, final-substep exits, interval-crossing latches and single payout.
Unsupported pre-control and forward-refresh combinations must fail explicitly.

## Bounded Training And Evaluation

Only after preflight passes, train fresh right-hand CPU PPO for 199,680 control
transitions: seed 1, four environments, 24 steps/update, 2,080 updates. Reuse the
previous optimizer/network/budget, without warm-start or checkpoint selection;
evaluate only the final checkpoint. Training physics is 0.25 ms. Evaluate the
same complete 96-row pool at both 0.25 and 0.125 ms, retaining every failure and
checking all-substep contacts, stability, limits and <=6 mm penetration.
Left-hand evaluation remains untrained transfer, not a separately learned skill.

An increase in reward from better event capture is not policy improvement.
Success still requires blade-first contact, strict first-separation vx >1 m/s,
and the existing complete physical gates. If it fails, retain the failure and
diagnose bat motion before changing another axis. Neither this repair nor its
tests establish material calibration, learned bowling, a tournament result or
a showcase video.
