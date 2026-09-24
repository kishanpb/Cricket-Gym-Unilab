# G1 forward-swing reward experiment

Parent: the final impact-events-v1 right-hand PPO checkpoint and its complete
two-resolution evaluation. Its first-impact headroom diagnostic finds receding
bat-point velocity in all 24 matched-hand trials, shoulder-yaw/elbow residual
clipping throughout the preceding 100 ms, but no arm motor force-limit hits in
that window. This does not establish a physical speed ceiling or justify larger
motor limits. The experiment changes only approach shaping, not robot dynamics.

Owner: `g1_cricket_swing_v1/mujoco`. Replace the proximity-only term with:

```text
approach_rate = 5 * exp(-(distance / 0.18 m)^2)
                * clip(blade_center_world_vx / (1 m/s), 0, 1)
                * (ball_world_vx < 0) * (not any_blade_contact_seen)
```

The existing reward manager integrates this rate over the 20 ms control step.
Compute blade-center velocity from the public tracked cricket-bat body's world
linear velocity plus angular velocity crossed with the center-to-body offset.
Use existing cached solved-phase body/sensor values, not a finite difference,
post-forward state, new sensor, added policy observation or scripted swing.
Positive world x is the desired outgoing direction for either hand. This is
blade-center approach shaping, not measured contact-point speed or a shot gate.

The physics-rate observer sets `hit_seen` before reward evaluation. Suppress
approach shaping for the entire control interval containing first blade contact
and every later interval, including recontact; collision-induced blade velocity
must not earn approach shaping. Zero/receding blade velocity earns zero approach
reward. Speeds >=1 m/s saturate shaping rather than encouraging unlimited speed.
The first-separation event latch and bonus `5 * (1 + tanh(first_exit_vx - 1))`
remain unchanged and pay once. Partial reset clears only reset rows through the
existing parent implementation. No additional reward history is introduced.

Keep the frozen locomotion prior, 115-input/7-action policy, residual limits,
asset inertias/motor limits, bat/ball compliance, collision exclusions, reset,
opponent lanes, reward weights, failure/action-rate terms and optimizer fixed.
Use a fresh right-hand seed-1 CPU PPO run: four environments, 24 steps/update,
2080 updates / 199680 transitions, 0.25 ms physics and 20 ms control. Do not
resume or select intermediate checkpoints. Retain only final checkpoint,
config, summary, complete scalar history and evaluation evidence.

Before training require analytical velocity/reward/reset tests and both-hand
native fixed-action physics/observation parity with the parent. Evaluate the
final policy on all 96 identities at each of 0.25 and 0.125 ms: both hands,
zero residual/PPO, three lanes and seeds 4301-4308. Left PPO is untrained transfer,
not an independently learned left skill. Require exact zero-residual physical
baseline reproduction, excluding returns because the reward changed. Preserve
all failed trials and original numerical-comparison tolerances. Pin parent,
headroom, checkpoint, config, summary, source and external-asset hashes.

Raw success remains clean first blade separation with vx >1 m/s, full two-second
stability, unchanged ball-contact/body/fixture/joint/actuator gates and blade
penetration <=6 mm. More approach reward or contacts is not a successful shot.
Stop on non-finite training or failed provenance/parity. Close this bounded
candidate without promotion if the final full pool still has zero valid shots;
do not extend its budget or relax gates after seeing results. Material calibration,
both independently learned hands, bowling and showcase readiness remain separate.
