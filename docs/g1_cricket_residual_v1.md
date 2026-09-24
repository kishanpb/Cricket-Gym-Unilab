# G1 residual interception, v1

This is a staged airborne soft-toss task, not full cricket or learned bowling.
The 29-DoF robot keeps the external Unitree zero-command locomotion policy frozen.
A CPU PPO policy learns seven bounded position corrections on the bat-side arm.
The rigid single-wrist fixture is not a human grip. Neither motion nor ball
position is scripted during an episode; the delivery is initialized only at reset.

## Fixed Experiment

- Owner: `g1_cricket_residual_v1/mujoco`.
- Prior: upstream revision `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`, with the
  two asset hashes checked by the action term. Weights remain externally cached,
  not redistributed; see the prior report for the upstream licensing caveat.
- Prior history: original 480 values, including its own previous 29 actions.
  Residual actions never replace that history. Zero residual has exact native
  state/history/target parity tests for both hands.
- Residual limits in SDK arm order: `[.35, .35, .35, .35, .15, .15, .15]` radians.
- PPO: right hand only, seed 1, 4 environments, 24 transitions per update,
  2,080 updates (199,680 transitions), 64/64 MLP, initial action std .2,
  learning rate .0001, two minibatches, zero entropy coefficient.
- Physics/control: .002/.02 seconds; two-second episodes, original prior gains,
  body/wicket/bat guards, unchanged robot joints/inertias and .70 kg fixture.
- Reset: original joint jitter +/- .005 rad; ball launch
  `(1.1013225616, center_y + sign * offset, .8717760803)` m,
  velocity `(-2.5, 0, 0)` m/s. Right/left sign is +1/-1 and center_y is
  `.0432396123`/`-.0432296123` m. Uniform fixed offsets `[-.12, -.10, 0]` m.
  The away-side lanes were chosen by cold geometry before training; a pelvis-side
  lane was excluded because a miss could strike a foot and a simple candidate
  arm pose penetrated the hip. This does not establish dynamic feasibility.
- PPO observes 115 privileged simulator values: gravity, world root velocities,
  root height, 29 joint offsets/velocities, pelvis-frame ball-minus-blade position
  and ball world velocity rotated into pelvis axes, world blade quaternion,
  seven previous residual actions, 29 prior actions, and a handedness flag.
  This is not vision-only or a hardware-sensor deployment claim.
- Reward: upright, incoming ball/blade approach shaping, a one-shot sampled
  blade-contact bonus plus clipped forward velocity, failure penalty, and action
  rate penalty. Sampled reward is not the contact/success audit.

## Evaluation Declared Before Training

Keep all 96 rows: zero residual and final deterministic PPO checkpoint, both
hands, all three fixed lanes, eight seeds 4301-4308. Seeds 4301 and 4302 also
appear in implementation contract checks; the other six are untouched held-out
seeds. Left hand is untrained transfer, not a separately trained policy. No
selection of checkpoints, successful frames, lanes, or seeds may redefine the pool.

Require blade-first contact before ground/body/handle/wicket contact and outgoing
world x velocity strictly above 1 m/s immediately after blade separation. Also
require the full two seconds without a native termination, forbidden bat/body/
wicket/non-foot-ground contact, pelvis height below .48 m, pelvis up-z below .65,
joint-limit excess above 1e-6 rad, or actuator-limit fraction above 1+1e-6. A ball
landing after a valid hit is allowed; a pre-hit bounce or any ball/body/wicket
contact fails. Report per-physics-step contact forces, contact timing, simulated
tactile occupancy, and fixture loads separately from encoder/vision state.

Physics-rate evidence is admissible only after an independent serial replay of
each executed target interval matches the native endpoint and named sensors.
Overflow, non-finite state, or mismatched replay invalidates the audit. This is
discrete simulation evidence, not continuous collision or calibrated tactile
hardware validation. Report full counts and exact failures even if PPO does not
beat the zero-residual prior; do not advertise a successful learned shot yet.
