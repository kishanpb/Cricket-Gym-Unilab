# G1 Actor Imitation Initialization and Bounded PPO

This experiment follows the paired contact-model transfer, not a successful
scripted demonstration. Its purpose is to learn a closed-loop bat-arm residual
from an imperfect controller and then improve it with PPO. It is not learned
whole-body locomotion, a grasp policy, material calibration or learned bowling.
The frozen Unitree locomotion prior and rigid wrist bat remain explicit.

## Fixed Model and Budget

- Owner: `g1_cricket_bc_v1/mujoco`, `G1CricketImpactV2`, 2 ms bat-ball time
  constant, 62.5 us training physics, 20 ms control and two-second episodes.
- All geometry, masses, actuators, gains, limits, observations, reward and
  termination rules remain the existing opt-in model's settings. No new clock.
- Teacher: `pmppppp_s10`, right hand, all three existing lanes, seeds 5301-5308.
  Retain all 24 episodes, including failures and genuine early terminations.
  Record pre-action cloned 115-dimensional observations, applied raw seven-axis
  commands, tick and episode identity, and every full episode gate result.
- Targets come directly from the existing raw `action_at` function, already
  arctanh-transformed; never apply arctanh or tanh to those targets again.
- Actor: existing 115->64->64->7 ELU MLP without observation normalization.
  Fit only its MLP using Adam, learning rate .001, 2,000 updates, seed 1.
  Each 96-sample batch contains 32 samples drawn with replacement from each
  tick range [0,10), [10,20), [20,100). This balances actual retained phases;
  it is not success filtering. Missing phases or nonfinite values stop the run.
- Retain every imitation loss and phase MSE. Assert fresh critic, initial .2
  action-noise distribution, empty PPO optimizer and iteration remain unchanged
  through BC. Save the BC-only checkpoint with explicit model/data provenance.
- Continue that actor with a fresh PPO state for exactly 256 updates, four
  environments and 24 steps per update: 24,576 transitions. No budget extension,
  cherry-picked intermediate checkpoint, changed reward or widened action bounds.
  Existing PPO settings remain unchanged, including learnable noise after BC.
  Count actual wrapper transitions, not only logger estimates.
- Save the final checkpoint and native scalar diagnostics. Compact redundant
  TensorBoard/model snapshots only after portable scalars and final saves verify.

The single change in learning method is actor imitation initialization before
PPO. This is a new run on the explicitly declared 2 ms parent model, not a
same-physics comparison to earlier learned 4 ms policies. Old actor, critic and
optimizer states are not restored. Failed teacher/BC episodes do not by themselves
cancel PPO; nonfinite state, solver errors or contract violations do.
The experiment upgrades the configured diagnostic NanGuard to a local strict
subclass: invalid control, observation, reward or captured physics state raises
before the framework can sanitize rewards. Historical framework code is unchanged.

## Complete Development Evaluation

Evaluate zero residual, BC-only and final PPO on both hands, all three lanes,
all eight development seeds 4301-4308, both 62.5/31.25 us timesteps and both
MuJoCo/native-mjbatch executors: 576 rows. The seeds are reused development
contexts, not held-out evidence. Left is untrained transfer until separately
trained. No selected success rows or imitation-loss-based promotion.

Use the retained full physics-step replay gate: complete native two-second
episode, blade-only first ball contact, outgoing velocity strictly >1 m/s,
maximum penetration <=6 mm, no forbidden ball/guard contact, pelvis height
>=.48 m and up component >=.65, original joint/actuator limits, finite state,
and exact native/replay endpoints. Preserve complete contact/fixture-load
evidence. Require exact outcome/impact equality across executors, unchanged
fine-timestep gate outcomes, existing 5%/absolute scalar tolerances, and fixture
force/torque agreement within max(5%, 1 N/0.1 N m) before qualifying a context.
Contact load is simulated and does not establish safe hardware grasping.

Checkpoint loading must verify the frozen preflight, teacher dataset and
physical-model metadata before actor loading. The generic sim2sim dimension
guard alone does not establish contact-model identity. Keep experiment input
hashes and runtime/native-executor identity; refuse to overwrite retained stages.
Record failures without silently rerunning a stopped run.

## Reproduction

With the documented CPU runtime and external Unitree prior installed:

```sh
PYTHONPATH=src:scripts uv run --no-sync python scripts/train_g1_cricket_bc.py preflight
PYTHONPATH=src:scripts uv run --no-sync python scripts/train_g1_cricket_bc.py collect
PYTHONPATH=src:scripts uv run --no-sync python scripts/train_g1_cricket_bc.py train
PYTHONPATH=src:scripts uv run --no-sync python scripts/train_g1_cricket_bc.py evaluate
```

Use two CPU/Torch threads. `tests/scripts/test_g1_cricket_bc.py` covers config
scope, actor-only updates, sample phases, provenance rejection, transition
counting and complete evaluation pairing. This experimental script does not
change the generic UniLab training API; existing ADR-0010/0011 ownership and
native execution boundaries remain. Training/evaluation completion alone does
not authorize a showcase, upstream PR, hardware claim or social publication.
