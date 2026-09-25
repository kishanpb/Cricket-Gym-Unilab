# Measured-Command Whole-Body Learning

This task replaces the unachieved, hand-prescribed approach with motor commands
from the [physically executed approach study](g1_cricket_approach_teacher.md).
That source still fails foot-slip and some lateral-drift checks. Reproducing it
is a controller-transfer check, not qualification of the gait or full bowling.

## Controller and Reset

At each 20 ms interval, the 29-joint action is:

```text
target = recorded_motor_target[frame] + 0.1 * clip(policy_action, -1, 1)
```

There is no online ONNX policy, added gravity/velocity feedforward, ankle/root
balance term or joint-target clipping. Original physical motor force caps and
joint limits remain. Some measured ankle targets lie outside the joint position
range while actual joints stay within it; clipping these commands would change
the baseline. The actor has motion, robot-state, previous-action, holder-load,
simulated-touch and phase observations. The critic also observes body poses.

Reset uses measured joint positions and native velocities, and measured robot
AND ball free-root states, preserving compliant holder displacement. Free-root
angular velocities are converted from native body coordinates to the world
coordinates required by the reset API. No ideal wrist attachment, soft-limit
pose clipping or live pose writes replace the recorded motion.

Uniform resets sample only the 400 command-bearing frames. All 401 measured
states remain available for reference and validation, but state 400 has no
outgoing command. Frame 399 executes once and truncates; a from-rest episode
executes exactly 400 intervals. Partial reset preserves other batch rows.

## Reproduction

The Hydra owner is `task=g1_cricket_measured_approach/mjbatch`. It uses native
CPU mjbatch, eight environments, uniform reference starts, a 128 x 128 PPO
actor/critic and 0.1 initial action noise. The bounded training budget is 256
updates with 24 steps/update: 49,152 transitions per hand. This configuration
is not evidence that local training or physical qualification has occurred.

First verify all eight retained hand/seed/timestep traces from frame zero:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python \
  scripts/replay_g1_cricket_measured_approach.py \
  g1_cricket_results/measured_approach_replay_v1
```

Every source seed is retained, not selected for controller compatibility.
The comparison records complete states, executed targets and holder peaks,
reports exact equality separately from numeric error, and never clears the
source's slip/drift failures. It does not perform an independent new contact
audit or demonstrate that an open-loop target sequence is a learned policy.

Before PPO, add learning signals for the failed contact slip and lane tracking,
with explicit sensor timing. Existing motion imitation alone would reward
following the sideways-drifting teacher. A training return cannot clear the
unchanged physical gates; final actors must undergo complete from-rest,
both-hand, two-resolution evaluation, including failed outcomes.
The subsequent moving gather, legal overarm delivery and recovery must be
continuous with the approach. No stopped-tail splice qualifies as bowling.

## Zero-Residual Results

The [complete replay report](../g1_cricket_results/measured_approach_replay_v1/summary.json)
retains all eight hand/seed/timestep cases. Each executes exactly 400 intervals
and truncates without a fall. Every recorded native state, all executed motor
targets and interval peak holder loads match the source exactly, including
the terminal state: maximum qpos and qvel error are both zero in every case.
All 57 recorded input hashes and eight output hashes verify. This establishes
command-preserving transfer only; all source slip/drift failures remain.

## Contact-Aware PPO Protocol

`task=g1_cricket_approach_learning/mjbatch` adds only foot-frame sensors and
learning terms to the measured-command task. Sensor additions do not alter
robot geometry, gains, motor limits, collision parameters or holder compliance.
Each foot's solved-frame origin, linear velocity and angular velocity share
the contact sensor's substep. Contact-point velocity is
`v_origin + omega x (contact_position - origin)`; remove its normal component
and include contacts with normal load above 1 N. Per-foot maximum contact
speed is squared and averaged over every equal-duration physics substep.
The reward subtracts five times the sum of these two mean-squared speeds.
An additional unit-weight lane reward is `exp(-(lateral_error / 0.1 m)^2)`
against the initial lane, not the drifting reference. Other rewards remain.
These integrated learning costs do not replace peak-speed or stance-slip gates.

The actor/critic additionally observe per-foot RMS slip, average normal load
and lane error. Native replay tests compare the contact-point speed at every
substep of frame 183 for both hands, including the measured peak-sliding phase;
sensor timing is not approximated with integrated-end entity velocities.

Train fresh right/left PPO actors independently with seed 1, eight CPU
environments, 256 updates and 24 steps/update (49,152 transitions per hand).
Uniform starts use the fixed seed-5301/62.5-us teacher for the corresponding
hand. Retain final actors without checkpoint selection. Evaluate zero residual
and each final actor from frame zero at 62.5 and 31.25 us, all eight outcomes.
The identical teacher commands remain fixed at both evaluation timesteps.
Use the original complete approach gates and independent native contact replay;
render both finer-grid PPO outcomes including any failures. No running-delivery
or local-learning success claim follows from the teacher replay above.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python \
  src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_approach_learning/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/approach_learning_v1/ppo_right
```

Repeat with `left` and `ppo_left`; do not tune the budget or gates after results.
