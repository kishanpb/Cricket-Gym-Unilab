# G1 First-Step Learning Pilot

This is a curriculum toward running bowling, not a replacement for the
complete approach, gather, overarm release and recovery. The measured parent
is the [failed first-step comparison](g1_cricket_running_startup.md#first-step-results):
both PD variants lift and land, then lose lateral balance.

Train both hands independently from scratch with seed 1, eight CPU mjbatch
environments, 256 updates and 24 steps/update: 49,152 transitions per hand.
Retain each final checkpoint regardless of outcome. PPO uses the existing
128x128 actor/critic, initial action noise 0.1 and a clipped 0.1-radian residual
on all 29 joint targets. Tracking rewards and termination thresholds are
inherited from the full running task; no reward increase is physical success.

The 10.5 s reference and static support torques are the frozen first-step
inputs. Support feedforward includes held-ball gravity and removes planned
air-foot load; native contact still supplies actual support. The actor and
critic additionally observe reference phase and left/right contact-present,
normal-load and shear-load snapshots. These are simulated sensor outputs,
not calibrated tactile taxels or force impulses. Existing integrated holder
force/contact observations remain. The holder stays enabled throughout this
curriculum; release timing is not learned or demonstrated here.

Both environments start from rest without reset perturbations. Actions only
change motor targets; no live root/joint state writes, external support force,
modified geometry, relaxed joint limits or higher motor caps are permitted.
The full-running configuration retains its existing bias and 1.82 s release.
The inherited trainer randomizes initial episode timeout counters, not motion
start frames or robot states; these initial partial horizons are not evaluation
evidence. Deterministic evaluation resets both counters and motion to zero.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python src/unilab/scripts/train_rsl_rl.py \
  task=g1_cricket_first_step/mjbatch env.handedness=right \
  training.log_dir=g1_cricket_results/first_step_tracking_v1/ppo_right
```

Use `left` and `ppo_left` for the second independent run. Tracking exports
come from `tracking.export_reference` applied to the complete frozen right/
left `running_first_step_v1` reference poses at 50 Hz, using the unchanged
compiled G1 delivery scene.

Evaluate zero residual and the final deterministic actor from frame zero,
seed 1, at the native training timestep and half that timestep. Retain all
eight outcomes, raw physical substeps and full videos including failures.
Require exact native endpoint/sensor replay and the unchanged first-step
settling, lift/landing, foot-path, contact, holder, motor and joint gates.
No selected interval, good training return or isolated landing qualifies a
run-up or cricket delivery. Do not advertise the intermediate videos.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 python \
  scripts/evaluate_g1_cricket_first_step.py \
  g1_cricket_results/first_step_tracking_v1/ppo_right --render
```

Use `ppo_left` for the other run. The evaluator applies the same
`replay_startup` telemetry and gates used for the parent controller comparison.
At each interval it copies the live environment's pre-step state into a
separate native replay model, applies the actual motor commands, collects every
physics substep, and requires byte-equal endpoint/sensor values in the
environment's storage dtype. It never writes the replay state into the live
environment. Reports retain all actions, motor commands and full physics states.

Implementation verification: 121 focused tests pass with warnings as errors,
including the unchanged full-running release behavior, both first-step
environments, exact state/sensor replay and bit-identical prior startup traces.
Both complete zero-residual environment smoke trials match native replay at
every interval (335 right, 334 left) and fail the first-step gate at about
6.70/6.68 s. Environment terminations and float32 state storage differ from
the standalone parent's native-double loop; comparison to PPO uses this same
environment baseline rather than treating the two runners as identical.

## Complete Pilot Results

Both hands completed 256 updates / 49,152 transitions independently, retaining
`model_255.pt`. All iteration-indexed scalar values are finite. Training task
source is `80177e8e58fd0ffedb363d000c3a9a223be31554`; evaluator source is
`0a56ed33940b344ae3fa636ddfed142992829d73`. The left process was paused during
local contention and resumed without restarting; wall-clock throughput is not
an algorithm or handedness comparison.

| Hand | Control | Physics step (microseconds) | Episode end (s) | Longest airborne interval (s) | Loaded landing (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Right | Zero residual | 62.5 | 6.70 | 1.9383 | 5.8705 |
| Right | Final PPO | 62.5 | 7.22 | 0.6571 | 5.8227 |
| Right | Zero residual | 31.25 | 6.70 | 1.9381 | 5.8703 |
| Right | Final PPO | 31.25 | 7.28 | 0.6864 | 5.8308 |
| Left | Zero residual | 62.5 | 6.68 | 1.9349 | 5.8661 |
| Left | Final PPO | 62.5 | 5.00 | 1.1304 | None |
| Left | Zero residual | 31.25 | 6.68 | 1.9348 | 5.8659 |
| Left | Final PPO | 31.25 | 4.98 | 1.0626 | None |

**All eight fail the complete first-step gate.** Right PPO delays instability
but does not settle; left PPO regresses and never lands. Both learned actors
also fail the initial quiet-hold gate. Every episode terminates on tracked
body-height deviation, with pelvis-orientation and foot-path failures in the
independent audit. No result reaches the 10.5 s horizon or releases the ball.
No selected hand, return increase or delayed termination is promoted.
All eight remain within joint and motor limits, without unintended loaded
contacts or ball penetration. Those checks do not clear the failed balance gate.

The [right report](../g1_cricket_results/first_step_tracking_v1/ppo_right/evaluation/evaluation.json)
and [left report](../g1_cricket_results/first_step_tracking_v1/ppo_left/evaluation/evaluation.json)
retain both timesteps, both controls, raw substeps, all actions, motor targets
and physics states. Each control interval passes exact independent endpoint
and sensor replay. The full [right PPO clip](../g1_cricket_results/first_step_tracking_v1/ppo_right/evaluation/ppo.mp4)
and [left PPO clip](../g1_cricket_results/first_step_tracking_v1/ppo_left/evaluation/ppo.mp4)
are development diagnostics at 0.5x, not running-bowling highlights; baseline
clips and fixed-frame review sheets remain alongside them.
All 1,230,080 substeps are finite, all 44 input hashes per report verify, and
all 1,286 frames decode nonblank at 960 x 540. Both review sheets were inspected.

Final checkpoints, run configurations/summaries and complete scalar CSVs are
retained. Redundant initial checkpoints, TensorBoard events and unrelated
installed-runtime git snapshots were removed after scalar verification.

## Next Bounded Comparison

The [uniform-start comparison](g1_cricket_first_step_uniform.md) below is now
complete: mixed handedness results, with neither hand passing the full gate.
The following records its original proposal, not a further unchanged rerun.

Do not extend these same from-rest runs unchanged. Compare uniform reference
state initialization during training, with the same reference, physical
model, reward, residual scale and budget. This is a proposed sampling-axis
change, not completed compute or evidence that the reference is dynamically
feasible. Reset-only held-ball position/velocity must remain consistent with
each sampled wrist pose. Explicitly force evaluation to frame zero, regardless
of the training sampler, and retain both hands, both controls and both physics
resolutions. Late-phase practice is a hypothesis; the full run-up, delivery,
recovery and both requested showcases remain unverified.
