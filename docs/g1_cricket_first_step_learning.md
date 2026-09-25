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
