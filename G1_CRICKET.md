# Native G1 cricket foundation

This experimental task uses UniLab's floating-base, 29-DoF Unitree G1 model
with its original joint limits, inertias and motor force limits. A 0.70 kg bat
is rigidly mounted to the selected wrist: this is a declared mechanical
fixture, not a dexterous grasp. A 0.156 kg, 36 mm-radius free ball, pitch,
creases, wickets and visual practice-net background are added at construction.
The source robot XML and existing demonstrations are unchanged.

![Untrained G1 with the rigid wrist fixture in the practice scene](g1_cricket_results/initial_stance.png)

The current geometry is a **practice drill**, not a regulation-match pitch:
visual practice lines are at x=0 and x=18 m, the retained wicket is at x=-0.6 m,
and the incoming ball starts at x=2.5 m. Regulation wicket separation, delivery
length and no-ball/crease semantics remain unverified. The robot stands 0.30 m
to the side of the ball/wicket line instead of on the stumps.

`G1CricketBatting` uses the native Manager-Based environment and CPU MuJoCo
backend. `task=g1_cricket_batting/mujoco` selects its owner configuration;
`env.handedness=left` selects the mirrored wrist fixture and lateral stance.
All 29 actions are joint-position target offsets. Initial standing pose and
incoming ball velocity are reset conditions; no robot root/joint pose is
overwritten during a policy step. The incoming ball is a bowling-machine
curriculum, not learned bowling.

## Signals

Named public sensor views expose ball/blade, ball/pitch and ball/wicket contact
records, both feet's full collision-body support records, and bat-fixture force/torque.
Contact records use MuJoCo contact-frame forces in N and torques in Nm; world
positions, normals and tangents are also retained. Fixture wrenches are in the
attachment-site frame and include rigid-body inertial/gravitational loads;
they are not finger pressure or hardware tactile data. Policy observations
include geometry-level simulated touch flags, normal/shear loads scaled by
100 N, and fixture wrench components scaled by 100 (N or Nm respectively).

These are **end-of-control-step sensor snapshots** from the native backend,
with `post_step_forward_sensor=false`. They are not time-aligned final-pose
force solves, all-substep force peaks or integrated impulses. Decimation is
five physics steps per policy step. A public post-substep backend contract is
needed before making transient impact-load claims. Contact slot overflow
raises an error rather than silently truncating evidence.

## Incomplete

This is a construction/reset/control foundation, not a trained cricket result.
The current upright/action-rate reward is only a curriculum diagnostic, not
a validated batting objective. Ball-strike attribution, hit quality and
cricket legality gates, impact convergence, long-horizon stability, trained
PPO/A2C evaluation and showcase videos remain to be implemented and validated.
Bowling and constraint-switch release are not implemented; no launch impulse
is disguised as a learned throw. No robot-learning claim should be made from
zero-action tests or the previous non-G1 videos.

## Foundation checks

`PYTHONPATH=src uv run python -m pytest tests/envs/test_g1_cricket.py -q`
checks both hands, original inertias/joints/actuators, two floating roots,
native observation/control dimensions, public sensor views, repeatable reset,
finite short dynamics and absence of pose writes during policy steps. Initial
feet have no floor penetration after a 4 mm cricket-only reset-height increase.

[Zero-action diagnostics](g1_cricket_results/zero_action_diagnostics.json) retain
source hashes and both hands with seed 4, two identical environments per hand.
Both runs fall at 1.39 seconds, without detected bat-ball contact at the sampled
instants. Maximum recorded joint-limit excess is about 0.026 rad during the
fall. The largest support snapshot is 279 N on one foot; the largest recorded
fixture component is 10.52 N and 2.10 Nm. These failed stability diagnostics
are retained intentionally and are not a learned-policy or force-validation
success. The next required step is a bounded balance/stance curriculum before
attempting bat-ball interception, followed by full contact and legality gates.

## Native PPO pipeline smoke

A bounded **10-update / 960-transition** run exercised UniLab's existing
`unilab.scripts.train_rsl_rl` entrypoint, `uni_rl` integration and RSL-RL PPO,
not a custom learner. The right-hand task trained with seed 1, four environments,
24 rollout steps/update, a 127-64-64-29 actor, and CPU thread caps of two.
`model_9.pt` is the final checkpoint because iteration numbering starts at zero.

[Full evaluation](g1_cricket_results/ppo_smoke/evaluation.json) retains all 16
declared episodes: PPO and zero control, both hands, seeds 4101-4104. Reset is
currently deterministic, so changing seeds does not create independent test
conditions. Left-hand evaluation is an untrained transfer diagnostic.

| Control | Right-hand falls / duration | Left-hand falls / duration |
| --- | --- | --- |
| Zero offsets | 4/4, 1.39 s each | 4/4, 1.39 s each |
| PPO after 960 transitions | 4/4, 1.33 s each | 4/4, 1.34 s each |

No bat strike was detected at the sampled control instants. This is **pipeline
verification with worse stability than zero control**, not learned cricket.
No new video is advertised from this checkpoint. The run config, summary,
final checkpoint and complete evaluation are retained; initial checkpoint,
TensorBoard events and an irrelevant installed-runtime git snapshot are omitted.
Task source hashes and the foundation commit in the evaluation are authoritative.

Reproduce from the repository root with Python 3.13, CPU MuJoCo and the existing
UniLab/uni_rl dependencies installed:

```sh
export PYTHONPATH=src PYTHON_CPU_COUNT=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2
uv run python -m unilab.scripts.train_rsl_rl \
  task=g1_cricket_batting/mujoco training.device=cpu training.no_play=true \
  training.log_dir=g1_cricket_results/ppo_smoke algo.max_iterations=10 \
  algo.num_envs=4 algo.num_steps_per_env=24 algo.save_interval=10 \
  env.adaptive_chunk_size=false
uv run python scripts/evaluate_g1_cricket_smoke.py \
  --run-dir g1_cricket_results/ppo_smoke
```

On this machine the verified interpreter was
`../unilab_submission_checkout/.venv/bin/python`, selected with
`uv run --no-project --python <interpreter> python ...` for both commands.
`PYTHON_CPU_COUNT=2` also bounds native worker sizing on this Python 3.13/macOS
runtime; `env.cpu_ids` is a Linux-only affinity interface and fails on macOS.
The existing interactive/evaluation entrypoint is
`unilab.cli.eval_main`, which routes PPO to the same runner's playback loader;
the retained headless evaluator additionally enumerates every test episode and
fails instead of falling back to zero actions when a checkpoint is missing.

## Balance curriculum v1: bounded negative result

The separate `g1_cricket_balance_v1/mujoco` owner config preserves all robot
inertias, joints, position actuators, action scaling, bat fixture and ball/reset
dynamics. It changes the objective to a three-second stance curriculum, reusing
the stock G1 height, orientation, vertical/angular velocity and weighted-pose
penalties. Upright reward is 2 and action-rate penalty is -0.01. This is a
versioned curriculum bundle, not a single-axis causal ablation against the smoke.

Observations add pelvis-local velocity to the original explicit root-height,
ball-state, encoder, gravity, gyro and contact-snapshot inputs: **130 dimensions**.
These are privileged simulation measurements, not a vision-only policy or a
validated hardware tactile interface. PPO starts at Gaussian standard deviation
0.2 with entropy coefficient 0 and native KL-stop target 0.02; variance remains
learned. Actor/critic networks stay 64-64 and actions remain 29 joint offsets.

Two fresh seed-1 runs each collected exactly **48,000 transitions** (500 updates,
four environments, 24 steps/update): **96,000 total**. Right training took 27.36 s
at 1,838 transitions/s; left took 26.27 s at 1,916 transitions/s. Python native
worker sizing and Torch/BLAS thread caps were two. Each checkpoint was evaluated
on both hands with zero control and deterministic actor means, seeds 4101-4104,
without dropping falls. All **32 declared evaluation rows** are retained; seeds
repeat deterministic initial conditions and are not independent robustness trials.

| Checkpoint | Matched-hand PPO falls / duration | Opposite-hand PPO falls / duration |
| --- | --- | --- |
| Right-trained, 48k | 4/4, 1.26 s | 4/4, 1.23 s |
| Left-trained, 48k | 4/4, 1.11 s | 4/4, 1.10 s |

Zero control falls at 1.39 s in every row. Neither policy passed three seconds,
so the predeclared conditional ten-second stress evaluation was **not run**.
This budget did not produce learned stable stance or learned cricket; it does
not establish that the balance curriculum is exhausted. No showcase is claimed.
Before a larger run, investigate policy-update diagnostics and failure trajectories
rather than infer improvement from shaped training return alone.

[Right evidence](g1_cricket_results/balance_v1/right/evaluation.json) and
[left evidence](g1_cricket_results/balance_v1/left/evaluation.json) include checkpoint,
resolved-config and source hashes, actual transition counts, root-height and
drift measurements, joint-limit excess and every evaluation result. Each run
retains only its final `model_499.pt`, config, summary, evaluation and exported
scalar diagnostics. All iteration-indexed scalar histories are in `training_scalars.csv`
(redundant wall-time-indexed `/time` series are omitted);
`training_diagnostics.json` gives finite checks, extrema and final values. The
native logger did **not** emit KL or clip-fraction series, so the configured KL
stop cannot be presented as measured update quality. Earlier smoke evidence
remains a frozen record at its recorded source hashes.

With the same runtime and CPU environment variables above, run each hand
sequentially (these commands retrain, not resume):

```sh
for hand in right left; do
  uv run python -m unilab.scripts.train_rsl_rl \
    task=g1_cricket_balance_v1/mujoco env.handedness=$hand \
    training.log_dir=g1_cricket_results/balance_v1/$hand
  uv run python scripts/evaluate_g1_cricket_smoke.py --scope balance-v1 \
    --run-dir g1_cricket_results/balance_v1/$hand
  uv run python scripts/retain_g1_training_diagnostics.py \
    g1_cricket_results/balance_v1/$hand
done
```
