# G1 Motor-Reference Timing

This is a bounded tracking experiment, not new RL training or a qualified
showcase. The retained dry-swing PPO actors have no ball observations. The
complete one-bounce delivery currently hits with both hands, but maximum bat
tracking error remains 0.1284-0.1371 m against the unchanged 0.08 m limit.

## Declared Comparison

Change only motor-reference lookahead: zero versus one 50 Hz motion frame
(20 ms). Advance reference joint position, joint velocity and gravity offset
together, clamping at the final reference frame. Keep observations, rewards,
root/waist balance feedback and measured tracking targets at the original
phase. Position-actuator velocity feedforward already exists and stays intact.
This does not write live robot poses or prescribe the ball after reset.

Retain all sixteen complete episodes: both hands, reference-only and frozen
PPO, both timing settings, and 31.25/15.625-microsecond physics. Use the same
`model_511.pt` actors, ankle gain 4, waist gain 1, root-position gain 4,
reset-only (4, 0, 1.3) m / (-3, 0, 4) m/s delivery and compact native recorder.
No joint/motor limits, geometry, grip constraints, contact materials, launcher,
episode duration, pool or metric thresholds change.

Candidate qualification requires every candidate episode and both-resolution
comparison to pass the existing gates. A missed/extra bounce, non-forward
exit, instability, excessive tracking error, unintended contact, joint/motor
violation, grip separation, penetration or failed contact-resolution check
closes this candidate as a qualified improvement. Keep all outcomes even if
an individual row looks better. The old zero-lead reports remain historical
evidence at their recorded source commits; rerun zero lead under current source
for the paired comparison rather than silently changing old hashes.

## Reproduction

From the repository root with its CPU environment and native mjbatch recorder:

```sh
for lead in 0 1 3; do
  for hand in right left; do
    for resolution in fine finest; do
      dt=0.00003125
      if [ "$resolution" = finest ]; then dt=0.000015625; fi
      PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python scripts/evaluate_g1_cricket_tracking.py \
        g1_cricket_results/bimanual_balanced_small_residual_v1/ppo_${hand} \
        --output g1_cricket_results/bimanual_motor_lead_v1/lead${lead}_${hand}_${resolution} \
        --waist-tracking-gain 1 --root-position-gain 4 --lookahead-frames "$lead" \
        --contact-dt "$dt" --bounced-delivery --compact-substeps
    done
  done
done
PYTHONPATH=src:scripts uv run python scripts/report_g1_cricket_motor_lead.py \
  g1_cricket_results/bimanual_motor_lead_v1
```

No new video is required to interpret a failed tracking comparison. Preserve
the existing one-bounce video and inspect newly rendered full episodes before
making any improved-motion or showcase claim.

## One-Frame Result and Bounded Continuation

All sixteen declared episodes completed. The fresh zero-lead traces reproduce
all eight retained baseline episode rows exactly. All eight timestep pairs
pass the unchanged contact-resolution checks. Every one-frame episode still
fails only bat tracking: finest reference/PPO maxima are 0.11271/0.10173 m
for right and 0.11251/0.10785 m for left. A completed one-bounce hit alone does
not clear the 0.08 m gate.

The baseline peak occurs in follow-through at 1.64-1.66 s. Projecting its
position error onto the instantaneous reference tangent suggests about
77-79 ms lag, with another 0.048-0.071 m orthogonal error; this local projection
is a diagnostic, not a prediction or qualification result. The measured
one-frame improvement motivates one further candidate at **three frames
(60 ms)**. Append eight episodes, both hands/controllers at both resolutions,
with no other change. Retain the complete zero/one-frame comparison in the
same canonical report and qualify each candidate separately. No two-frame or
larger-lead sweep, checkpoint selection or threshold change is authorized by
this protocol. The reproduction command above includes this continuation;
`--candidate-frames 1` reports just the initial comparison.

## Complete Results

All **24 episodes** complete three seconds and hit after exactly one incoming
bounce. All pass pelvis-height, grip, original joint/motor limits,
unintended-contact, root/joint tracking, forward-exit and penetration checks.
Every episode still fails the unchanged 0.08 m maximum bat-path error bound.
All twelve adjacent-resolution comparisons pass force, penetration and exit
velocity checks. These are uncalibrated simulated contact loads, not hardware
validation or a robustness claim beyond this nominal feed.

Maximum bat-path error at 15.625 microseconds:

| Lead | Right reference | Right PPO | Left reference | Left PPO |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 0.13689 m | 0.12840 m | 0.13654 m | 0.13707 m |
| 20 ms | 0.11271 m | 0.10173 m | 0.11251 m | 0.10785 m |
| 60 ms | 0.08861 m | 0.09785 m | 0.08861 m | 0.09421 m |

The larger lead reduces follow-through lag but moves the right-hand maximum
to downswing, around 1.34-1.36 s. It is not uniformly better motion tracking:
finest PPO returns fall from 17.5801/17.4397 (right/left, zero lead) to
17.2753/17.1396 (60 ms). Finest outgoing vx remains positive at 2.1387/2.0714
m/s for the latter PPO pair. No successful row is selected to override the
failed complete-candidate qualification.

The [canonical report](../g1_cricket_results/bimanual_motor_lead_v1/summary.json)
retains all outcomes, twelve resolution comparisons and sixteen paired timing
comparisons. Its twelve input evaluations contain **3,456,000** audited
physical substeps with exact native endpoint/sensor replay. All 74 input
hashes and the native recorder hash per evaluation verify. The eight zero-lead
episode rows exactly reproduce the earlier retained reports; old files are
unchanged. No training, checkpoint search, new video, threshold relaxation or
policy promotion follows from this experiment. The default lead stays zero.

This closes the two declared constant-lead candidates as complete solutions.
The next learning task needs explicit bat-state tracking and ball observations,
with full both-hand contact/physical evaluation, rather than another blind
constant-lead sweep. Running approach, legal bowling release/recovery and
qualified humanoid advertising videos remain unfinished.

Validation: 119 focused environment, control, contact-report, native substep
and constraint tests pass with warnings treated as errors. Tests cover zero
lead compatibility, coherent position/velocity/gravity advance, unchanged
measurement phase, clip-end clamping, invalid leads, distinct candidate pools,
missing cases and a failed single row blocking its entire candidate. Report
reproduction, Ruff formatting/lint and diff hygiene also pass.
