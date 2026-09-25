# Bounded Batting Acceleration Compensation

The exact grip/foot projection removed large static inverse-constraint loads
but changed actual swing error by less than 0.3 mm. This experiment tests one
new controller term on the same projected reference and frozen PPO actors.
It is not retraining, changed robot strength or a new motion trajectory.

## Controller

At each reference pose, compute the unconstrained generalized-force increment
`M(q) a + bias(q,v) - bias(q,0) - passive(q,v) + passive(q,0)`.
This follows the [MuJoCo force decomposition](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework),
but deliberately excludes native contact/equality stabilization forces. Preserve
floating-root derivatives when computing joint loads; zero the independent
parked-ball derivatives. Use the predeclared 20 ms central half-stencil,
including the same one-sided endpoint convention as the reference export.

Add only the 29 actuator-joint components to existing static support torque.
Clip the combined desired torque against both original motor force bounds and
the reference-state force reachable within the original position-command
bounds. Convert to a position-target offset using the original proportional
gain. Existing servo velocity feedforward appears exactly once; it is not
the passive joint damping term. Existing PPO residuals and balance feedback
are unchanged, with final command clipping and native motor-force limits.

This is an approximation: it does not allocate the extra base demand through
friction-limited feet or redistribute load through the three-dimensional lower
grip constraint. No root wrench is applied. The previous offline audit shows
about 55 N/34 Nm peak unapplied root increment and about 24 Nm peak joint
increment. Wider 40 ms derivatives reduce the joint peak to about 18 Nm;
that sensitivity is disclosed, not used to choose the favorable controller.
Reference-state clipping does not certify that runtime feedback stays unsaturated.
Every evaluation retains all 151 requested/bounded force vectors, clipping,
root increments, and complete runtime motor/contact/geometry traces.

## Predeclared Full Comparison

Use `bimanual_projected_v1` references for **both** baseline and candidate.
The only changed axis is `inertial_compensation`, default false. Retain all
16 baseline/compensated x right/left x reference-only/PPO x
31.25/15.625-microsecond outcomes, all from time zero through the complete
three-second swing unless terminated. Freeze final `model_255.pt` actors from
`bimanual_batting_learning_v1`, nominal one-bounce feed, gains, zero motor lead,
geometry, contacts, limits and all original physical/8 cm bat-path gates.
The bat target is still the original FK, not a phase-shifted easier metric.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python scripts/evaluate_g1_cricket_tracking.py \
  g1_cricket_results/bimanual_batting_learning_v1/ppo_right \
  --output g1_cricket_results/bimanual_inertial_v1/compensated_right_fine \
  --reference-directory g1_cricket_results/bimanual_projected_v1 \
  --inertial-compensation --contact-dt 0.00003125 \
  --bounced-delivery --compact-substeps
```

Repeat for both hands/resolutions. Omit only `--inertial-compensation` for
`baseline_*` cases. The complete report requires all eight files, both controls,
all source/checkpoint/recorder fingerprints and the candidate's full feedforward
audit; it retains failures and runtime saturation intervals:

```sh
PYTHONPATH=src:scripts uv run python scripts/report_g1_cricket_projected_batting.py \
  g1_cricket_results/bimanual_inertial_v1 --inertial-compensation
```

No new media is qualified from a partial result. Even a full nominal-feed pass
would remain a frozen-actor controller intervention, not learned generalization,
independent Menagerie training or completed running bowling.

## Complete Results

Source `1436ad35f046e14cc1420cee9ec584bf33e9a39c` was frozen before all
16 episodes. Every episode finishes three seconds, hits after exactly one
incoming bounce and recovers upright. All contact, penetration, grip, joint,
motor, height and root/joint-tracking checks pass, as do all eight contact-
resolution comparisons. Every episode still fails the unchanged 8 cm bat-path
limit: **compensation improves tracking but is not qualified or enabled by default**.

| Hand / controller | Baseline fine / finest (cm) | Compensated fine / finest (cm) |
| --- | ---: | ---: |
| Right / reference | 13.6854 / 13.6835 | 8.7837 / 8.8130 |
| Right / PPO | 14.7911 / 14.7761 | 9.7264 / 9.7211 |
| Left / reference | 13.6420 / 13.6381 | 8.7519 / 8.8129 |
| Left / PPO | 13.5799 / 13.5541 | 8.7024 / 8.6944 |

Across the complete paired pool, peak-error reduction is 48.252-50.646 mm.
All eight baseline rows exactly reproduce the projected-reference parent.
Frozen actors were not retrained, reference control also hits, and the feed is
nominal rather than held out; this is a controller improvement, not learned
interception generalization.

Each hand retains all 151 requested/bounded torque vectors. Twelve frames
require clipping, with maximum clipping 15.256 Nm right / 15.177 Nm left.
Maximum motor increments are 23.971/24.075 Nm. Unapplied base demands reach
54.411/54.785 N and 33.507/33.672 Nm; they are diagnostics, not injected forces.
All baselines and compensated reference controls reach a motor limit in 19 of
150 control intervals. Compensated PPO reaches a limit in 17/19 intervals
right fine/finest and 17/17 left. These are intervals containing at least one
saturated physical substep, not substep counts or force-limit violations.

The [complete report](../g1_cricket_results/bimanual_inertial_v1/summary.json)
retains all 16 outcomes and eight resolution comparisons. All 2,304,000 native
substeps pass exact independent endpoint/sensor replay; all 80 input hashes
and the recorder hash verify per evaluation. Report reproduction, 172 focused
tests with warnings as errors, Ruff and diff checks pass. No new training or
redundant video was produced; the earlier complete two-hand video is preserved.

The remaining bat-path error needs a separately declared controller or learning
change with the same full comparison, not a relaxed threshold. Running bowling
remains separate and unfinished: the next approach should use physically
achieved locomotion states as its teacher, then integrate gather, legal release
and recovery. A carry-only locomotion test cannot qualify that sequence.
