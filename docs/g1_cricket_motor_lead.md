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
for lead in 0 1; do
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
