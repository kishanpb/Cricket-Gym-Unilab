# Positive Shoulder Arc: Launch and Braking v1

Six predeclared scripted feasibility cases, not learned bowling. Cold kinematics
suggest increasing shoulder pitch around +2 to +2.5 rad can move the nominal
fixture ball forward/upward. This does not establish dynamic feasibility: the
original positive stop is +2.6704 rad and the compiled controller is kp 40,
kd 10, capped at +/-25 N m. Strong damping and limited travel may prevent speed.

- Same left development seed 6301, native Batch, .0625 ms physics, full 4 s,
  guarded prior, holder/pitch, signed and legacy delivery gates. No injections.
- Original settle/raise timing, now with pitch +1.0 rad preload by tick 80;
  roll .35, yaw 0, elbow 1.4, all wrists 0.
- Drive reference +2.6 at tick 94, 98 or 102. Brake reference +1.6 at tick 112
  or 114. Full Cartesian product, six trials, no adaptation based on results.
- Release at the START of tick 114. Brake-at-114 cannot alter release velocity;
  brake-at-112 has two held control intervals. Hold braking reference until
  tick 150, recover to neutral by 190 and evaluate the entire episode.
- Every simulation substep independently replayed with exact native endpoint
  and named sensor parity. Preserve full traces, every failure, contact forces,
  signed elbow extension, shoulder stop margin, speed, phase torque statistics
  and first post-brake turning point. Targets within limits do not prove the
  achieved motion stays within limits. A legal pitch bounce is not a forbidden
  contact; diagnostic labels/phases are corrected without changing the gate.
  A null turning point may mean the arm was already moving nonpositively when
  braking began; read the boundary joint-velocity trace before interpreting it.

The inherited scalar search score is only a diagnostic; this grid does not
optimize it or use it for promotion. Only a full signed gate pass could become
a candidate demonstration, followed by learning and broader validation. Do not
silently change the release time, damping, limits, pitch length or contact model.

```sh
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_positive_arc.py
```

The runner refuses to overwrite existing results and checks frozen inputs at
completion. Canonical preflight/evaluation live in
`g1_cricket_results/positive_arc_v1/`; retain every trial, including failures.
