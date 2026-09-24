# Terminal arm-command attenuation

This bounded, scripted diagnostic follows the retained reversal search. It is
not policy training, a public showcase, generalization or physical calibration.
The parent `pmppppp_s10` completed both engines' full shot gate at 0.25 ms but
failed the unchanged 6 mm penetration gate at 0.125 ms (6.620823 mm).

## Frozen experiment

Keep the same robot, bat/ball model, frozen prior, reward, reset, action limits,
seven-joint sign vector and reversal at tick 10. Change **only tick 19's normalized
arm-residual scale**, holding that command from 0.38 to 0.40 s. Scales are exactly
`[0, 0.25, 0.5, 0.75, 0.875, 1]`; zero command from tick 20 is unchanged.
Use inverse tanh to deliver the requested normalized command through the existing
action owner. No state/pose writes or model changes are allowed.

The parent first contacts at approximately 0.3836 s. Attenuation therefore keeps
the entire approach identical until 0.38 s and tests whether reducing the last
command can lower overlap while maintaining first-exit vx strictly above 1 m/s.
This effect is a hypothesis, not a monotonicity or feasibility assumption.

Every scale gets a full 100-tick / 2 s trial at dt 0.00025 and 0.000125 s in both
MuJoCo rollout and native mjbatch: **24 retained rows**, right hand, seed 4301,
center toss. There is no prefix screening, adaptive search, early successful-row
selection or training. Native termination ends a failing episode normally and
remains a failure; retain all rows. Scale 1 is replayed first in each context and
must exactly reproduce the parent outcome, termination flags and impact details.

A scripted witness requires all four full shot gates, exact cross-executor
outcomes and impact evidence, and the unchanged timestep comparator. Full gates
include blade-first contact, no forbidden contacts/guard occupancy, stability,
joint/actuator limits, first-exit vx >1 m/s, maximum penetration <=6 mm across all
impacts and actual 2 s completion. Do not loosen thresholds or report a single
passing engine/timestep as success. Every native interval still receives independent
serial state/sensor replay plus the separate contact-impulse diagnostic pass.

## Reproduce

Use the existing runtime and external pinned prior from the shared G1 setup:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_terminal_residual.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_terminal_residual.py
```

Preflight freezes all parent input/source hashes, the new runner/test/contract,
the parent report, versions and installed executor hashes. Results are retained
in `g1_cricket_results/terminal_residual_v1/evaluation.json`, without replacing
previous reports or videos. All force/tactile quantities remain simulated and
uncalibrated. A negative result rejects only this finite terminal-command family;
a positive result is still scripted, single-reset evidence, not learned cricket.
