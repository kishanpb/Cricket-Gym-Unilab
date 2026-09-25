# Native Running Preview

This experimental controller evaluates future native physics before applying
one bounded 20 ms motor command. It is not a learned policy, and its planning
objective cannot qualify a bowling delivery or replace full physical gates.

## Fixed Design

The complete momentum-repaired running reference, original robot, joint limits,
motor caps, 62.5 microsecond physics and scheduled release remain unchanged.
Every decision evaluates two rounds of 24 candidate six-command sequences:
a 120 ms horizon. Seed 1 and two native rollout threads are fixed for both
hands. Candidates include the shifted previous plan, nominal reference PD,
the contact-aware foot-tracking controller's offset, and sampled smooth motor
offsets. Sampling scales are 0.12 and 0.06 rad; all commands are clipped to
the original limits. Only the selected first command reaches the live robot.

The native [MuJoCo rollout API](https://mujoco.readthedocs.io/en/stable/python.html#rollout)
advances scratch states with the same timestep, warm start and holder-release
schedule. No live pose/root write, root force or prescribed ball velocity is
introduced. Prediction uses the same physical model with a smaller sensor
set: six foot-position values and five signed inter-leg distance probes.
The real evaluation retains the original full force/contact diagnostics.

The objective averages root-position, orientation, joint and foot tracking
with weights 200, 40, 2 and 100. Every predicted substep is checked for joint
excursion above 1e-6 rad, inter-leg penetration above 1 mm, pelvis height below
0.48 m, non-increasing simulation time and non-finite state. Unsafe candidates
receive a penalty, not a guarantee: if all candidates are unsafe the chosen
command can still fail. Full actual contact and cricket gates remain required.
The five distance probes are planning heuristics, not complete collision coverage.

The reduced sensor model must match the original inertias, geometry, collision
masks, actuator parameters, constraints and solver settings. Native replay tests
compare every state across initial stance and scheduled release; planning must
leave the entire live integration state unchanged. Tests also inject violations
between control frames to ensure the penalty uses all predicted substeps.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/evaluate_g1_cricket_contact_control.py \
  g1_cricket_results/running_momentum_v1 g1_cricket_results/running_preview_v1 \
  --preview --render
```

The output directory must not already exist. Retain both complete episodes,
all candidate scores and physical failures; do not report the chosen planning
row as an independently verified outcome.
