# Isolated contact-compliance sensitivity

This experiment diagnoses the contact model before further G1 training. It is
not a robot-policy trial, a cricket-material calibration or a showcase. The
current robot task and its 6 mm penetration gate remain unchanged. The prior
wrist and elbow experiments remain failed controllability diagnostics.

## Frozen Plan

Use a fixed blade box with half-sizes (0.025, 0.055, 0.20) m and a free 0.156 kg,
36 mm sphere initially at x=0.1 m. Disable gravity. Test normal incoming speeds
2.5 and 4.67 m/s: the former matches the historical isolated probe, the latter
approximates the retained scripted strike's relative normal closing speed. The
fixed blade removes robot motion and fixture compliance; this is not an exact
replica of that oblique moving-blade strike.

Change only the positive-format contact time constant: **4 ms control, 2 ms
candidate**, both damping ratio 1. Use the current task's explicit pair:
condim 3, solimp (0.9, 0.95, 0.001, 0.5, 2), friction
(0.6, 0.6, 0.01, 0.001, 0.001), margin and gap zero. Use implicitfast with
default Newton solver and retain the compiled parameters. Fixed damping ratio
does not mean fixed damping coefficient: changing the time constant changes
both stiffness and damping, and does not independently set restitution.
See the [MuJoCo solver documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).

Predeclare four physics steps: **0.25, 0.125, 0.0625 and 0.03125 ms**. Every
combination runs one full second: **16 rows**, without adaptive refinement or
successful-row selection. No robot policy, motor limit, reward or other contact
pair changes. The historical probe remains immutable and is reproduced in tests;
its default-pair response is not assumed identical to this explicit-pair setup.

Retain every contacting solve's signed world force on the ball, pre/post
velocity, distance and solve-start time. Contact caches after mj_step describe
the solve's pre-integration geometry; velocity_after is post-integration.
Occupancy and positive-load duration count these solve intervals separately.
Retain first separation, rebound ratio, penetration, peak normal force, signed
impulse and force-norm integral (different quantities), initial/final translational
and final rotational kinetic energy, and translational contact work. Net energy removed includes physical-model dissipation
and discretization effects; it is not separately measured material loss.

Require one completed and loaded contact episode, finite states, no solver
warnings, no net energy gain and nonnegative normal loads. Record numerical
sideways/rotation drift: maximum transverse speed <=1e-6 m/s and angular speed
<=1e-4 rad/s. Preliminary collector tests at 4 ms/2.5 m/s/0.25 ms showed
component magnitudes 8.92e-8 m/s and 5.96e-6 rad/s (vector norms 1.26e-7 m/s
and 8.43e-6 rad/s), so exact symmetry is not assumed; these tests
precede the frozen matrix and do not select a contact candidate. Check every step and total
impulse against momentum change at 1e-10 N s, and midpoint-velocity contact work
against translational kinetic-energy change at 1e-10 J. Include the sphere's
rotational energy in the no-energy-gain check. Those identities check force accounting,
not material realism. Tests also reverse incident direction to check force sign.

Compare **all three adjacent timestep pairs** at each speed and time constant.
Use max(5% of finer result, absolute tolerance): 0.1 mm penetration, 1 N peak
force, 0.125 ms occupancy/loaded duration, 0.01 m/s first-exit velocity,
0.001 N s force-norm integral and 0.005 J final energy. Missing exits fail.
Retain failures; these finite-grid checks are not asymptotic convergence.
Candidate peak-load increases must be reported alongside depth reductions.
No row can promote a policy or validate a humanoid video. A later task revision
and frozen-controller transfer require a separate contract and full robot gates.

## Reproduce

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_compliance.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_compliance.py
```

The preflight pins source, tests, this plan, model owners and parent evidence.
The report retains every row and contacting solve in
`g1_cricket_results/compliance_v1/evaluation.json`; the runner refuses to
overwrite retained evidence. Use a clean checkout without that output directory
for reproduction. Both-hand learned batting and bowling remain the final goal.
