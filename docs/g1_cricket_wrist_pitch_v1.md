# Forward-phase wrist-pitch probe

This is a bounded control diagnostic, not learned batting or a showcase.
The retained parent `pmppppp_s10` has positive first-exit vx but excessive
fine-timestep overlap. Changing only its last command did not resolve that
tradeoff. Its first loaded contact normal has z approximately 0.272; vertical
relative motion contributes approximately -1.07 m/s to -4.67 m/s normal closing
speed. A more forward-facing blade could reduce that contribution, but moving
the wrist also changes contact position/velocity: success is not assumed.

Cold reset kinematics identify right wrist pitch as the strongest of the three
wrist axes for blade vertical tilt (normal z derivative approximately -0.674
per radian, versus +0.169 roll and -0.149 yaw). This is only a local reset
Jacobian, not evidence about achieved impact orientation or learned control.

## Frozen experiment

Keep the robot, ball, bat mount, frozen prior, reset, reward, timing, action
limits and all shot gates unchanged. In the parent's forward phase, ticks
10 through 19 inclusive, multiply **only wrist-pitch channel 5's normalized
residual** by one fixed scale. The seven scales, in execution order, are
`[1, 0.75, 0.5, 0, -0.5, -0.75, -1]`. The original normalized command is -0.95;
therefore all resulting pitch residuals stay within +/-0.1425 radian using the
existing 0.15-radian limit and inverse tanh. All other six channels and all
other 90 control ticks exactly preserve the parent command. Never write robot
poses or physics state except the existing reset/replay machinery.

Every scale receives a complete 100-tick / 2-second trial at both 0.25 ms and
0.125 ms physics steps in both MuJoCo rollout and native mjbatch: **28 rows**.
Right hand, seed 4301, center toss only. Scale 1 must exactly reproduce the
retained parent in every context before alternatives run. No prefix screening,
adaptive grid refinement, stopping after a passing row or unreported trials.
Native early termination remains an explicit failure.

Use the existing complete shot gate and timestep comparator unchanged, including
first blade-only contact, no forbidden/guard contacts, stability, joint/actuator
limits, first outgoing vx strictly >1 m/s, penetration <=6 mm over every impact,
and native completion. A scripted witness requires all four contexts to pass,
exact executor outcomes/first-impact evidence, and numerical timestep agreement.
Retain first-loaded-contact normals and signed per-axis contributions to normal
relative velocity; these explain the mechanism but cannot replace the gates.
Contact normals can differ from the blade face normal at edges and corners.
The commanded residual is not the achieved wrist angle or an isolated change
in contact orientation: the prior and coupled robot motion still respond.

Preflight pins parent evidence, sources, tests, this contract, package versions
and installed native executor hashes. No training or promotion is authorized by
passing this scripted diagnostic. A positive result is only a control witness
for a separately specified learned-policy experiment; a negative result closes
this finite wrist-pitch family, not all humanoid batting. Forces are simulated,
uncalibrated loads; neither generalization nor hardware readiness is established.

## Reproduce

Use the existing G1 runtime and externally cached, hash-checked Unitree prior:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_wrist_pitch.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_wrist_pitch.py
```

Canonical outputs are `g1_cricket_results/wrist_pitch_v1/preflight.json` and
`evaluation.json`. Existing checkpoints, reports and videos remain unchanged.
