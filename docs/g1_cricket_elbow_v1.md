# Forward-phase elbow control probe

The wrist-pitch sweep barely changed impact orientation, and a separate audit
confirmed full forward-phase negative saturation for scales 1 and 0 at the
unchanged +/-5 N m wrist limit. This experiment tests a stronger proximal joint
without increasing hardware authority or changing the physical contact model.
It is a scripted diagnostic, not learned cricket, calibration or a showcase.

## Frozen contract

Use parent `pmppppp_s10`, right hand, seed 4301, center toss. Change **only
residual channel 3, right elbow**, during control ticks 10-19 inclusive. Multiply
the parent's normalized -0.95 command by each of `[1, 0.75, 0.5, 0]`, in that
order. Inverse tanh feeds the existing smooth bounded action owner; the elbow
residual changes from -0.3325 toward 0 rad within its unchanged 0.35-rad limit.
All other six channels and all other 90 control ticks stay exactly unchanged.
Retain the original local-y elbow joint, gains, +/-25 N m motor limit, bat/ball
objects, prior, reset, reward and all physical/shot gates.

Each scale runs a full 100-control-tick / 2-second trial at both 0.25 ms and
0.125 ms physics steps in both MuJoCo rollout and native mjbatch: **16 rows**.
Scale 1 must exactly reproduce each retained full parent before alternatives
run. No prefix selection, adaptive refinements, successful-row stopping or
changing the toss. A native early termination remains a retained failed trial.

Keep the full shot gate: blade-first-only contact, no forbidden/guard contacts,
stability, joint and actuator limits, first separation vx strictly >1 m/s,
maximum penetration <=6 mm across all contacts, and native two-second
completion. Keep the existing numerical timestep comparator. A scripted witness
requires all four contexts to pass and exact executor outcome/impact evidence.

Every physics solve also records elbow requested and applied torque, achieved
joint position and velocity. Use matched solve-phase actuator length/velocity
caches, unit gear, fixed gain, affine bias and no activation dynamics. Verify
applied force equals the torque request clipped to the actual unchanged limits;
require <=1e-10 N m reconstruction error and exact independent replay endpoints.
Retain per-control-tick min/max, saturation counts and targets across the entire
trial. Motor traces must be complete and exactly equal across executors before
a witness is accepted. Retain first-loaded contact direction and signed normal
closing-speed contributions. Neither target changes nor reset kinematics prove
achieved orientation; edge/corner contact normals need not be blade-face normals.

Preflight freezes inherited evidence, sources, this contract, tests, versions
and installed executor hashes. All positive and negative rows remain retained.
No pose writes except reset/replay, motor/gain changes, new policy training or
promotion. A positive witness informs a separately specified observation-policy
learning experiment; it is not itself learned batting or generalization.

## Reproduce

Use the existing runtime and pinned external Unitree prior:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_elbow.py --preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  uv run --no-project --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_elbow.py
```

Retain `g1_cricket_results/elbow_v1/preflight.json` and `evaluation.json`.
Existing checkpoints, reports and videos remain unchanged. All reported loads
are simulated and uncalibrated; both-hand learning and bowling remain separate
unfinished requirements of the full goal.
