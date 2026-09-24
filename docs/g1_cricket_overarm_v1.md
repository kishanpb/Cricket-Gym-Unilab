# Absolute Arm Reference: Overarm Reach

The previous moving-prior residual search never raised either upper arm through
horizontal. This version changes only ownership of the selected seven arm motor
targets: bounded absolute targets replace offsets around the moving prior. The
other 22 joints keep the same frozen prior. Eight action channels, release
semantics, 122 observations, reward, holder, pitch_contact_v2, robot dynamics,
PD gains, joint/torque limits and the independent full delivery gate are unchanged.
Zero arm action means the SDK neutral pose, not the former moving-prior pose.
Historical checkpoints therefore are not compatible evidence for this owner.

Each raw arm action is tanh bounded, then mapped from SDK neutral to the original
soft limit in its sign direction. This neither increases motor torque nor writes
robot or ball state during a step. The eighth channel still irreversibly disables
the holder when strictly above .5; no launch velocity is supplied.

## Frozen Reach Diagnostic

Before a drive/release search, test four absolute poses for each hand: shoulder
pitch -2.45 or -2.80 rad, elbow 1.10 or 1.40 rad, outward roll .35 rad,
shoulder yaw and wrists zero. Cold forward kinematics checks the proposed arm
geometry and ball/body clearance; those poses are not dynamic demonstrations.
At seed 6301, command SDK neutral until .2 s, settle elbow/roll/wrists by .6 s,
raise the shoulder with a smoothstep until 1.6 s, hold until 2.4 s, recover to
SDK neutral by 3.4 s, and complete the full four-second episode. Never release.
Use native mjbatch at .0625 ms with independent serial substep contact/limit
inspection and exact endpoint state/sensor replay. Retain all eight outcomes
and control-boundary achieved-pose traces, including failures and termination.

A reach witness requires four consecutive 20 ms endpoints in the hold window
with upper-arm vertical fraction >=.75, ball >= shoulder+.25 m, and elbow target
error <=.1 rad, with no robot/ball forbidden contacts, joint/actuator limit
exceedance or stability failure over the entire trial. This is sampled preload
readiness, not substep continuous hold proof, a qualified delivery or learning.
No-release and flight-gate failures remain in the independent delivery report.
Do not use a failed reach as a teacher or expand this family after seeing results.

Only a measured reach witness warrants a separately frozen drive/release study,
with actual stride timing and paired-resolution/executor full delivery checks.
If the reference change fails, inspect achieved tracking/contact failure first;
do not extend the failed carry-only PPO or increase physical motor limits.

```sh
PYTHONPATH=src:scripts uv run --no-sync python scripts/probe_g1_cricket_overarm.py
PYTHONPATH=src:scripts uv run --no-sync python -m pytest tests/scripts/test_g1_cricket_overarm.py
```

Retain only canonical preflight/evaluation JSON under
`g1_cricket_results/overarm_v1`. Source and runtime hashes are checked before
and after execution. Simulated holder/contact forces are uncalibrated; this
experiment does not produce a trained policy or advertising video.
