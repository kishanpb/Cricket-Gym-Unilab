# G1 Fixed Overarm Drive and Release Study

This is a scripted development experiment, not learned bowling or a showcase.
The retained left-hand preload (-2.80, 1.40) from `overarm_guard_v1` is the
only parent. It has a near-straight physical elbow (about 3.10 rad), with sampled
overhead readiness well before its original legal stride window. Read-only
replay finds back-foot landing at 1.706749 s and front-foot landing at 2.258311 s;
the next back-foot landing at 2.444061 s ends that baseline window. New drives
alter the gait, so these times are design context, never assumed new legality.
All six trials use seed 6301, native mjbatch, 0.0625 ms physics and the unchanged
four-second full delivery gate. No held-out claim is made.

Through tick 109 each trial exactly repeats its parent's target trajectory.
At tick 110 (2.2 seconds), the shoulder target changes to +0.3 or +1 radian while the
other six arm targets remain at their preloads. The original bounded PD motors
produce the motion: this is a reference step, not a state or velocity impulse.
Release occurs at tick 114, 118 or 122 (2.28, 2.36 or 2.44 seconds), by disabling
the existing compliant wrist fixture, never by assigning ball velocity. Hold
the follow-through through tick 150 and smoothly recover SDK neutral by 190.
Only this motor-reference/release schedule changes; the 90% prior-target guard,
robot dynamics, collision geometry, force limits, pitch, reward and gate remain.

Every candidate is retained, including early terminations. An independent serial
MuJoCo replay checks every physics substep for contacts, joint/actuator limits,
stability, elbow extension, delivery-stride legality and ball flight. Native
endpoint state and every named sensor must match exactly. Control-boundary
actual arm positions/velocities, targets, last-substep torques and ball motion
are retained alongside first/worst violation attribution and actual pre-release
foot landings. These torques are motor outputs, not validated holder torques;
holder/contact forces remain simulated and uncalibrated.
The first contact excluding foot/pitch diagnostic includes legitimate released-ball/pitch
contact; only the full outcome gate classifies forbidden contacts. Diagnostic
phase names follow this drive/recovery schedule, not the earlier reach schedule.

The existing anatomical-angle proxy is unsigned and folds near straight (about
elbow joint q=1.38 rad in offline FK). This study keeps the elbow reference fixed
and retains actual joint positions/velocities as well as the proxy angle; it
does not certify cricket elbow legality. Any training that varies the elbow
needs a signed extension audit rather than relying on the unsigned proxy alone.

This diagnostic deliberately preserves the four-second horizon. Late release
leaves little flight time; a target miss is not evidence of an absolute robot
speed limit. A successful single development row would still need additional
seeds and paired-timestep/executor validation before use as a training teacher.
No failed gate may be waived, no selected row may be called learned, and no
training is authorized by this script's result alone.

Run from the integration root with its pinned runtime:

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 python scripts/probe_g1_cricket_overarm_release.py
```

The script refuses to overwrite existing evidence. Inputs (including current
sources, runtime, resolved config, executor and frozen parent artifact hashes)
are checked before and after all six rows. Canonical outputs are
`g1_cricket_results/overarm_release_v1/preflight.json` and `evaluation.json`.
