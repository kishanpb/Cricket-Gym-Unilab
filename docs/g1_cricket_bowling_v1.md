# G1 Bowling Task Foundation

This is an **untrained arm-and-release task**, not a learned bowling result or a
legal cricket delivery. Both hands use the same owner contract; `mjbatch` changes
only the CPU executor. The existing batting experiments and media are unchanged.

## Control and Reset

Owners: `task=g1_cricket_bowling_v1/mujoco` and
`task=g1_cricket_bowling_v1/mjbatch`, registered as `G1CricketBowling`.
The pinned external Unitree locomotion prior receives a fixed 0.4 m/s forward
command. The future cricket policy controls seven selected-arm offsets plus one
release channel. Offsets use tanh bounds of `[2.5, .8, 1.2, 1, .4, .4, .4]` rad,
then clip the requested arm targets to the original soft joint limits. This is
not retrained whole-body locomotion; target clipping does not prove actual joint
motion stays within limits.

Release occurs once when the raw eighth channel is strictly greater than .5.
It disables the declared compliant holder without writing any pose or velocity.
The integrated FULLPHYSICS boundary state, ball translation/velocity and elbow
angle are retained before release. Solved-frame sensors are not substituted for
that boundary state. Only reset can reattach the ball, through the public
equality-activation capability documented in ADR-0011.

Each reset jitters the original 29 joint positions by +/- .005 rad, recomputes
the ball pose from that exact wrist pose using cached-model forward kinematics,
and commits robot and ball states in one reset transaction. Partial resets leave
other rows and release latches unchanged. The original free ball, robot masses,
gains and actuator limits remain; a massless fixed holder sensor body is added.
The root starts at x=-1 m, y=0; target wickets move to x=18.6 m. The 4-second
episode uses 20 ms control and .25 ms physics (also checked at .125 ms).

## Observations and Telemetry

The task exposes 122 values: robot gravity/position/velocities, joint offsets and
velocities, ball position relative to the wrist and ball velocity, wrist height,
last eight policy actions, the frozen prior's 29 outputs, release flag, episode
time, interval peak holder force, endpoint local holder force, selected-hand
touch occupancy and handedness. This includes privileged simulator state, not
an onboard sensing claim. The separate 480-value prior input stays unchanged.

The massless holder force sensor reports fixture-transmitted force on the ball,
in the wrist site frame. Tests transform it into world coordinates and compare
it to the ball's translational generalized constraint force during an isolated,
contact-free moving-wrist trial. Every solve is sampled. `peak_load` is the
interval maximum force norm in N; `impulse_world` is the signed world force
integral in N s over that interval. Neither is cumulative or total ball-impact
load; released ball/ground impacts do not count as holder load.

`touch_fraction` is the fraction of solves with a geometric ball/selected-hand
contact, using four contact slots. It is not finger tactile hardware and cannot
exclude contact with the forearm, torso or opposite hand. Attachment itself is
never counted as touch. Nonfinite samples or contact-capacity overflow fail.

**No holder torque is exposed.** In the installed MuJoCo 3.11.0, a weld-site
torque signal failed comparison to the ball's physical generalized rotational
constraint torque. The [versioned engine implementation](https://github.com/google-deepmind/mujoco/blob/3.11.0/src/engine/engine_core_smooth.c#L2288-L2295)
copies rotational weld multipliers directly into spatial torque. The task omits
this unvalidated channel rather than labeling it N m. This finding concerns the
new equality holder, not the earlier rigid bat-fixture measurements.

## Complete Mechanics Smoke

[Retained results](../g1_cricket_results/bowling_v1/mechanics_smoke.json): both hands,
both executors, .25/.125 ms physics, four jittered rows at seed 5301. Two rows
hold throughout and two release at 1 s, always with zero arm residuals: 32 rows,
not 32 independent learned trials. All complete four seconds; all 16 paired
executor outcome/telemetry dictionaries match exactly. Sixteen rows release.
Minimum sampled pelvis height is .779097 m, minimum sampled up component is
.999129, maximum holder force is 4.208319 N, and geometric hand-touch occupancy
is zero. This checks carrying and timed dropping, not throwing or bowling.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-sync python \
  scripts/audit_g1_cricket_bowling.py --output /path/to/new/mechanics_smoke.json
```

The script refuses an existing output, retains every row and source hashes, and
raises on nonfinite state or executor disagreement. Unit tests additionally
compare complete short substep trajectories/sensors, release continuity,
gravity-only flight, force sign/frame/integration, reset alignment, latch
thresholds and absence of step-time pose writes. No broad performance or
contact-convergence claim follows from these checks.

Validation at this revision: 176 focused UniLab tests pass (13 slow tests
deselected, two existing RSL-RL observation-group warnings), including 13 new
bowling/report tests, plus 47 native Batch/held-control tests. Ruff and diff
checks pass. The 32-row smoke was rerun with identical rows and refreshed source
hashes. This is not full repository CI or an upstream PR validation run.

## Before Learning Can Qualify

The provisional reward shapes wrist height, approach speed and a one-shot
forward/high release; upright, failure and action-rate terms remain explicit.
It is **not the success gate**. Checking ball release x<0 does not check the
front foot. A throw, sidearm fling, late foot fault or subsequent body collision
can earn positive reward. A stable non-release can accumulate shaping reward.

Before a pilot's result is called bowling, freeze an independent full-episode
gate for one policy-owned release, pre-release delivery-arm motion (not just
one elbow angle), front-foot/crease geometry, all ball/body contacts, solver and
joint/actuator behavior, flight/first-bounce/target crossing, stability, and
paired physics resolution/executor checks. Include synthetic negative tests
for sidearm/low release, foot fault despite a ball behind the line, collision or
fall after release, and no release. Record every evaluation context and failed
gate. No bowling checkpoint, new bowling video, hardware safety claim or
upstream submission is produced by this foundation.
