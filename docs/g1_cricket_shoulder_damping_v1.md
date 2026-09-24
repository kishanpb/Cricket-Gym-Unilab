# Selected Shoulder Damping Comparison v1

This locally retunes a controller, not the robot's motor strength. The active
SDK-derived cricket controller uses shoulder kp 40/kd 10, not the raw source
asset's kp 14.251/kd .907. The positive-arc baseline's release speed stays below
1.01 m/s, with at least .5635 rad remaining before the positive stop.

Versioned owner `g1_cricket_shoulder_damping_v1/{mujoco,mjbatch}` changes only
the throwing shoulder-pitch actuator's velocity feedback coefficient from
10 to 2. Keep kp 40, +/-25 N m, all other 28 actuator controllers, physical
joint damping/armature/ranges, inertias, contacts, holder, observations, actions,
reward, initialization and full/signed cricket gates unchanged. This is not an
upstream-certified gain setting or a claim that closed-loop dynamics are unchanged.
The frozen external policy's weights and deploy contract remain intact, but it
is now operating outside its original whole-body closed-loop calibration.

Replay all six positive-arc launch/brake schedules, same left seed 6301,
.0625 ms physics, native Batch, fixed tick114 release and full4s recovery. Do not
select only the favorable parent. Retain failures, joint stop margins, torques,
contacts, force telemetry and exact independent substep replay. No training,
checkpoint or showcase is authorized by a better scalar search score.

The new action config has a distinct identity covered by the existing strict
`env.actions` checkpoint contract, and validates its compiled gain/cap at build.
Strict original-to-new and new-to-original checkpoint checks must reject;
same-controller executor transfer remains permitted. Do not bypass strict mode
or treat an old sidecar without a contract as validated under the new controller.

```sh
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_shoulder_damping.py
```

Canonical preflight/evaluation live in `g1_cricket_results/shoulder_damping_v1/`.
Original positive-arc evidence remains frozen at source `b8918a72`; this
comparison changes controller damping explicitly and does not rewrite parents.
