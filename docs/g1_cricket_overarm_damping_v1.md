# Paired Overarm Damping Study v1

The positive-arc controller comparison increased shoulder speed but produced
zero full signed-gate passes. In the retained 102/114 lower-damping trajectory,
maximum sampled held-ball speed is 4.999 m/s and maximum forward component is
1.893 m/s. Moving release among those control endpoints cannot meet the
unchanged forward-speed gate. This is a diagnostic of that trajectory only,
not a continuous-time bound or global impossibility result.

Return to all six original overhead drive/release schedules, not a selected
successful row. Change only the selected shoulder-pitch controller kd from 10
to 2 using the existing, checkpoint-distinct shoulder-damping owner. Keep kp
40, +/-25 Nm, remaining controllers, physical joint limits, guarded frozen
locomotion prior, holder, ball/pitch model, observations and reward unchanged.
No grip, body, joint or ball state/velocity injection after reset.

Exact paired schedule: left hand, seed 6301, physics dt .0000625, native Batch.
Preload shoulder pitch -2.8, roll .35, yaw 0, elbow 1.4, wrists 0. At tick 110,
drive pitch to .3 or 1.0. Release at tick 114, 118 or 122 (start of control
tick); recovery begins 150 and finishes 190, full episode 200 ticks/four seconds.
Reuse the original target generator without modifying its targets or timing.
Earlier slow preload safety does not establish safety under retuned dynamics.

The hypothesis is that reduced opposing velocity feedback preserves more
forward speed in the overhead-to-forward arc, where cold FK shows greater
forward leverage than the positive-arc start. FK is not a dynamics result.
Potential failures include preload overshoot, self/ball contact, leg-limit or
balance loss, non-overarm release, illegal stride and short/downward flight.
Do not adapt cases after seeing outcomes or bootstrap failed throws as teachers.

Retain all six full or terminated motion traces, contact/holder force summaries,
hard-limit attribution and the additive signed-elbow audit. Every substep
independently replay-checks native endpoint state and named force/tactile sensors;
raw sensor time-series are not serialized in these reports. All full and signed gates stay
unchanged: no relaxed speed, flight, contact, elbow, balance or stride criteria.
The full delivery benchmark is an engineering test, not umpiring certification.
No policy checkpoint, learned video or promotion claim follows from this study.

```sh
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_overarm_damping.py
```

Canonical outputs are `g1_cricket_results/overarm_damping_v1/preflight.json`
and `evaluation.json`; per-case saves retain partial progress without treating
it as complete. Inspect a running process before considering recovery, and
never overwrite retained results. The original six cases and signed audit
remain unchanged in `g1_cricket_results/overarm_release_v1/`.
