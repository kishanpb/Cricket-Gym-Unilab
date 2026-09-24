# Bounded Overarm Motor Search

The preceding from-scratch PPO pilot retained the ball in every evaluation.
This separate diagnostic searches for a physical motor demonstration before
attempting imitation initialization. It is scripted, not learned cricket.

Keep the delivery_v1 model, prior, motor limits, reward, resets and independent
full-episode gate unchanged. Use only the public eight-action motor/release
interface; no pose overwrite, release impulse or ball-velocity assignment.
Both hands start at seed 6301; native mjbatch executes .25 ms physics / 20 ms
control, up to the complete four-second episode or actual termination.

Each hand receives all 16 combinations, in declared Cartesian order:

- Settle through tick 10, smoothstep to the wind-up by tick 45, hold to tick 55.
- Wind-up shoulder-pitch offset -2.45 rad, outward shoulder-roll offset .15 rad
  (negative for right, positive for left); elbow offset .35 or .60 rad.
- At tick 55, shoulder-pitch offset changes to +1.0 or +2.3 rad.
- Release through the existing irreversible channel at tick 57, 58, 59 or 60.
- Keep the swing target until tick 65, smoothstep to zero residual by tick 95.
- Other arm residuals stay zero. Convert desired bounded offsets through the
  existing tanh scale; the unchanged owner still clips original soft targets.

These are offsets around the frozen prior, not guaranteed joint trajectories.
The wrist/elbow geometry was checked kinematically before choosing this family:
the G1 elbow's joint zero is not a geometrically straight arm. Only actual
executed geometry, contact, limits and flight count as evidence.

Retain every one of the 32 attempted full-episode outcomes, including failures,
not selected successful rows. Independent replay must match endpoint state and
all named sensors on every control interval. A pass here is only a candidate
demonstration at one development seed and timestep, not policy qualification;
additional seeds and both timestep/executor checks remain necessary. If every
candidate fails, close this finite family without extending its budget or
weakening the gate. Do not turn a scripted result into a learned-policy video.

```sh
PYTHONPATH=src:scripts uv run --no-sync python scripts/probe_g1_cricket_delivery_motion.py preflight
PYTHONPATH=src:scripts uv run --no-sync python scripts/probe_g1_cricket_delivery_motion.py run
```

The child preflight pins this script, tests, plan and parent preflight/evaluation.
The unchanged parent contract verifies all model/task/runtime inputs before and
after execution. Results belong in `g1_cricket_results/delivery_motion_v1`.
