# Frozen-Policy Lane Feedback

The previous closed-loop PPO approach completes but drifts sideways. This
bounded comparison changes only the external locomotion policy's lateral speed
request: `clip(-lane_error / 1 second, -0.25, 0.25)` m/s. Lane error is world y
relative to the declared reset lane, sent as the prior's body-lateral request;
there is no additional yaw controller or world/body heading compensation.
The original forward command, prior history, both final local PPO actors,
geometry, gains, friction, timesteps and physical gates remain unchanged.

Evaluate all four hand x 62.5/31.25-us combinations from rest at seed 1, compared
with the immutable parent PPO outcomes. Verify identical compiled physics
models and source/checkpoint hashes; independently replay every native substep.
Keep complete traces and both finest-resolution half-speed videos regardless
of outcome. No training or checkpoint selection occurs. The command is an
explicit feedback controller, not a newly learned locomotion policy.

The target is lateral drift below the original 15 cm limit without losing
complete upright approach, legal-side foot coverage, settling or contact gates.
Loaded-foot slip below 0.2 m/s and integrated stance slip below 3 cm remain
required, not silently removed if lane control succeeds. This is a walking
approach experiment; gather, legal overarm release and recovery remain unfinished.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python \
  scripts/evaluate_g1_cricket_approach_lane.py \
  g1_cricket_results/approach_lane_v1
```
