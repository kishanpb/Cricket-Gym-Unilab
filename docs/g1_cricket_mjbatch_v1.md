# Native mjbatch execution contract

Change only the physics executor for the retained tanh-v1 final checkpoint.
Use owner `g1_cricket_tanh_v1/mjbatch`; keep the task, materialized G1 model,
0.70 kg wrist fixture, ball, observations, raw/tanh actions, prior, rewards,
reset pool, control period and physical gates identical. No new training,
checkpoint selection, policy promotion, force calibration or video claim.

The companion mjbatch fork's held-control recorder copies the fully materialized
models and reproduces ADR-0010 interval semantics through public Batch state and
sensor APIs. Only initial model variants are supported; reset-time model changes
are rejected. State resets and external wrenches retain the existing behavior.
See ADR-0011 for the experimental ownership and optional-dependency boundary.
Any native MuJoCo warning aborts the interval before output can be accepted;
warning counters are reset at each interval. Divergence is fail-fast, not a
claim to reproduce official Rollout's frozen-divergence trajectories.

Before the complete evaluation, require unit parity through actual contact with
held actuator targets, external wrenches, heterogeneous fixed models and reset
auxiliary/warmstart state. Exercise the native G1 environment for both hands,
both timesteps, lanes 0/-0.12 and zero/predeclared sinusoidal raw actions (including
magnitudes above 1), up to two seconds or its actual termination. Compare every
substep state and sensor, float32 caches, 480-input prior history, 115-input policy
observations, all 29 targets, rewards, termination and separation latches. Add
two-row partial-reset and unsupported reset-model-randomization tests.

Freeze current sources, executor Python/native hashes, companion source revision,
the parent full report, checkpoint and training metadata before evaluation.
Then execute ALL 96 identities at EACH of 0.25 and 0.125 ms: both hands,
zero residual/PPO, three lanes and seeds 4301-4308. Run independent serial replay
at every interval and require every resulting row, including returns, to match
the immutable parent report exactly. Preserve all failures; forward-shot vx >1 m/s
and penetration <=6 mm stay unchanged. Left PPO is still untrained transfer.

Stop on any source, runtime, parity or complete-pool mismatch. Even exact native
execution is only an integration result: it cannot turn the parent's backward
exits into valid shots or establish material realism. Native-mjbatch training,
independent left-hand learning, bowling and validated videos remain separate work.
