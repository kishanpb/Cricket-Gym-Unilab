# Bounded G1 Shoulder Trajectory Search v1

Predeclared development search, not learned bowling or a showcase. The retained
left preload is physically stable but the shoulder-only releases travel too
slowly and downward. This changes only coordinated shoulder reference history.

- Left hand, seed 6301, native mjbatch, 0.0625 ms physics, 20 ms control,
  complete four-second recovery and all original full + signed delivery gates.
- Original guarded owner, motor gains/limits, elbow target 1.4, fixed wrist
  targets, compliant holder and repaired pitch contacts. No state/velocity injection.
- Original preload through tick 98. Smoothstep shoulder pitch/roll/yaw references
  from the preload at 98 to an optimized knot at 106, then a second knot at 114.
  Hold until 150, recover to neutral by 190. Release at the start of tick 114:
  that tick's target cannot accelerate the released ball; tick 113 is the last
  held-ball control interval. Actual stride legality is measured, not assumed.
- First knot bounds: pitch [-2.85,-2.45], roll [.10,.55], yaw [-.45,.45] rad.
  Second: pitch [-.3,1.3], roll [-.05,.55], yaw [-.6,.6] rad. These are motor
  references, not guaranteed achieved positions or collision-free poses.
- SciPy differential evolution, best1bin, seeded 8-member initial population,
  three generations, at most 32 full trials, no polishing, one worker and
  immediate updates. Seed 7351, mutation dithering [.5,1], recombination .7;
  convergence tolerances zero. Initial population retained verbatim. First
  row uses the old shoulder-only reference values with the NEW smooth schedule,
  so it is not claimed to reproduce the old step trajectory.

Every trial independently replays every simulation substep and requires exact
native endpoint state and named sensor equality. All rows, full control traces,
contacts/force diagnostics, signed geometry, failures and normalized objective
terms are retained. The optimizer uses a development score: 10000 for any
physical safety failure, 1000 for any gate failure, 20 per failed gate, plus
seven clipped [0,1] flight/release tie-breaker terms. Raw forward speed,
sideways/downward velocity, actual first bounce, furthest sampled ball position
and target crossing determine those terms. This score is not an RL reward,
running-best release metric or promotion authorization. Full independent pass
status alone qualifies a delivery; a useful optimized witness would still need
BC/PPO and cross-context/hand/timestep evaluation before any learned claim.

Run from the fork with its pinned runtime:

```sh
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/search_g1_cricket_shoulder.py
```

The runner refuses an existing output directory. Canonical artifacts are
`g1_cricket_results/shoulder_search_v1/preflight.json` and `evaluation.json`.
The latter is checkpointed after every complete trial and explicitly marked
running until the optimizer finishes and all input hashes are rechecked.
