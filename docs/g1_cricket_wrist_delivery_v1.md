# Wrist Posture and Release Window v1

The six retuned overhead cases complete four seconds without physical safety
failures, but none qualifies. Earliest releases reach 4.118 m/s forward with
2.876 m/s downward speed. Held-ball traces show more speed later, mostly directed
downward. Test whether wrist pre-cocking flattens flight before adding controller
authority or treating failed deliveries as imitation teachers.

Use the existing shoulder-damping owner (kp 40/kd 2 at selected shoulder pitch;
all other controllers unchanged, including wrist pitch kp 40/kd 10, +/-5 Nm).
Left seed 6301, native Batch, dt .0000625, complete four seconds or retain actual
termination. Parent preload -2.8/.35/0/1.4/0/0/0; drive at 110 to shoulder pitch
1.0, recover 150-190. The wrist-pitch offset alone ramps in with the existing
settle phase 10-30, stays fixed through the drive and recovers 150-190. Configured
wrist targets are not assumed to be achieved; retain measured positions/speeds.

Predeclared 3 by 3 factorial: wrist-pitch offsets [0, -.8, -1.2] rad and release
starts [114, 115, 116]. This tests wrist posture with an explicit release-time
interaction, not a pure single-axis causal comparison across different release
times. Compare wrist levels within each release time. Keep all nine outcomes;
do not select a favorable time after observing a wrist level. The zero-offset
tick114 case must reproduce the previous drive1.0/delay4 outcome, signed audit,
return and every common motion-trace field exactly before the study continues.

Cold FK at shoulder pitch -2.4 suggests wrist pitch -1.2 rotates the shoulder
Jacobian's vertical component from approximately -.295 to -.082 m/rad while
preserving forward leverage (.427 to .421 m/rad). This is only a geometric
hypothesis; added inertia, slow wrist tracking, collisions and altered support
timing may defeat it. No configuration edit, torque-cap increase, grasp change,
ball/root injection or change to the elbow target is included.

Every full and signed delivery gate remains unchanged. Retain all motion traces,
force/contact summaries, failures and independent exact substep replay of native
endpoint state/named sensors. Raw tactile sensor time-series are replay-checked,
not serialized. Positive search-cost movement is not qualification, learning,
hardware certification or permission to publish a learned showcase.

```sh
env PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/probe_g1_cricket_wrist_delivery.py
```

Retain only canonical `g1_cricket_results/wrist_delivery_v1/preflight.json` and
`evaluation.json`. Partial per-case saves remain status running; final status
requires all nine cases and unchanged input hashes. Never overwrite or restart
a still-live run based on a polling timeout.
