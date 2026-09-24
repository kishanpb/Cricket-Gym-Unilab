# PPO learning against impact v1

The frozen residual-v3 checkpoint was trained against excessive compliance.
It produces zero valid shots after the contact correction. Test learning on
the corrected model rather than changing the shot threshold or selecting clips.

Before starting, complete the full 0.125 ms frozen-policy resolution extension
and require every new row to finish two seconds without guarded contacts,
stability, joint-limit or actuator-limit failures, and with penetration <=6 mm.
Numerical resolution consistency and physical material calibration remain
separate requirements; passing the development preflight does not validate loads.

Train fresh right-hand seed-1 native CPU PPO using the existing impact-v1 owner
with `env.sim_dt=0.00025`. Keep 20 ms control, the frozen Unitree prior, bounded
seven-joint arm residual, unchanged nonnegative separation reward, observations,
three toss lanes, termination rules and all optimizer settings. Use exactly
199,680 transitions (4 environments x 24 steps x 2,080 updates). Do not warm
start, change the reward, or select an intermediate checkpoint. Retain the final
checkpoint, complete scalar series, run configuration and summary.
This matches the earlier control-transition/update budget, not its compute:
0.25 ms physics uses eight times as many substeps as the old 2 ms model.

Evaluate the final policy and zero residual on all 96 existing rows at both
0.25 and 0.125 ms: both hands, three lanes and seeds 4301-4308. The left hand is
untrained transfer, not a separately learned left-handed skill. Compare every
zero-residual row against the corresponding frozen-transfer report to prove
unchanged baseline dynamics. Retain all failures and exact native/serial state
and sensor verification, not just aggregate rewards or successful strikes.

Keep blade-first contact, no forbidden ball/body/handle/wicket or pre-hit ground
contact, first-separation vx strictly >1 m/s, full two-second stability/guard/
joint/actuator checks, and maximum penetration <=6 mm. Reuse the declared
resolution tolerances when comparing new learned-policy outcomes. Passing
development shots is not promotion: any unresolved resolution sensitivity must
remain explicit, and no realistic peak-force or material-calibration claim is
allowed. Held-out deliveries, independent left training, bowling, mjbatch and
the final tournament/showcase requirements remain separate unfinished work.

```sh
uv run python -m unilab.scripts.train_rsl_rl \
  task=g1_cricket_impact_v1/mujoco env.sim_dt=0.00025 \
  training.log_dir=g1_cricket_results/impact_v1/right
```
