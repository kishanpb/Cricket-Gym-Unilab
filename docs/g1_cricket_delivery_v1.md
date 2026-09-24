# G1 Delivery Learning Pilot

This fixed-budget experiment trains separate right/left cricket actors with PPO
on native CPU mjbatch, above the unchanged frozen Unitree locomotion prior.
The scene and success gate are versioned separately from the untrained bowling
mechanics smoke. No old result is relabeled or overwritten.

## Frozen Experiment

- Owner: `g1_cricket_delivery_v1/mjbatch`, 122 observations, eight actions,
  unchanged provisional bowling reward, 64x64 actor/critic and initial .2 noise.
- Each hand: fresh actor/critic/optimizer, seed 1, four environments, 24 steps
  per update, exactly 256 PPO updates / 24,576 actual wrapper transitions.
- Physics .25 ms, control 20 ms, full four-second episodes. No random initial
  episode clock, imitation, reference arm trajectory or direct ball velocity.
- The frozen prior still receives .4 m/s; this is not newly learned locomotion
  or a high-speed run-up. The fixture is not an articulated grasp.
- Stop on nonfinite actions/state/loss, solver warning or contract mismatch.
  Retain final checkpoint and all native scalar diagnostics; no best-seed or
  intermediate-checkpoint selection. Refuse existing run directories.
- Evaluate zero arm/hold baseline and the final separately trained PPO actor
  on both hands, seeds 6301/6302, both .25/.125 ms timesteps and both executors:
  all 32 rows. These are development contexts, not broad generalization evidence.

## Scene and Qualification

Both wicket sets are 20.12 m apart, bowler wicket x=-1.22 m, bowler popping back
edge x=0, striker wicket x=18.90 m and popping edge x=17.68 m. Return inside edges
are y=+/-1.32 m. Paint is noncolliding. The robot begins at x=-1 m and y=+.5 m
(right hand) or -.5 m (left hand), declaring over-the-wicket delivery on that
side. Bowler wicket contacts now have explicit task guards.
Dimensions/edge conventions follow [MCC Law 7](https://www.lords.org/mcc/the-laws/the-creases)
and [Law 6](https://www.lords.org/mcc/the-laws/the-pitch); simplified fixed stumps
without bails and an abstract holder are not a complete regulation ground.

The independent serial replay must reproduce native endpoint state and all
named sensors exactly at every control interval. It inspects every physics
solve and the integrated release boundary, never forwarding the replay solver
just to refresh poses. Qualification requires all of:

- One policy-owned release, a full episode, pelvis height >= .48 m and up >= .65,
  no forbidden robot/ball contact, original joint/actuator limits (1e-6
  numerical tolerance), finite state and no solver warnings.
- Upward upper-arm shoulder-height crossing, then at most 15 degrees geometric elbow
  straightening through release, including extend-then-reflex excursions.
  Shoulder/elbow/wrist-roll origins define the arm; distal wrist rotation
  cannot fake elbow motion. Release arm must remain above horizontal and the
  ball >= .12 m above the shoulder. The extension threshold follows the
  [ICC tolerance](https://www.icc-cricket.com/about/cricket/rules-and-regulations/illegal-bowling-actions),
  not an assertion that raw [MCC Law 21](https://www.lords.org/mcc/the-laws/no-ball)
  contains that tolerance. The height rule is an engineering overarm criterion.
- Conventional front foot (opposite the bowling hand) must have a loaded
  landing after the back foot, both within the preceding second, front landing
  no more than .5 s before release. Initial settling contacts before .2 s do not
  count as a delivery stride. A landing requires the complete foot to clear the
  pitch by more than 2 mm for at least 20 ms before loading above 1 N; planted
  force fluctuations do not count. The latest back landing must precede the
  latest front landing. At landing, some of the
  front foot is behind the popping back edge, the complete back foot is inside
  the return edges, and both footprints remain on the declared side. The last
  requirement is deliberately more conservative than the human rule. All 11
  collision geoms per foot are measured, not the ankle or ball position.
- Release vx > 6 m/s, |vy| <= 2 m/s; exactly one bounce before the target plane,
  first bounce 4 < x < 17.68 m and |y| < 1.32 m; target crossing |y| <= .5 m
  and .04 < z <= 1.2 m. Ball penetration may not exceed 6 mm. These are delivery-quality
  criteria, not a claim to implement every cricket law or batter interaction.

No-release, sidearm, foot fault despite a ball behind the line, post-release
collision/fall, missing stride, throw-like elbow extension and bad flight all
fail even with positive reward. Raw all-geometry contacts supplement the task's
selected-hand touch sensor. Force reporting remains simulated holder force and
impulse, not finger tactile hardware, total impact force or validated weld torque.

For each hand/controller/seed, both executor outcomes must match exactly.
Require identical failure sets across timesteps, holder force difference within
max(5%, 1 N), penetration difference within max(5%, .5 mm), and release/bounce/target
positions and release velocities within .1 m/.1 m/s. Only all four passing cases
qualify a context. The full delivery gate is a conservative robotics diagnostic,
not ICC certification, hardware safety, or a public-showcase authorization.

## Commands

Use the documented external prior and native CPU runtime, with two Torch threads:

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-sync python scripts/train_g1_cricket_delivery.py preflight
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-sync python scripts/train_g1_cricket_delivery.py train --hand right
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-sync python scripts/train_g1_cricket_delivery.py train --hand left
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-sync python scripts/train_g1_cricket_delivery.py evaluate
```

The preflight retains resolved configuration, source hashes, external prior
hashes, package versions and native executor identity. Checkpoints carry hand
and preflight identity. Qualification and videos remain unproven until the
complete resulting evidence is inspected; training return alone is insufficient.
