# G1 One-Bounce Incoming Delivery

This replaces the lofted soft-toss input with a single declared bounced practice
delivery. It does not change the frozen dry-swing PPO actors, give them ball
observations, or establish learned interception. The robot still uses two
mechanical hand grips and whole-body motor control, not live pose writes.

## Protocol

Reset the ball at `(4, 0, 1.3)` m with velocity `(-3, 0, 4)` m/s and zero spin.
After reset, native gravity/contact dynamics determine its entire trajectory;
there is no bounce impulse assignment or mid-episode ball repositioning. Keep
the existing pitch pair (`solref = .002 .3`), blade response, original robot
geometry, motor limits, ankle gain 4, waist gain 1 and root-position gain 4.
Use the existing final `model_511.pt` actors from
`bimanual_balanced_small_residual_v1`, with zero residual as the control.

The delivery was designed from a ball-only flight check, not from selecting
successful robot episodes. With the unchanged ball/pitch geometry, contact
pair and 62.5 microsecond timestep, four vertical launch velocities gave:

| Initial vz (m/s) | First pitch contact (s) | Bounce x (m) | Ball z at 1.4 s (m) |
| ---: | ---: | ---: | ---: |
| 3.0 | 0.898500 | 1.304503 | 0.109523 |
| 3.5 | 0.977312 | 1.068066 | 0.320762 |
| 4.0 | 1.058937 | 0.823191 | 0.461676 |
| 4.5 | 1.142937 | 0.571191 | 0.466661 |

The 4.0 m/s option supplies a bounce farther in front than 4.5 while reaching
the lower blade region around the existing swing time. This is a low-speed
practice feed, not a regulation-speed bowling result. The ball-only check
cannot prove G1 stability, contact or policy quality.

The subsequent finer 31.25 microsecond ball-only check retains the same four
launches, without changing the selected feed:

| Initial vz (m/s) | First pitch contact (s) | Bounce x (m) | Ball z at 1.4 s (m) |
| ---: | ---: | ---: | ---: |
| 3.0 | 0.898469 | 1.304595 | 0.105323 |
| 3.5 | 0.977281 | 1.068157 | 0.315721 |
| 4.0 | 1.058875 | 0.823376 | 0.425088 |
| 4.5 | 1.142938 | 0.571188 | 0.487403 |

The selected feed's rebound height differs by 0.03659 m. The ball-only fixture
test locks both observed values, not a convergence claim. Full robot evaluation
must retain and compare both resolutions; the original contact tolerances are
not relaxed to accommodate this discrepancy.

Run both hands, reference-only and PPO, at 62.5 and 31.25 microseconds: eight
complete or failed episodes, without tuning the launch after seeing their
results. Retain every row and complete fine-resolution videos. Use the verified
compact native recorder; preserve independent native endpoint/sensor replay
and all existing nine physical/tracking and three contact checks, including
the currently failing 0.08 m bat-path bound and 6 mm ball-penetration limit.

Additionally require exactly one loaded pitch interval to have ended before
the first loaded blade contact, with downward incoming and upward outgoing
pitch velocity. Its location must be in front of the crease and on the pitch;
the ball must still approach the bat in negative x. Full tosses, two bounces,
an unfinished contact, simultaneous pitch/blade contact and a post-hit bounce
cannot satisfy this gate. Record every pitch interval and the first blade
contact at physical-substep resolution. Contact loads remain simulated and
uncalibrated, not hardware tactile measurements.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 \
  python scripts/evaluate_g1_cricket_tracking.py \
  g1_cricket_results/bimanual_balanced_small_residual_v1/ppo_right \
  --output g1_cricket_results/bimanual_bounced_delivery_v1/right_coarse \
  --waist-tracking-gain 1 --root-position-gain 4 \
  --contact-dt 0.0000625 --bounced-delivery --compact-substeps
```

Repeat for left and for fine physics (`0.00003125`), adding `--render` to both
fine runs. Summarize with `scripts/report_g1_cricket_bounced_delivery.py`.
Compare every matched resolution pair under the existing force, penetration
and exit-velocity tolerances. Neither a nominal hit nor a rendered video clears
failed physical/tracking gates or the separate unfinished running-bowling goal.

## Complete Results

Source `b070b49f635a4b1896736fe46a4b68a4013c2fae`. All eight episodes finish
150 controls / three physical seconds. Every row has exactly one completed
pitch bounce before loaded blade contact, negative incoming x velocity and
forward first-exit vx above 1 m/s. Both frozen PPO actors and both reference
controls hit; this does not establish a learned interception improvement.

All rows pass pelvis height, grip, original joint/motor limits, unintended
contacts, root tracking and joint tracking. All fail the unchanged 0.08 m
bat-path gate. The complete 576,000 physical substeps are audited with exact
independent native endpoint and sensor replay. All 74 source/input hashes plus
the recorder hash per evaluation verify. No task parameters changed during
these eight evaluations.

Finest-resolution results, with zero residual retained alongside PPO:

| Hand / control | Exit vx (m/s) | Blade peak force (N) | Blade penetration (mm) | Peak bat-path error (m) |
| --- | ---: | ---: | ---: | ---: |
| Right reference | 2.6031 | 647.90 | 4.090 | 0.13699 |
| Right PPO | 2.2022 | 587.39 | 3.724 | 0.12848 |
| Left reference | 2.5997 | 646.79 | 4.065 | 0.13659 |
| Left PPO | 2.2971 | 613.97 | 3.855 | 0.13718 |

Every ball-contact penetration remains below 6 mm. All four resolution pairs
fail the pitch peak-force check: 1453.72 N versus 1378.99 N, a 5.4195% difference
against the 5% bound. Right reference also fails blade-penetration convergence
(3.787 versus 4.090 mm). All other paired checks pass. These failures are not
waived because the ball is hit or the clip looks plausible. The ball-only
rebound-height discrepancy above remains visible as additional evidence.

The [complete eight-row report](../g1_cricket_results/bimanual_bounced_delivery_v1/summary.json)
retains all four resolution comparisons and links all evaluation evidence by
hash. The [two-hand slow-motion video](../g1_cricket_results/bimanual_bounced_delivery_v1/two_hand_ppo_bounced_delivery.mp4)
contains both complete final-actor episodes at 0.5x, right then left: 350 frames
at 25 fps, including terminal labels. It is a development diagnostic, not a
qualified advertising reel. Full zero-residual videos remain for
[right](../g1_cricket_results/bimanual_bounced_delivery_v1/right_fine/reference_only_diagnostic.mp4)
and [left](../g1_cricket_results/bimanual_bounced_delivery_v1/left_fine/reference_only_diagnostic.mp4).
All 1,050 original/combined video frames decode nonblank at 960 x 540, and the
[fixed-time review sheet](../g1_cricket_results/bimanual_bounced_delivery_v1/bounced_delivery_contact_sheet.png)
was inspected. Redundant individual PPO clips were removed after combining;
the evaluation commands reproduce them before `--media` assembly.

All 103 focused tests pass with warnings as errors, including reset-only ball
state, native replay, contact ordering, preserved tracking/contact gates and
both timestep-specific ball-only fixtures. No new actor training, independently
learned Menagerie policy, running-bowling result, upstream PR or social launch
is part of this comparison.

## Third-Resolution Check

The next check changes only the physics timestep to 15.625 microseconds.
Keep the launch, both frozen actors, reference controls, gains, robot, contact
pairs, complete episodes and original gates fixed. Append all four outcomes
under `right_finest` and `left_finest`; do not replace the original eight.
Use the same evaluation command with `--contact-dt 0.000015625`, initially
without rendering. Run the reporter with `--include-finest` to retain all 12
rows and both adjacent-resolution comparisons in `resolution_refinement.json`.
The original eight-row summary and its videos remain unchanged.

This tests whether timestep refinement resolves the contact discrepancy,
not whether a different contact material or launch can produce a better clip.
Inspect first-bounce separation, first blade contact, complete peak force,
penetration and exit velocity. A converged contact comparison still cannot
clear the failed bat-path tracking check or establish learned interception.

### Refinement Results

All four added episodes finish three seconds, hit after one completed incoming
bounce, and pass the same height, grip, joint/motor, unintended-contact, root
and joint tracking checks. All still fail the 0.08 m bat-path bound. No actor
was retrained. Finest-resolution values:

| Hand / control | Exit vx (m/s) | Blade peak force (N) | Blade penetration (mm) | Peak bat-path error (m) |
| --- | ---: | ---: | ---: | ---: |
| Right reference | 2.6204 | 652.35 | 4.100 | 0.13689 |
| Right PPO | 2.2298 | 593.58 | 3.747 | 0.12840 |
| Left reference | 2.6169 | 651.08 | 4.070 | 0.13654 |
| Left PPO | 2.3095 | 616.86 | 3.825 | 0.13707 |

All four 31.25-to-15.625-microsecond comparisons pass every original contact
check. Pitch peak force changes from 1378.99 to 1387.88 N (0.6404% relative to
the finer result), compared with the original coarse pair's failing 5.4195%.
The first incoming pitch event is identical across all four controllers/hands
at each resolution:

| Physics step (microseconds) | Pitch separation (s) | Rebound vx (m/s) | Rebound vz (m/s) |
| ---: | ---: | ---: | ---: |
| 62.5 | 1.0604996 | -2.288405 | 2.920507 |
| 31.25 | 1.0604996 | -2.285301 | 2.812882 |
| 15.625 | 1.0604840 | -2.287324 | 2.852002 |

This is finite-grid consistency for the declared nominal feed, not material
calibration or proof that every future policy/contact will converge. Subsequent
batting controller studies should use 31.25 microseconds with a 15.625-microsecond
paired audit, repeating all contact checks rather than inheriting a pass.

The [12-row refinement report](../g1_cricket_results/bimanual_bounced_delivery_v1/resolution_refinement.json)
retains both adjacent comparisons for every hand/controller. The original
eight rows and four failed comparisons reproduce unchanged; the full report
remains failed, not promoted. The added 768,000 substeps bring the complete
pool to 1,344,000 audited substeps with exact independent native endpoint/sensor
replay. All 74 input hashes plus the recorder hash per case verify, and all
104 focused tests pass with warnings as errors. No media is regenerated or
replaced; the existing video remains a diagnostic. Tracking accuracy,
ball-aware learning and full running bowling remain unfinished.
