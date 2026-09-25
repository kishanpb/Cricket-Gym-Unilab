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
