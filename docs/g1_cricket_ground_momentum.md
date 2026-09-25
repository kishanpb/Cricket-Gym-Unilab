# Running Stance Momentum

The preceding lateral-support reference still required fore/aft ground-force
locations outside its planted feet. This candidate changes only the run-up
centroidal angular-momentum target, retaining the full lateral-support COM,
foot and arm targets, stride/release times, original model and controller.
It is offline retargeting followed by unassisted native PD, not learned bowling.

## Complete Results

![All support estimates, including startup and transitions](../g1_cricket_results/running_ground_momentum_v2/support_comparison.png)

Both hands retain all 119 run-up times at two derivative resolutions. At the
finer 0.625 ms derivative resolution, 91 samples per hand have defined support
estimates, including seven phase-boundary samples. The projected foot box is
a necessary outer bound, not a complete contact or actuator certificate.

| Reference | Fore/aft violations, right/left | Lateral violations, right/left | Physical fall, right/left |
| --- | --- | --- | --- |
| Lateral-support parent | 33/33 of 91 | 3/3 of 91 | 0.66/0.66 s |
| Stance momentum, 20 ms knots | 3/3 of 91 | 18/18 of 91 | 0.64/0.64 s |
| Stance momentum, 5 ms knots | 0/0 of 91 | 2/2 of 91 | 0.68/0.68 s |

The dense reference removes sampled fore/aft violations, but does not pass:
lateral support excess reaches 261.3/261.9 mm at startup (0.01 s), and another
violation occurs at the 0.90 s landing boundary. Discrete momentum errors below
5e-13 Nms over 240 intervals per hand do not certify interpolated dynamics.
The dense reference also has a shoulder/torso overlap of 1.99/2.03 mm at
0.995 s, maximum foot error 5.01/5.02 mm and arm error below 5.91 mm. All IK
solves terminate successfully, but that cannot override these defects.

![Complete physical failures; every controller runs at 20 ms](../g1_cricket_results/running_velocity_v1/physical_comparison_review.png)

The relative-rate comparison below reproduces both absolute-rate parent pose
sequences bit-for-bit. Both settings fall at 0.68 s with hand/hip/thigh/wrist
contacts, no release and no ball penetration. Absolute-rate first joint-stop
crossings are 0.18175/0.18181 s; relative-rate crossings are slightly earlier,
0.17875/0.17881 s. Relative-rate peak joint excess is 0.08601/0.08575 rad,
versus 0.08935/0.08910 rad for absolute damping. It is not a successful control
fix and is not enabled in the default controller or existing trained actors.

Complete [coarse](../g1_cricket_results/running_ground_momentum_v1/evaluation.json),
[dense](../g1_cricket_results/running_ground_momentum_v2/evaluation.json), and
[four-row rate comparison](../g1_cricket_results/running_velocity_v1/evaluation.json)
reports remain, with all reference/control knots and 43,520 substeps for the
rate comparison. Frozen source revisions are `d2e545b7`, `634385fd`, and
`40f899eb`. The corresponding 36/36/38 input hashes verify; both support
audits' 37-input fingerprints also verify against their frozen sources.
All 101 focused tests pass with warnings as errors, including shared batting
balance regressions; Ruff passes. All 750 generated frames decode nonblank,
and the motion/support reviews were inspected. Redundant coarse MP4s were
pruned after review; complete poses, reports and the comparison sheets remain.
The dense and relative-rate videos are retained as failed diagnostics.

Next: full inverse-dynamics/contact feasibility under the original motor
limits, including startup and landing transients and shoulder clearance.
Do not equate centroidal support or lower joint error with a runnable teacher,
relax the physical gates, or train PPO on this reference as if it passed.
Batting and the independent Menagerie integration are unchanged and unfinished.

## Construction

The COM advances at constant forward velocity during the run-up. Its stance
vertical acceleration is constant, giving `F_z = mass * (gravity + z_ddot)`.
Choose a fixed fore/aft CoP at the COM's position halfway through each 0.22 s
stance. Required pitch torque is then `F_z * velocity_x * (phase - 0.11)`.
Integrating gives a pitch-momentum offset
`F_z * velocity_x * phase * (phase - 0.22) / 2`, returning to zero at takeoff.
Momentum is constant through each 0.08 s flight. The lateral COM already
supplies the alternating-foot force with zero required roll torque.

The constant momentum offset preserves the parent's mean over its first
complete 0.60 s gait cycle; there is no outcome-based seed or checkpoint
selection. Initial momentum and inferred vertical force are recorded in the
evaluation. The held-ball implicit-midpoint solver now determines offline
root rotation throughout stance and flight, instead of forcing the torso
back to level at every landing. After 1.20 s it uses the existing gather
recovery. No live root forces, pose writes, joint-limit changes or prescribed
ball velocity are introduced.

The ideal reduced-model support relation is not a full feasibility proof.
Whole-body retargeting, interpolation, joint limits, motor authority, yaw
friction and contacts can still invalidate it. The full-momentum support audit
and both complete physical trials remain required, including failed motion.

The coarse candidate uses 20 ms retargeting. Its discrete momentum matches do
not prevent large derivative errors between frames. A fixed 5 ms refinement
keeps the same mechanism and 20 ms physical controller, saves all 541 dense
poses per hand, and uses the corresponding dense forward-difference velocity
at each control knot. The support audit reads the dense reference explicitly,
not a cubic reconstruction from only the 136 control knots.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_ground_momentum_v2 --render \
  --ballistic-parent g1_cricket_results/running_front_raise_v1 \
  --lane-offset 0.2 --conserve-momentum --lateral-support \
  --ground-momentum-parent g1_cricket_results/running_support_v1 \
  --retarget-substeps 4
```

Then run `scripts/audit_g1_cricket_running_support.py` on the output directory.
Existing output directories are not overwritten. Passing an offline momentum
check cannot qualify a physical bowling delivery or authorize showcase claims.

## Reference Angular Rate

The existing ankle balance term damps absolute angular velocity. At the exact
initial dense target, intended pitch rate is about 3.58 rad/s, yet this term
adds its maximum 0.3 rad ankle correction. An opt-in comparison instead
subtracts desired angular velocity after expressing both rates in the reference
body frame. An exactly tracked rotating target then has zero rate error.
The default controller and existing learned-policy behavior remain unchanged.

`scripts/evaluate_g1_cricket_running_velocity.py` compares both settings for
both hands against the frozen dense reference, retaining all substep records.
The absolute-rate controls must reproduce the saved parent poses bit-for-bit.
Only relative-rate videos are newly rendered; parent baseline videos remain
in the reference directory. Use reference directory
`g1_cricket_results/running_ground_momentum_v2` and a fresh output directory
`g1_cricket_results/running_velocity_v1`, with `--render`.
