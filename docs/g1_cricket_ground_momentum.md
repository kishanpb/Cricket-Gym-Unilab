# Running Stance Momentum

The preceding lateral-support reference still required fore/aft ground-force
locations outside its planted feet. This candidate changes only the run-up
centroidal angular-momentum target, retaining the full lateral-support COM,
foot and arm targets, stride/release times, original model and controller.
It is offline retargeting followed by unassisted native PD, not learned bowling.

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
