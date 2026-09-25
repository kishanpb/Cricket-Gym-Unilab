# Lateral Running Support

This experiment changes only the offline lateral center-of-mass (COM) target.
The earlier run-up keeps COM lateral position fixed while one foot supports
the robot, approximately 12 cm off the centerline. A support-wrench audit
checks whether that motion demands a ground-force location outside the foot.
These inferred forces are not measured contact telemetry or learned results.

## Design

For a level ground plane, total force is `F = mass * (COM_acceleration - gravity)`.
The moment about the world origin is `H_dot + COM cross F`, where `H_dot` is
the whole-system centroidal angular-momentum derivative, including the held
ball. Required center of pressure (CoP) is `[-moment_y, moment_x] / F_z`.
The [centroidal ZMP derivation](https://scaron.info/robotics/zero-tilting-moment-point.html)
explains this necessary balance relation; being inside the foot does not
certify joint torques, friction, yaw moment or a feasible full-body motion.

`LateralSupportCOM` first neglects `H_dot_x` to construct a candidate:
`y_ddot = (gravity + z_ddot) * (y - support_y) / z`. It solves the linear
variable-height stance equation for a periodic alternating solution, with
0.22 s stance and 0.08 s ballistic flight per step. Lateral velocity is
continuous at takeoff and landing. The existing 1.20-1.32 s gather connects
position/velocity back to the parent COM trajectory. Forward and vertical COM,
foot targets, arm motion, release time and original robot limits are unchanged.
This is an offline reference, not a live root force or simulation pose write.

`audit_g1_cricket_running_support.py` then restores full angular-momentum
terms using MuJoCo and differentiates the fixed reference curve at two
resolutions. Every 10 ms run-up sample, including phase boundaries, remains
in the report. Feet within 2 mm of the pitch contribute to a conservative
axis-aligned projected support box. Leaving that outer box rejects planar
support at the sample; entering it is not a feasibility certificate.
Flight rows without appreciable vertical force have no defined CoP.

```sh
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run --no-project \
  --python ../unilab_submission_checkout/.venv/bin/python \
  python scripts/retarget_g1_cricket_running.py \
  g1_cricket_results/running_support_v1 --render \
  --ballistic-parent g1_cricket_results/running_front_raise_v1 \
  --lane-offset 0.2 --conserve-momentum --lateral-support
```

Then run `scripts/audit_g1_cricket_running_support.py` on each of
`g1_cricket_results/running_momentum_v1` and `g1_cricket_results/running_support_v1`.
Outputs must not already exist. Both complete physical PD episodes remain
required; reference improvements alone cannot qualify a bowling video.
