# Ball-Pitch Contact Repair

The fixed motor search at UniLab revision
`d0cba961b0dfce81d0f1c4089bea9fdb9f235f17` exposed 34-48 mm pitch penetration.
Those failed trials and the delivery_v1 PPO pilot remain historical evidence;
reproduce them at that revision, not against the new registry below.

The opt-in `g1_cricket_delivery_pitch_v2/{mujoco,mjbatch}` owner changes only
ball/pitch contact response. It adds one explicit contact pair with `solref`
2 ms / damping ratio .3, unchanged default impedance and the original combined
friction coefficients. Geometry, mass, gravity, robot dynamics/actuators,
foot/ground contacts, holder, action space, reward and delivery gates are
unchanged. The new owner requires physics timestep <=.0625 ms, with .0625 ms as
its default, after the coarser force-resolution checks failed. No trained policy
is promoted or retrained by this repair.

[MuJoCo's solver documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)
defines the positive pair as time constant and damping ratio; a smaller time
constant stiffens contact and subcritical damping permits rebound. This is an
explicit engineering model, **not measured cricket-ball/pitch material calibration**.
Measured forces must retain that qualification; visually less overlap does not
prove real-world impact loads or robot safety.

## Declared Validation

The isolated matrix compares original and revised models at .25, .125,
.0625 and .03125 ms, starting the same .156 kg / 36 mm ball at 1.2 m with velocities
(0,0,0), (8,0,-4), (12,0,-8) m/s. Retain all 24 rows, each for .6 s. Initial
velocities are impact-test conditions, not a bowling demonstration. Native
Batch and independent serial MuJoCo must match every substep state/sensor;
world contact impulse must match momentum change minus gravity impulse.

The initial .125/.0625 ms comparison failed the 5% peak-force tolerance for
both oblique cases (6.31% and 6.17%); preserve those rows. Add the .03125 ms
resolution check rather than changing the force tolerance. For the revised
.0625/.03125 ms pairs, require penetration <6 mm, a positive
first upward rebound with outgoing/incoming translational energy ratio <.95,
momentum residual <1e-10 N s, peak force agreement <5%, impulse agreement within
1% plus .01 N s, and first-exit velocity agreement within 1% plus .1 m/s.
The .25/.125 ms rows are resolution diagnostics, not qualified impact rates.

Full robot integration uses the same fixed zero-residual hold/drop-at-1 s drill,
both hands, both executors and .0625/.03125 ms at seed 6301: eight complete
four-second rows. Retain all independent delivery-gate failures; a successful
ground-contact repair does not make this low drop an overarm or legal delivery.
The model test must prove the added pair leaves robot, holder, other contact
materials and keyframes unchanged. Read-only source/runtime hashes are recorded
before and checked after the audit; inputs are not edited during execution.

```sh
PYTHONPATH=src:scripts uv run --no-sync python scripts/audit_g1_cricket_pitch_contact.py
PYTHONPATH=src:scripts uv run --no-sync python -m pytest tests/scripts/test_g1_cricket_pitch_contact.py
```

Results are retained under `g1_cricket_results/pitch_contact_v2`. No new policy,
generalization claim or showcase video follows solely from this model audit.
