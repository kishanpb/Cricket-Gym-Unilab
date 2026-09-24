# Frozen-policy contact-resolution audit

Before further residual reward tuning, replay the final v3 checkpoint and zero
residual on the complete existing development pool: both hands, three toss lanes,
seeds 4301-4308. Run native closed-loop physics at 2, 1 and 0.5 ms while keeping
20 ms policy control, reset distribution, contact parameters and all gains fixed.
No training, checkpoint selection, model promotion or showcase claim is allowed.

At every control interval, independent serial replay must exactly match native
endpoint state and named sensors after native dtype conversion. The 2 ms rows
must also reproduce the retained evaluation's return, duration, blade occupancy,
peak blade load, first separation velocity and guard occupancy.

Retain all 288 rows: contact duration, maximum penetration, force-norm peak and
time integral, signed world contact impulse on the ball, first separation
velocity, guard presence and episode duration. Integral of force norm is not
the norm of net impulse. Contact loads come from the solve at the start of each
substep; velocity is the subsequently integrated state. A nonzero force norm
defines active load, whereas collision presence includes zero-force contacts.

Compare 1 vs 0.5 ms paired rows with identical presence/guard/completion outcomes.
Require differences within max(5% of the finer value, absolute tolerance):
1 N peak load, 0.005 N s force-norm integral, 2 ms contact occupancy, 0.1 mm
penetration and 0.05 m/s first separation velocity. These are predeclared
numerical consistency checks, not experimentally calibrated cricket material
properties. Report every mismatch; repeated seeds remain development data.

The world impulse uses MuJoCo's geom1-to-geom2 contact-force convention and
the transpose of its contact-frame rotation; a free-body momentum test checks
both geometry orderings and two impact directions. See the official
[contact formulation](https://mujoco.readthedocs.io/en/stable/computation/#contact).
