# First-impact motion and control headroom

This is a diagnostic of the final impact-events-v1 checkpoint, not training or
a new promotion gate. Freeze its fine-resolution evaluation, checkpoint, config,
reward, physics, reset seeds, both hands and zero/PPO controllers. Keep all 96
trial identities. Replay every contact-bearing row from reset through the control
interval containing its first blade separation; retain misses and guarded exits
as unavailable, with their original failures, rather than substituting zero-speed
measurements. Do not select episodes by success, force or appearance.

Require exact independent/native endpoint state and named-sensor parity for
each replayed interval. Require first blade onset and first separation time/vx
to equal the parent. This is prefix reproduction, not a second full-horizon
evaluation; the existing full report remains authoritative for safety and shots.

At the first blade contact, record solved contact position, signed normal and
force on the ball, bat-point velocity and relative normal velocity. Preserve
the onset load and first positive-load sample separately: zero-force occupancy
is not a loaded impact. Retain the complete first contact interval's integrated
world impulse, duration and peak force using the existing audit helpers.

For arm-control headroom use the 100 ms preceding first contact, excluding the
contact solve itself. Select by global solve index, with the earliest index
clipped to zero. Capture the single action actually applied by the native env;
derive joint columns through actuator/joint addresses. Store per-control targets,
raw actions, residual corrections, actual joint positions and overlap counts.
Compute physics-step-weighted per-joint fractions for abs(raw)>1 (clipped) and
abs(raw)>=1 (at a residual bound). Report integrated target-error RMS/peak and
joint-speed peak, plus solved actuator-force fractions and the fraction at
>=1-1e-6 of the symmetric actuator-force limit. Do not include impact forces in
these pre-contact statistics or combine phases into an instantaneous PD law.

The +/-0.35 rad first-four-joint and +/-0.15 rad wrist bounds are positional
corrections around an external frozen prior, not physical speed ceilings.
Receding/static/glancing impact despite unused range suggests inspecting the
learned timing or objective; persistent clipping/tracking lag suggests checking
control expressivity or actuator demand. Neither finding alone authorizes a
larger action range, a reward change, policy promotion, material calibration or
a showcase video. Pick a subsequent single-axis experiment only after inspecting
this diagnostic alongside the unchanged full failure table.
