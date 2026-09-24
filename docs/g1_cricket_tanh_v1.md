# G1 smooth residual action experiment

Parent: impact-events-v1's contact-seeking reward and action contract, not the
failed forward-swing reward. The parent reaches all 24 right-hand deliveries but
has no valid shots. Its pre-contact headroom diagnostic shows shoulder-yaw and
elbow raw actions beyond the hard-clipping threshold for the entire 100 ms window.
The swing-v1 reward replacement regressed to zero matched-hand contacts and is
closed. This candidate changes action parameterization rather than reward again.

Owner: `g1_cricket_tanh_v1/mujoco`. The only changed axis is the seven-joint arm
residual transform:

```text
old residual = clip(raw_action, -1, 1) * residual_limits
new residual = tanh(raw_action) * residual_limits
```

Limits remain +/-0.35 rad for the first four arm joints and +/-0.15 rad for the
three wrists, added to the same frozen prior's targets. This removes exactly
flat executed-action tails at moderate raw values; it also changes amplitudes
inside the old clipping interval (tanh(1) is about 0.7616). At extreme inputs,
floating-point tanh can still saturate. This is not an expanded action range,
more powerful robot or proof of an achievable swing-speed ceiling.

PPO samples and records raw Gaussian actions. Keep those arrays, raw action
history, action-rate penalty and 115-input/7-output policy contract unchanged;
apply the deterministic transform only to executed residuals. Do not mutate
input arrays that may alias Torch tensors. PPO still scores raw actions, so no
transformed-action Jacobian belongs in its likelihood calculation. No simulator
gradient is being restored. Inherit the existing prior inference, actuator
application and partial-reset behavior; there is no reference swing or pose write.

The changed action config is a sim2sim-denylisted field. Reject loading the old
action contract into this owner even though dimensions agree. Train from scratch
with seed 1, right hand, four CPU environments, 24 steps/update and 2080 updates
(199680 transitions). Preserve the parent reward, optimizer, network, external
prior, reset lanes, 0.25 ms physics, 20 ms control, robot inertias/motor limits,
bat/ball compliance, sensors, guards and observation meanings. No resume, budget
extension or intermediate checkpoint selection.

Before training check exact zero-action native physics/reward/observation parity
for both hands and both evaluation timesteps; residual bounds, odd/monotone
mapping, finite extreme values, distinct mapped commands at raw1.5/raw1.6,
unaltered input/history, unchanged other joints and partial resets. Pin config,
source, parent, headroom and the closed swing result before the run.

Evaluate only the final checkpoint on all 96 identities at each of 0.25 and
0.125 ms: both hands, zero residual/PPO, all three lanes, seeds 4301-4308.
Left PPO remains untrained transfer. Require exact zero-residual baseline rows,
INCLUDING returns because this candidate's reward is identical to the parent.
Keep strict first-separation vx >1 m/s, first contact blade-only, full two-second
stability, all contact/joint/actuator gates, penetration <=6 mm and unchanged
numerical-comparison tolerances. Keep every failed row and input hash.

Stop on non-finite training or failed provenance/parity. If the final full pool
still has no valid shots, close this bounded candidate without promotion rather
than extending its budget or relaxing gates. More contacts or training return is
not learned batting. Physical calibration, independently learned hands, bowling,
native mjbatch execution and validated videos in both READMEs remain separate.
