# Signed G1 Elbow and Release Reward Audit

This is an additional gate, not replacement evidence for the six scripted
`overarm_release_v1` trials. Every trial is replayed with unchanged controls,
reward, dynamics and horizon, and every old outcome and complete control trace
must match exactly. Native endpoint state and named sensors still match serial
MuJoCo; every substep is inspected. No new training or video is claimed.

The old shoulder/elbow/wrist angle used unsigned arccos, which folds through
straight. The new measurement projects the forearm and proximal segment onto
the plane perpendicular to the actual elbow hinge axis, uses oriented atan2,
and unwraps sequential angles. Its proximal landmark is the shoulder-yaw body
origin, rigidly attached to the elbow's parent. The shoulder-roll origin is
upstream of a translated yaw joint and would confound extension with shoulder
yaw. Both-hand randomized whole-body/shoulder poses and the complete elbow
range verify signed angle minus elbow joint position stays constant to 1e-12.
The angle is robot hinge geometry, not a calibrated human anatomical angle.

Upward shoulder-level crossing uses the original upper-arm geometry. From that
first crossing until release, increasing signed angle above its running minimum
measures extension; >15 degrees adds a failure. All old failures remain. Missing
signed release or crossing evidence fails closed. Body positions and axis come
from the same solved substep state; release uses fresh kinematics of the actual
integrated pre-release snapshot. The audit never writes into the running state.
This is a simulation engineering gate, not umpiring certification.

The existing reward also uses release-ball x<0 as a proxy for legal foot
placement. All six retained releases have ball x>0 despite passing the current
foot/stride checks; their legacy release bonus is therefore zero. The separate
owner `g1_cricket_overarm_reward_v2/{mujoco,mjbatch}` removes only this reward
proxy. It changes no dynamics, actions, observations, contact checks or delivery
gate. Reward is still shaping, not evidence of legality or a successful delivery;
foot faults and all other failures remain disqualifying in independent evaluation.
Tests preserve zero bonus for backward or repeated releases and prove all other
owner configuration is identical to `g1_cricket_overarm_guard_v1`.

The report includes the counterfactual integrated release component under the
new reward, computed from retained raw release state. It is not a recomputed
episode return, a trained-policy result or an improvement in ball dynamics.

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 python scripts/audit_g1_cricket_signed_release.py
```

The canonical output is `g1_cricket_results/overarm_release_v1/signed_elbow_audit.json`.
The script refuses to overwrite it and checks pinned sources, runtime, executor,
resolved old-owner configuration and parent hashes before and after all rows.
Original six-release evidence remains frozen at revision
`e45cfdcddf18996a7158a3d073d17756e8e71760`; do not rewrite its preflight to claim
it was generated with newer gates or rewards.
