# Prior Target Guard: Fixed Reach Comparison

The eight absolute-arm trials achieved sampled overhead positions in seven
cases, but all exceeded leg joint limits. Exact diagnostic replay attributes
every first/worst joint violation to the other 22, prior-controlled joints.
Six first violations occur with a target outside that joint's physical range;
one later hip-roll target is -3.826 rad against a -.5236 rad lower limit.
Right-hand trials also make foot/foot or opposite-hand/hip contact. No prior
reach trial is a validated teacher or delivery.

Change one control axis: after computing the frozen prior, constrain its 22
joint-position targets to the central 90% of their original model joint ranges.
The seven absolute bowling-arm targets, selected trajectories, seed 6301,
handedness, .0625 ms native executor, four-second horizon, reward, contact
materials, PD gains, physical joint/torque limits and independent gate stay
unchanged. This is a command guard, not stronger motors or a claim that clipping
targets guarantees actual joint limits. Default overarm_v1 remains unguarded.
The explicit owner is `g1_cricket_overarm_guard_v1/{mujoco,mjbatch}`.

Replay all eight parent cases in the original order. Retain full pose traces,
all failures, first/worst physical violations, control-target clipping counts
and the same sampled-preload readiness definition. Require exact independent
serial endpoint state/sensor replay. Check source/runtime/executor identities
before and after. No adaptive poses, extra seeds, relaxed limits or new learning
are included. A passing reach is still only a scripted preload witness; it
requires subsequent paired-resolution delivery and learning validation before
it can support a showcase claim.

Keep parent reach evidence frozen at `1784ee652712ab08f6ecaf95368bc00646d8d518`
and its diagnostic replay at the following attribution commit; source-hash
contracts are revision-specific. Outputs are limited to
`g1_cricket_results/overarm_guard_v1/{preflight,evaluation}.json`.

```sh
PYTHONPATH=src:scripts uv run --no-sync python scripts/probe_g1_cricket_overarm_guard.py
PYTHONPATH=src:scripts uv run --no-sync python -m pytest tests/scripts/test_g1_cricket_overarm_guard.py
```
