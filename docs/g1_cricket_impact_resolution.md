# Impact v1 finer-resolution extension

The completed 0.5/0.25 ms frozen-policy comparison passed the 6 mm penetration
and stability preflight but failed numerical consistency in 30/96 pairs.
Before interpreting peak loads, extend the same complete 96-row development
pool to 0.125 ms physics. Compare against the retained 0.25 ms report, verifying
its source, checkpoint, run configuration and parent-report hashes first.

Only physics resolution changes. Keep the explicit 4 ms ball/blade contact
response, 20 ms control interval, frozen residual-v3 checkpoint, zero-residual
baseline, both hands, three lanes, eight seeds and all shot/physical gates.
The same 6 mm penetration bound applies. No training or selected-row replay is
part of this diagnostic extension; this is not a held-out evaluation.

Paired rows must retain contact presence, failure reasons and full duration.
Numerical tolerances remain 5% of the finer value with absolute floors of
0.1 mm penetration, 1 N peak contact-force norm and 0.05 m/s first-separation
x velocity. Report non-contact pairs separately from contact-bearing pairs.
All pairs must pass before claiming consistency over these two resolutions;
even that does not establish physical calibration or asymptotic convergence.

Retain one complete `impact_v1/resolution_extension.json` with all new rows,
parent hash and per-pair differences. Keep the old frozen-transfer report intact.
If the check fails, identify which observables remain resolution-sensitive;
do not loosen tolerances, discard failures or claim realistic force peaks.
