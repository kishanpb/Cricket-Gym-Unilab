# Installed CPU Validation

Validated September 28, 2026 on macOS ARM64, Python 3.13, MuJoCo 3.11.0 and
PyTorch 2.8.0. The framework wheel was built from this fork at
`0833803bfb21810b7f284d77a12d5a2ec680d16c`; the adapter was installed alongside
it in a fresh environment without a development `PYTHONPATH`.

## Environment Parity

### Recovery Fade Fix: 0.1.1

The registered batting factory in 0.1.0 omitted the accepted policy's
`recovery_fade=True` setting. A learned-policy replay first diverged after
three seconds; zero-residual parity could not expose this omission. Version
0.1.1 restores the setting. Existing 0.1.0 scratch-training results below are
historical evidence, not results from the corrected task.

A separate environment, with dependency versions held fixed, verifies the
corrected wheel through the installed public registry. The original SKRL PPO
weights reproduce all fourteen hand/feed cases exactly: 140 trace arrays,
1,011,437 native samples, 5,614 observations including terminal observations,
5,600 actions/rewards, every tactile record and every physical result. Outcomes
remain 10 physical passes, 8 qualified hits and zero boundaries. This is not
new RSL-RL training or improved cricket skill. The compact replay summary has
SHA-256 `988181b2abe86b63be1de75852bc4f812759a95c2b69658654b04dc559c19617`.

The rebuilt source closure also incorporates the current optional bowling
diagnostic helpers. Default behavior is unchanged: all eight overarm/underarm
cases again match 112 arrays, 1,536,000 native samples and their observation,
reward and terminal-result streams. One zero-residual batting episode also
matches. All 87 installed adapter and 574 framework package files match their
wheels; 449 focused development tests pass, with one optional-package skip
covered by eight passing installed-environment checks.

The factory regression test is included in this package. From the repository
root with the runtime dependencies installed, run
`PYTHONPATH=g1_cricket_cpu/src uv run --no-sync pytest g1_cricket_cpu/tests -q`.

Repository-wide validation is not green: `UV_NO_SYNC=1 make test` reports
1,557 passed, four failed, 56 skipped, 848 deselected and one xfail. Failures
are a missing Allegro mesh, two unavailable MotrixSim cases and an IndexError
in the documentation validator. `make type check-tests` passes (one optional
Drake import warning). Read-only formatting/lint checks report ten formatting
files and six import-order issues outside this adapter; automatic fixes were
not applied to unrelated, fingerprinted release files. These results are not
framework-wide CI approval or upstream acceptance.

### Original 0.1.0 Checks

- One complete 400-step zero-residual batting episode reproduces the prior
  registry evaluation's observation/reward stream and terminal result exactly.
  This is not a repeat of the complete 14-case trained batting cohort.
- All eight retained bowling cases reproduce exactly: overarm and experimental
  underarm, both hands, at both native timestep resolutions. This covers 3,200
  control steps, 112 native arrays and 1,536,000 physics samples, plus complete
  terminal results and observation/reward streams.
- All loaded task/framework modules originate in the installed environment.
  All 79 adapter and 574 framework package files match the corresponding wheels.
  The exported package's 82 manifest files match their recorded hashes.

## Ordinary PPO Entry Point

Both tasks completed the documented `unilab.scripts.train_rsl_rl` connection
test with fresh learners, not the showcase checkpoint:

| Task | Rollout steps | Optimizer updates | Episode return | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| G1CricketResidualCpu | 400 | 2 | -16.233671 | 164.79 |
| G1CricketBowlingCpu, overarm | 400 | 2 | -5.403251 | 54.45 |

Strict normal-runner reload preserves actor/critic weights, all 13 optimizer
states with finite nonzero moments, and the 400-step logger count. Inference
returns a finite 29-action vector. These runs do not retain physical traces
and are not evidence of contact, legal bowling or improved policy performance.
The scoped exporter, CLI, registry and adapter test suite passes 50 tests;
this is not a framework-wide CI result or upstream approval.

## Artifact Identity

| Wheel | SHA-256 |
| --- | --- |
| g1_cricket_cpu-0.1.0-py3-none-any.whl | `4dda1b34c012384686b3d6cd89f87bbcb5f1556d7e6dcf89a4c87c43e3bc97ad` |
| g1_cricket_cpu-0.1.1-py3-none-any.whl | `722f49e916bcd0edaa3c1639cef8ce243eccf10cd7c00b231fcf3a9822ae372a` |
| unilab-1.2.0-py3-none-any.whl | `a10ed94a5a83edbab31b5716f727c709f61c8b30755050a258fc1187caa17047` |

The current source tree rebuilds the tested 0.1.1 wheel byte-for-byte; the 0.1.0
hash is retained for historical comparison. Meshes and controllers remain
separate asset inputs.

## Physical Limits Remain

Packaging changes neither the accepted videos nor the physics. No bowling
case fully qualifies: overarm retains three delivery failures per case,
underarm seven. Underarm has greater airborne carry in this setup, not greater
total distance including roll, and uses different release settings. It is an
experimental controller/setup comparison, not proof of a G1 hardware limit.
Mechanical ball holding and reference-arm control remain explicit limitations;
simulated force/contact signals are not measurements from hardware tactile
sensors. See the complete [showcase evidence](../g1_cricket_showcase/README.md).
