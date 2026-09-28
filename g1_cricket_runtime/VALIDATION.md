# Validation Scope

Validated locally on macOS arm64, Python 3.13.13, September 28, 2026.
This is a standalone development reproduction release, not a green full
UniLab integration or a completed legal-bowling policy.

| Check | Result |
| --- | --- |
| Native relocated batting, all 14 cases | 140 arrays and 1,011,437 physics samples exactly match; tactile records and result rows unchanged |
| Native relocated bowling, all 8 cases | 112 arrays and 1,536,000 physics samples exactly match |
| Fresh mjbatch wheel, relocated batting right-feed0 and right underarm | 24 arrays and 200,578 physics samples exactly match; two episodes, not a full cohort |
| Focused packaging/controller/scene/evaluation tests | 196 passed; exporter/verifier tests also rerun after metadata changes: 8 passed |
| Fresh mjbatch bridge tests | 3 passed; source-tree header inventory test excluded because wheels omit headers |
| UniLab `make check` | Passed with 7 optional-import warnings; its two unrelated formatting edits were restored to preserve existing files |
| UniLab `make test`, default fresh environment | Collection stopped because optional MuJoCo was absent |
| UniLab `uv sync --extra mujoco`, then `UV_NO_SYNC=1 make test` | 1,535 passed, 12 failed, 15 errors, 56 skipped, 848 deselected |

The broader UniLab suite is **not green**. An untouched Git archive of its base
also fails: 1,544 passed, 17 failed, 56 skipped, 848 deselected, one expected
failure. That archive has different asset paths and no Git worktree, so these
counts are not a matched regression comparison. The documentation checker
raises `IndexError: no such group` in both snapshots; other framework/asset
failures remain unresolved. No core framework source or tests were changed to
hide them, and no upstream PR was created or updated. `make test-all` was not
run after this failed non-slow gate.

The separately locked reproduction environment and relocated episodes are the
evidence for the commands in this package. Do not infer full framework, other
platform, new training, legal delivery or hardware validation from those checks.
