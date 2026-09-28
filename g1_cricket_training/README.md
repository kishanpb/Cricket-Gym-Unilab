# Train G1 Bowling on CPU

Optional SKRL PPO training for the [G1 cricket reproduction](../g1_cricket_runtime/README.md).
The installer verifies the exact playback-runtime version and creates a separate
training directory. It does not modify the playback package, checkpoints or videos.

## Run

Requires Python 3.13, uv and a C++ toolchain for the optional mjbatch backend.
Tested on macOS arm64; other platforms have not been validated. From the fork root:

```sh
python3 g1_cricket_training/install.py g1_cricket_runtime/runtime g1_cricket_training/runtime
cd g1_cricket_training/runtime
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --frozen --extra mjbatch --python 3.13
test -d ../../g1_cricket_runtime/g1_controllers/groot || uv run --frozen python install_controllers.py groot ../../g1_cricket_runtime/g1_controllers/groot --runtime-manifest runtime_manifest.json
test -d ../../g1_cricket_runtime/g1_controllers/amp || uv run --frozen python install_controllers.py amp ../../g1_cricket_runtime/g1_controllers/amp
uv run --frozen python ../../g1_cricket_runtime/g1_scene_source/install_meshes.py ../../g1_cricket_runtime/g1_scene_source
PYTHONPATH=src:src/scripts OMP_NUM_THREADS=1 uv run --frozen --extra mjbatch python -s -m integrations.g1_dynamics.portable_bowling_ppo . ../../g1_cricket_runtime/g1_controllers/groot ../../g1_cricket_runtime/g1_controllers/amp ../../g1_cricket_runtime/g1_scene_source ../../g1_cricket_runtime/results/bowling_ppo --backend mjbatch --style overarm --steps 400 --seed 0
```

Use a new output directory. Existing controller downloads are reused; the learner
checks their required source and checkpoint hashes before simulation.
The command first compares all four hand/timestep cases against frozen-controller
MuJoCo rollouts, then trains, reloads one final checkpoint and evaluates all four
cases. Budgets must be positive multiples of 400 transitions. Source or asset
drift fails the run; incomplete runs have no completed summary.

Use `--backend mujoco` for direct MuJoCo stepping. The code also accepts
`--style underarm`, but the portable training validation below covers overarm only.
One native mjbatch member is used per environment; no batching speedup is claimed.

## Verified Scope

The fresh locked-environment run completed four-case preflight, 400 training
steps, 16 updates and four final evaluations in about 14 minutes including
preflight. It reproduced the development checkpoint byte-for-byte, all 64
evaluation arrays and every physical outcome. See [validation.json](validation.json)
for all four cases, failures, simulated contact/holder forces and source hashes.
The package contains source changes and evidence, not external models or raw traces;
the command produces new traces and a checkpoint.

**Zero of four deliveries passed full qualification.** Early/repeated bounce
and target-corridor failures remain. PPO controls bounded torque residuals over
frozen AMP running, GR00T balance, reference arms and a mechanical ball holder;
it does not learn the complete bowling action or a finger grasp from scratch.
This is a reproducible learning connection, not improved skill, convergence,
hardware validation or proof of a G1 hardware ceiling. The accepted videos remain
development demonstrations with their existing limitations.

The extension inherits the source licenses in the verified base runtime.
External controllers and robot assets retain their upstream notices and terms.
