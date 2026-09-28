# G1 Cricket Runtime Snapshot

Local development package for the accepted 112,000-step PPO residual policy,
frozen GR00T balance and reference swing at evaluation pace 0.65. This is not
new training, a learned grasp, a boundary-hitting policy or the finished
UniLab/mjbatch task release. The shared package also preserves the overarm and
underarm bowling controllers: frozen AMP/GR00T locomotion, reference arm motion
and a mechanical ball holder, not learned bowling arms.

The snapshot includes the imported Gym-Cricket/UniLab Python source closure,
our PPO checkpoint, both original swing references and both height-adjusted
references, plus bilateral references for both bowling styles.
`runtime_manifest.json` fingerprints these inputs, the external
GR00T files and the expected simulator/policy runtime versions. Original source
files are copied unchanged except for the separately declared evaluator
scene-path options; historical training and evaluation proofs are not rewritten.

## Prerequisites

The complete fourteen-case batting cohort, executed from a relocated package
in the fresh environment with a freshly downloaded GR00T checkout, reproduces
the retained native evaluation exactly: 140 arrays, 1,011,437 physics samples,
simulated tactile records and physical/reward results. All 119 imported source
locations verify. Eight qualified contacts, ten physical passes and zero
boundaries remain unchanged, including four grip and three guard-return failures.

The complete relocated bowling cohort, using freshly downloaded GR00T/AMP
controllers, reproduces all eight retained cases exactly: 112 arrays and
1,536,000 physics samples, with 138 imported source locations checked. Both
styles release safely in the retained checks, but neither passes full bowling
qualification. Underarm first-bounce carry is 3.72--3.84 m versus 1.90--1.99 m
overarm; four-second travel including bouncing/rolling is 15.53--16.01 m versus
21.18--21.66 m. The release rules differ, so this is not a single-axis experiment
or evidence of a G1 hardware ceiling. Underarm is a non-regulation diagnostic.

- The locked Python environment below, tested on macOS arm64 with Python 3.13.13.
  `uv pip check` passes; Linux/Windows have not been validated.
  `runtime_manifest.json` also enforces the recorded core versions.
- A separate GR00T-WholeBodyControl checkout at revision
  `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`, including the real Balance and Walk
  ONNX files, not Git LFS pointers. The loader checks the revision and checkpoint
  hashes, and this package additionally checks the imported source/config bytes.
  The supplied installer fetches and verifies these inputs; external model
  weights are not redistributed in this package. Preserve the upstream license.
- The companion `g1_scene_source` directory with its verified meshes installed.
  Keep its Unitree and UniLab licenses; robot meshes are not part of this package.

## Run

From this directory, install the locked environment and external controller:

```sh
uv sync --frozen --python 3.13
uv run --frozen python install_controllers.py groot ../g1_controllers/groot \
  --runtime-manifest runtime_manifest.json
uv run --frozen python ../g1_scene_source/install_meshes.py ../g1_scene_source
```

The controller installer requires Git, not git-lfs; it fetches the pinned source
and verifies both ONNX payloads against reviewed LFS hashes. Existing controller
destinations are not overwritten. For bowling, also install the pinned AMP
running checkpoint, preserving its upstream license:

```sh
uv run --frozen python install_controllers.py amp ../g1_controllers/amp
```

Evaluate with live policy inference:

```sh
PYTHONPATH=src:src/scripts OMP_NUM_THREADS=1 uv run --frozen python -s -m integrations.g1_dynamics.portable_batting \
  . ../g1_controllers/groot ../g1_scene_source /path/to/new_output
```

The default executes all fourteen cases, both hands and all seven feeds, and
retains failures as well as contacts. `--workers 1` reduces the default two-worker
CPU load. `--case right 0` is a single complete
eight-second episode for relocation diagnostics, explicitly labeled a smoke
test, with no cohort-level success metric. Output must not already exist.

Run both bowling styles, both hands and both timesteps:

```sh
PYTHONPATH=src:src/scripts OMP_NUM_THREADS=1 uv run --frozen python -s -m integrations.g1_dynamics.portable_bowling \
  . ../g1_controllers/groot ../g1_controllers/amp ../g1_scene_source /path/to/new_bowling_output --workers 1
```

`--case underarm right 0.0000625` selects one complete diagnostic episode;
it is not a full-cohort result. No style's qualification gates are relaxed.

The optional native backend is pinned to the public mjbatch fork revision:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --frozen --extra mjbatch --python 3.13
```

Then add `--extra mjbatch` to `uv run` and `--backend mjbatch` to either runner.
The fresh native build passes three live bridge tests for contact forces,
adaptive stepping, holder release and motor limits. The source-tree header
inventory test is excluded because wheels omit C++ headers; the build is bound
to its Git revision and installed binary hash in `native_build_proof.json`.
This does not claim a full cricket-cohort rerun with the freshly built wheel;
earlier complete mjbatch equivalence runs used the development checkout.

The process checks packaged input bytes before evaluation and verifies imported
source locations so it cannot silently use the original workspace's cricket or
UniLab modules. GR00T is an explicitly supplied external dependency, not an
implicit workspace fallback. Traces retain policy actions, state/control arrays,
per-substep ball diagnostics and simulated tactile records.

Keep `GYM_CRICKET_LICENSE` and `UNILAB_LICENSE`. The original upstream projects
are https://github.com/kishanpb/gym-cricket,
https://github.com/unilabsim/UniLab,
https://github.com/NVlabs/GR00T-WholeBodyControl and
https://github.com/Jiarui-Xie/AMP_Running_baseline.
