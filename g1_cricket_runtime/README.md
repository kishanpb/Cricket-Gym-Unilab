# Reproduce G1 Cricket

An isolated CPU reproduction of the [batting and bowling videos](../g1_cricket_showcase/README.md).
It does not replace the fork's installed library or register a new UniLab task.
The frozen source snapshot is deliberately separate from the framework source
tree. Final task/learner integration and physical qualification remain open.

## What Runs

- Batting: our 112,000-step SKRL PPO residual checkpoint, frozen GR00T balance,
  bilateral reference swings and physical bat/ball contact. Evaluation pace is
  0.65; changing pace after training is not new learned power.
- Bowling: frozen AMP running and GR00T balance with reference arm motion and
  a mechanical ball holder. Both overarm and underarm, both hands, two physics
  timesteps. Bowling arms and finger grasp are not learned policies.
- Observations include privileged simulator state and simulated contact signals,
  not a vision-only policy or hardware tactile sensors.

## Install And Run

Requires Git, uv and Python 3.13; tested on macOS arm64, not Windows/Linux.
The optional mjbatch backend also requires a working C++ build toolchain.
From this directory:

```sh
uv run --no-project python verify.py
cd runtime
uv sync --frozen --python 3.13
uv run --frozen python install_controllers.py groot ../g1_controllers/groot --runtime-manifest runtime_manifest.json
uv run --frozen python install_controllers.py amp ../g1_controllers/amp
uv run --frozen python ../g1_scene_source/install_meshes.py ../g1_scene_source
PYTHONPATH=src:src/scripts OMP_NUM_THREADS=1 uv run --frozen python -s -m integrations.g1_dynamics.portable_batting . ../g1_controllers/groot ../g1_scene_source ../results/batting --workers 2
PYTHONPATH=src:src/scripts OMP_NUM_THREADS=1 uv run --frozen python -s -m integrations.g1_dynamics.portable_bowling . ../g1_controllers/groot ../g1_controllers/amp ../g1_scene_source ../results/bowling --workers 1
```

These execute all fourteen batting and eight bowling cases, retaining failures.
Output directories must be new. Runs take minutes on CPU; `--workers 1` limits
batting to one worker. `--case right 0` for batting or
`--case underarm right 0.0000625` for bowling runs one diagnostic episode only,
not a complete success-rate evaluation.

For live native mjbatch stepping, install the pinned optional backend:

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --frozen --extra mjbatch --python 3.13
```

Then add `--extra mjbatch` to `uv run` and `--backend mjbatch` to either evaluation
command. The bridge uses one native batch member per CPU worker; no throughput
speedup or independent training claim is made. The policy still runs live.

## Evidence And Limits

The fresh-environment relocated native runs exactly reproduce all retained
cases: 140 batting arrays/1,011,437 physics samples and 112 bowling
arrays/1,536,000 samples. Batting tactile JSON and complete physical/reward
results also match. The two `*_relocation_proof.json` files bind the actual
tested source snapshots, which differ as bowling was packaged after batting.
Raw traces and archived historical source snapshots are not redistributed here;
the commands generate new traces. The package manifest checks shipped bytes,
not policy performance.

Earlier complete live mjbatch runs matched the native cohorts; the newly built
wheel separately passes three contact/release/motor bridge tests. Two complete
relocated episodes using that wheel, right-feed0 batting and right underarm,
also match all 24 arrays and 200,578 physics samples exactly, including batting
tactile records. This is not a fresh full cricket cohort.
`runtime/native_build_proof.json` and `native_wheel_smoke.json` record these
scopes. The [complete demo results](../g1_cricket_showcase/evidence.json) retain
all failures, rather than selecting successful episodes.

Batting has eight qualified contacts, ten physical passes and zero boundaries;
four grip and three guard-return failures remain. Neither bowling style passes
full delivery qualification. Underarm carries farther before the first bounce
(3.72--3.84 m versus 1.90--1.99 m), but travels less over four seconds including
rolling (15.53--16.01 m versus 21.18--21.66 m). Release rules differ: this is a
controller/setup comparison, not a single-axis ablation or a G1 hardware ceiling.
Underarm is a non-regulation diagnostic, not a legal-delivery substitute.

## Assets And Licenses

Our PPO checkpoint and reference files are included; external AMP/GR00T weights
and Unitree meshes are downloaded separately at pinned revisions and verified
by hashes. Keep their upstream notices and model license terms. Existing
controller downloads are never overwritten; the mesh installer rejects corrupt
existing files. Generated downloads, environments and results are gitignored.

Source notices: `runtime/GYM_CRICKET_LICENSE`, `runtime/UNILAB_LICENSE`,
`g1_scene_source/UNITREE_LICENSE`, and `g1_scene_source/UNILAB_LICENSE`.
Controller installers preserve the original upstream license files.
