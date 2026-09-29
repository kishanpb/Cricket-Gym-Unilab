# G1 Cricket CPU Tasks

Installable external UniLab tasks for the retained G1 batting, overarm bowling
and experimental underarm environments. Physics helpers live in the private
`g1_cricket_cpu` namespace; installation does not replace UniLab source files
or register the copied helpers as legacy tasks.

The required robot meshes, AMP/GR00T controllers, PPO checkpoint and motion
references are separate from this wheel. Install the matching asset bundles
from the fork's `g1_cricket_runtime` release instructions first. The tested
platform is macOS ARM64, Python 3.13, CPU MuJoCo 3.11.0 and PyTorch 2.8.0;
other platforms are not certified by local verification.

## Install

Build the matching UniLab fork wheel and this adapter, then install both into
a fresh environment. Use the wheel paths produced by each build:

```sh
uv build --wheel UNILAB_CHECKOUT
uv build --wheel G1_CRICKET_CPU_SOURCE
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python UNILAB_WHEEL CRICKET_WHEEL
```

No development `PYTHONPATH` is needed. Expose only the external registry package:

```sh
export UNILAB_EXTRA_REGISTRY_PACKAGES=g1_cricket_cpu
CONFIG=$(.venv/bin/python -c 'from importlib.resources import files; print(files("g1_cricket_cpu") / "conf")')
.venv/bin/python -m unilab.scripts.train_rsl_rl --config-dir "$CONFIG" \
  task=g1_cricket_residual_cpu/mujoco \
  env.runtime_bundle=RUNTIME_BUNDLE env.groot_checkout=GROOT_CHECKOUT \
  env.scene_bundle=SCENE_BUNDLE training.log_dir=BATTING_OUTPUT
```

For bowling, select `task=g1_cricket_bowling_cpu/mujoco` and additionally set
`env.amp_checkout=AMP_CHECKOUT env.delivery_style=overarm`; use `underarm` for
the separate experimental delivery. Choose a distinct output directory.
Replace capitalized placeholders with installed asset paths.

## Scope

The supplied command config is a 400-step, one-iteration PPO connection test,
not a trained-skill result. It initializes a new learner, not the retained
batting PPO checkpoint. Actions, rewards, contact telemetry, native force
limits and qualification rules retain the existing environment contract.
Both task registrations use CPU MuJoCo; this is not a ManagerBasedRlEnv rewrite
or an installed mjbatch task registration.

Batting uses a physical two-handed bat grasp, reference motion, frozen GR00T
balance and optional learned residual control. Version 0.1.1 restores the
retained policy's recovery fade, which attenuates residual actions after the
swing; 0.1.0 omitted this setting in the registered task. Its zero-action check
could not expose the omission. Bowling uses learned locomotion
priors, reference arms and a mechanical ball holder, not learned free-finger
release. Current bowling fails the full delivery checks; underarm is an
experimental comparison. Simulated contact signals are not hardware tactile
measurements. Installation or a PPO update does not establish legal bowling,
new learned skill, hardware transfer or release-goal completion.
