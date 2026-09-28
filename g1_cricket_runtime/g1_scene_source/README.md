# G1 Cricket Scene Sources

Three portable MuJoCo scenes from the retained G1 cricket demonstrations,
without redistributed robot meshes or machine-specific asset paths.

From this directory, install the pinned official Unitree meshes:

```sh
uv run --no-project python install_meshes.py .
```

The installer uses only Python's standard library. It downloads 47 meshes
from Unitree's `unitree_ros` revision
`ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee`, verifies their sizes, SHA-256 hashes
and Git blob hashes, then installs `assets/` only after every check passes.
Running it again checks existing files without downloading. A corrupt or
incomplete existing `assets/` directory is rejected, not silently overwritten.
Network failures leave no partial installation. Keep `assets/` out of git.

Load `batting.xml`, `bowling_right.xml` or `bowling_left.xml` with MuJoCo 3.11.0.
Both overarm and underarm controllers use the same bilateral bowling scenes.
The batting scene includes articulated hands, a physical bat and cricket ball;
runtime initialization supplies handedness, grasp, incoming delivery and policy.

The whole directory can be moved after installation. A fresh network install
followed by relocation preserved all three source models exactly across 484
compiled arrays, 127 scalar fields and names, excluding asset path metadata
and buffer sizes. This checks the scene plant, not a controlled cricket rollout.

`manifest.json` and `unitree_assets.json` preserve scene and mesh provenance.
Keep the original `UNITREE_LICENSE` (BSD-3-Clause) and `UNILAB_LICENSE`
(Apache-2.0). Mesh source: https://github.com/unitreerobotics/unitree_ros.

This is a scene-source package, not the complete runnable cricket release.
AMP/GR00T/PPO checkpoints, their live controllers and task/learner integration
remain separate work. It does not clear grip failures, certify legal bowling,
establish a G1 hardware limitation or constitute a new trained policy.
