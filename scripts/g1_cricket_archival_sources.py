"""Verify retained experiments across the explicitly versioned executor extension."""

import hashlib
import subprocess

REVISION = "d62de8a79c75ec7873587849fe9116eee2622fa1"
MIGRATED = {
    "src/unilab/base/base.py": "bee6e7f5bdee5bb8a676161af22d522aca4329dcab74843dc6e90689ffea6693",
    "src/unilab/base/backend_factory.py": "c901cd62ca6372edd77ddf7952c4e13411a3e603c1d68e2b58e7513c5b5c87b9",
    "src/unilab/base/mujoco_substeps.py": "08a265cade7e54b728bf12087985b4609be838f397eb9bdd217e8ea822c7d8b0",
}


def check_retained_hashes(root, hashes):
    """Only these exact historical source pairs use Git; other inputs stay live."""
    for name, expected in hashes.items():
        content = (
            subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=root)
            if MIGRATED.get(name) == expected
            else (root / name).read_bytes()
        )
        assert hashlib.sha256(content).hexdigest() == expected, name
