"""Resolve explicitly migrated legacy sources at their immutable audit revision."""

import hashlib
import subprocess

REVISION = "d088364fe7caaf61f0a1922cd83a066cf7447495"
MIGRATED = {
    "src/unilab/tasks/manipulation/g1_cricket/task.py",
    "scripts/evaluate_g1_cricket_residual.py",
}


def legacy_source_digest(root, name):
    content = (
        subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=root)
        if name in MIGRATED
        else (root / name).read_bytes()
    )
    return hashlib.sha256(content).hexdigest()
