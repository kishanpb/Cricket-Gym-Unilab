"""Check the distributed cricket files; does not run or qualify a policy."""

import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"File escapes package: {name}")
        raw = path.read_bytes()
        if {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != expected:
            raise ValueError(f"File differs from manifest: {name}")
    return len(manifest["files"])


if __name__ == "__main__":
    count = verify(Path(__file__).resolve().parent)
    print(f"Verified {count} distributed files; this is not a physics or policy test.")
