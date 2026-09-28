"""Check downloaded preview-package bytes, not simulation correctness."""

import hashlib
import json
from pathlib import Path


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = folder / name
        if path.resolve().parent != folder.resolve():
            raise ValueError("Manifest member must be a file within this directory")
        data = path.read_bytes()
        actual = dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if actual != expected:
            raise ValueError(f"Package file differs from manifest: {name}")
    evidence = json.loads((folder / "evidence.json").read_text())
    batting = evidence["batting"]["rows"]
    expected = {(hand, feed) for hand in ("right", "left") for feed in range(7)}
    if len(batting) != 14 or {(r["hand"], r["feed_index"]) for r in batting} != expected:
        raise ValueError("Incomplete batting cohort")
    expected = {(hand, dt) for hand in ("right", "left") for dt in (0.0000625, 0.00003125)}
    for style in ("overarm", "underarm"):
        rows = evidence["bowling"][style]["rows"]
        if len(rows) != 4 or {(r["hand"], r["timestep"]) for r in rows} != expected:
            raise ValueError(f"Incomplete {style} cohort")
    if evidence["release_goal_complete"] or evidence["hardware_validated"] or evidence["raw_traces_included"]:
        raise ValueError("This preview cannot claim complete release/hardware/raw-trace validation")
    return len(manifest["files"])


if __name__ == "__main__":
    count = verify(Path(__file__).resolve().parent)
    print(f"Verified {count} package files and all 22 reported cases; this is not a physics rerun.")
