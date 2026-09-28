"""Install a verified training extension without changing the playback runtime."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(raw):
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(root, name, expected):
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
        raise ValueError(f"Invalid package path: {name}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"File escapes package: {name}")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"File differs from manifest: {name}")
    return raw


def install(base, output, extension=None):
    extension = Path(__file__).resolve().parent if extension is None else extension
    manifest = json.loads((extension / "manifest.json").read_text())
    extension_files = {name: checked(extension, name, expected)
                       for name, expected in manifest["files"].items()}
    original = json.loads(checked(base, "runtime_manifest.json", manifest["base_runtime_manifest"]))
    base_files = {name: checked(base, name, expected) for name, expected in original["files"].items()}
    target_bytes = extension_files["runtime_manifest.json"]
    target = json.loads(target_bytes)
    files = {}
    for name, expected in target["files"].items():
        if name in manifest["overrides"]:
            raw = extension_files["overrides/" + name]
        else:
            raw = base_files[name]
        if digest(raw) != expected:
            raise ValueError(f"Assembled runtime differs: {name}")
        if Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name:
            raise ValueError(f"Invalid runtime path: {name}")
        files[name] = raw
    files.update({name: checked(base, name, expected)
                  for name, expected in manifest["auxiliary_files"].items()})
    files["runtime_manifest.json"] = target_bytes
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = install(args.base.resolve(), args.output.resolve())
    print(f"Installed {count} verified runtime files; no training or policy qualification performed.")
