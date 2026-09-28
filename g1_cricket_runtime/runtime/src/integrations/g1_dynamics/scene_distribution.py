"""Export cricket scene sources and install their pinned Unitree meshes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request


REVISION = "ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee"
REPOSITORY = "https://github.com/unitreerobotics/unitree_ros"
LICENSE_SHA256 = "84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d"


def digest(raw):
    return dict(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked_file(bundle, name, expected):
    path = bundle / name
    if not path.resolve().is_relative_to(bundle.resolve()):
        raise ValueError(f"File escapes bundle: {name}")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"File fingerprint mismatch: {name}")
    return raw


def read_contract(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if not manifest["mesh_source_verified"]:
        raise ValueError("Mesh provenance has not been verified")
    provenance = json.loads(checked_file(
        bundle, "unitree_assets.json", manifest["mesh_source_provenance"],
    ))
    if provenance["revision"] != REVISION or provenance["repository"] != REPOSITORY:
        raise ValueError("Unexpected Unitree source revision or repository")
    license_bytes = (bundle / "UNITREE_LICENSE").read_bytes()
    if hashlib.sha256(license_bytes).hexdigest() != LICENSE_SHA256:
        raise ValueError("Unitree license fingerprint mismatch")
    if provenance["assets"].keys() != manifest["assets"].keys():
        raise ValueError("Mesh inventory differs from provenance")
    for name, expected in manifest["assets"].items():
        if name != f"assets/{expected['sha256']}.stl":
            raise ValueError("Mesh filename must be its content hash")
        source = provenance["assets"][name]
        if {key: source[key] for key in expected} != expected:
            raise ValueError("Mesh fingerprint differs from provenance")
        for upstream in source["upstream_paths"]:
            path = PurePosixPath(upstream)
            if (not path.is_relative_to("robots/g1_description/meshes")
                    or ".." in path.parts or path.suffix.lower() != ".stl"):
                raise ValueError("Unexpected upstream mesh path")
        if not source["upstream_paths"]:
            raise ValueError("Missing upstream mesh path")
    for name, scene in manifest["scenes"].items():
        if Path(name).name != name or not name.endswith(".xml"):
            raise ValueError("Scene must be a plain XML filename")
        checked_file(bundle, name, scene["packaged"])
    return manifest, provenance


def install(bundle):
    manifest, provenance = read_contract(bundle)
    destination = bundle / "assets"
    if destination.is_symlink():
        raise ValueError("Asset directory cannot be a symlink")
    if destination.exists():
        for name, expected in manifest["assets"].items():
            checked_file(bundle, name, expected)
        return len(manifest["assets"])
    # Publish the directory only after every download passes both content checks.
    with tempfile.TemporaryDirectory(prefix=".mesh-install-", dir=bundle) as temporary:
        staging = Path(temporary) / "assets"
        staging.mkdir()
        for name, expected in manifest["assets"].items():
            source = provenance["assets"][name]
            upstream = source["upstream_paths"][0]
            url = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{REVISION}/{upstream}"
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read(expected["size_bytes"] + 1)
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if digest(raw) != expected or blob != source["git_blob"]:
                raise ValueError(f"Downloaded mesh fingerprint mismatch: {name}")
            (staging / Path(name).name).write_bytes(raw)
        staging.rename(destination)
    return len(manifest["assets"])


def export(source, destination):
    manifest, _ = read_contract(source)
    destination.mkdir(parents=True, exist_ok=False)
    for name in (*manifest["scenes"], "manifest.json", "unitree_assets.json",
                 "UNITREE_LICENSE", "UNILAB_LICENSE"):
        shutil.copyfile(source / name, destination / name)
    shutil.copyfile(__file__, destination / "install_meshes.py")
    return len(manifest["scenes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--export-from", type=Path)
    args = parser.parse_args()
    if args.export_from:
        print(f"Exported {export(args.export_from, args.bundle)} scenes without meshes.")
    else:
        print(f"Verified {install(args.bundle)} installed meshes; controllers are not included.")
