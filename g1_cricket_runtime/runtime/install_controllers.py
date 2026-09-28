"""Fetch pinned GR00T/AMP controller inputs without cloning unrelated assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import urllib.request


GROOT_POLICY = "decoupled_wbc/sim2mujoco/resources/robots/g1/policy/"
CONTROLLERS = {
    "groot": dict(
        repository="NVlabs/GR00T-WholeBodyControl",
        revision="b042411fae38ee4d1af9aac82a37a1f8d14d6dd0",
        license="LICENSE",
        checkpoints={
            GROOT_POLICY + "GR00T-WholeBodyControl-Balance.onnx":
                "f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18",
            GROOT_POLICY + "GR00T-WholeBodyControl-Walk.onnx":
                "7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407",
        },
    ),
    "amp": dict(
        repository="Jiarui-Xie/AMP_Running_baseline",
        revision="4a4dff791ec9ba376a39894d60832754124247de",
        license="LICENCE",
        checkpoints={"checkpoints/model_6200.pt":
                     "5de7628fe92bd74118016c592b95eb17183547a5289708622c80a7815da025ac"},
    ),
}


def fingerprint(raw):
    return dict(size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def source_files(kind, manifest):
    spec = CONTROLLERS[kind]
    expected = {} if kind == "amp" else manifest["upstream_files"]
    for name, sha256 in spec["checkpoints"].items():
        if kind == "groot" and expected[name]["sha256"] != sha256:
            raise ValueError("Runtime requests an unreviewed controller checkpoint")
    names = sorted(set(expected) | set(spec["checkpoints"]) | {spec["license"]})
    for name in names:
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or path.parts[0] == ".git"
                or path.as_posix() != name or any(c in name for c in "\n\r*?[]\\")):
            raise ValueError("Controller source must be a literal relative path")
    return names, expected


def lfs_payload(repository, revision, name, pointer, expected):
    fields = dict(line.split(" ", 1) for line in pointer.decode("ascii").splitlines())
    if (fields.get("version") != "https://git-lfs.github.com/spec/v1"
            or fields.get("oid") != f"sha256:{expected}"):
        raise ValueError("Checkpoint pointer differs from reviewed hash")
    size = int(fields["size"])
    url = f"https://media.githubusercontent.com/media/{repository}/{revision}/{name}"
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read(size + 1)
    if fingerprint(raw) != dict(size_bytes=size, sha256=expected):
        raise ValueError("Downloaded checkpoint differs from its pinned LFS pointer")
    return raw


def install(kind, destination, *, manifest=None):
    spec = CONTROLLERS[kind]
    names, expected = source_files(kind, manifest)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Controller destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryDirectory(prefix=".controller-install-", dir=destination.parent) as tmp:
        checkout = Path(tmp) / "checkout"

        def git(*args, **kwargs):
            return subprocess.run(["git", "-C", str(checkout), *args], env=env,
                                  text=True, check=True, stdout=subprocess.PIPE, **kwargs)

        checkout.mkdir()
        git("init", "--quiet")
        # Download pinned LFS payloads ourselves, without requiring git-lfs.
        for key, value in (("process", ""), ("smudge", "cat"), ("clean", "cat"), ("required", "false")):
            git("config", f"filter.lfs.{key}", value)
        git("remote", "add", "origin", f"https://github.com/{spec['repository']}.git")
        git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", spec["revision"])
        git("sparse-checkout", "set", "--no-cone", "--stdin",
            input="".join(f"/{name}\n" for name in names))
        git("checkout", "--quiet", "--detach", spec["revision"])
        if git("rev-parse", "HEAD").stdout.strip() != spec["revision"]:
            raise ValueError("Controller checkout revision differs")
        for name, expected_hash in spec["checkpoints"].items():
            path = checkout / name
            raw = lfs_payload(spec["repository"], spec["revision"], name,
                              path.read_bytes(), expected_hash)
            path.write_bytes(raw)
        for name, digest in expected.items():
            if fingerprint((checkout / name).read_bytes()) != digest:
                raise ValueError(f"Controller runtime source differs: {name}")
        report = dict(kind=kind, repository=spec["repository"], revision=spec["revision"],
                      files={name: fingerprint((checkout / name).read_bytes()) for name in names},
                      scope="Verified source/checkpoint installation, not policy performance")
        (checkout / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
        checkout.rename(destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=CONTROLLERS)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    if args.kind == "groot" and args.runtime_manifest is None:
        parser.error("GR00T requires --runtime-manifest")
    manifest = None if args.runtime_manifest is None else json.loads(args.runtime_manifest.read_text())
    report = install(args.kind, args.destination, manifest=manifest)
    print(f"Installed {report['kind']} at its pinned revision; all {len(report['files'])} files verified.")
