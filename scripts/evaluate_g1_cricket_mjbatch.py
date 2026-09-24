"""Compare native mjbatch execution against every retained tanh-v1 outcome."""

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import mjbatch
import mjbatch._bindings as native
import mjbatch.held_control as held
from evaluate_g1_cricket_impact_events_learning import apply_penetration_gate, validate_pool
from evaluate_g1_cricket_residual import ROOT, evaluate, sha256
from evaluate_g1_cricket_swing import check_hashes
from g1_cricket_archival_sources import check_retained_hashes

DIRECTORY = ROOT / "g1_cricket_results/native_mjbatch_v1"
PARENT = ROOT / "g1_cricket_results/tanh_v1"


def executor_manifest():
    return {
        "version": importlib.metadata.version("mjbatch"),
        "python_sha256": sha256(Path(held.__file__)),
        "batch_wrapper_sha256": sha256(Path(mjbatch.__file__)),
        "native_extension_sha256": sha256(Path(native.__file__)),
    }


def preflight(mjbatch_root):
    if DIRECTORY.exists():
        raise FileExistsError("native mjbatch experiment already has retained evidence")
    parent = json.loads((PARENT / "trained_evaluation.json").read_text())
    check_retained_hashes(ROOT, parent["input_sha256"])
    sources = {name: sha256(ROOT / name) for name in parent["reports"][0]["source_sha256"]}
    for name in (
        "scripts/evaluate_g1_cricket_mjbatch.py",
        "scripts/g1_cricket_archival_sources.py",
        "src/unilab/conf/ppo/task/g1_cricket_tanh_v1/mjbatch.yaml",
        "docs/g1_cricket_mjbatch_v1.md",
        "docs/sphinx/source/adr/ADR-0011-experimental-mjbatch-recorder.md",
    ):
        sources[name] = sha256(ROOT / name)
    inputs = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            PARENT / "trained_evaluation.json",
            PARENT / "right/run_config.json",
            PARENT / "right/run_summary.json",
            PARENT / "right/model_2079.pt",
        )
    }
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=mjbatch_root, text=True
    ).strip()
    for name, key in (
        ("src/mjbatch/held_control.py", "python_sha256"),
        ("src/mjbatch/__init__.py", "batch_wrapper_sha256"),
    ):
        source = mjbatch_root / name
        assert sha256(source) == executor_manifest()[key]
        assert (
            subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=mjbatch_root)
            == source.read_bytes()
        ), "installed executor must match a committed companion source"
    result = {
        "scope": "native_mjbatch_frozen_policy_execution_preflight",
        "source_sha256": sources,
        "input_sha256": inputs,
        "executor": executor_manifest(),
        "mjbatch_source_revision": revision,
        "new_training": False,
        "policy_promoted": False,
    }
    DIRECTORY.mkdir(parents=True)
    (DIRECTORY / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")


def run():
    path = DIRECTORY / "preflight.json"
    output = DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError("native mjbatch evaluation is already retained")
    contract = json.loads(path.read_text())
    pinned = {**contract["source_sha256"], **contract["input_sha256"]}
    pinned[str(path.relative_to(ROOT))] = sha256(path)
    check_hashes(pinned)
    assert executor_manifest() == contract["executor"]
    parent = json.loads((PARENT / "trained_evaluation.json").read_text())
    reports = []
    for index, dt in enumerate((0.00025, 0.000125)):
        report = evaluate(
            PARENT / "right", {"env": {"sim_dt": dt, "mujoco_substep_engine": "mjbatch"}}
        )
        apply_penetration_gate(report)
        validate_pool(report["rows"])
        assert report["rows"] == parent["reports"][index]["rows"], (
            "executor changed a retained outcome"
        )
        for key in ("checkpoint", "versions", "external_asset_sha256", "contact_models"):
            assert report[key] == parent["reports"][index][key]
        report["source_sha256"].update(contract["source_sha256"])
        report["scope"] = "native_mjbatch_frozen_tanh_policy_execution_not_promoted"
        reports.append(report)
        print(
            json.dumps(
                {"dt": dt, "exact_rows": len(report["rows"]), "aggregates": report["aggregates"]}
            ),
            flush=True,
        )
    check_hashes(pinned)
    assert executor_manifest() == contract["executor"]
    result = {
        "scope": "native_mjbatch_complete_frozen_policy_execution_parity",
        "input_sha256": pinned,
        "executor": contract["executor"],
        "mjbatch_source_revision": contract["mjbatch_source_revision"],
        "reports": reports,
        "all_192_rows_exact": True,
        "new_training": False,
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mjbatch-root", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.mjbatch_root is None:
            parser.error("--preflight requires --mjbatch-root")
        preflight(args.mjbatch_root)
    else:
        run()
