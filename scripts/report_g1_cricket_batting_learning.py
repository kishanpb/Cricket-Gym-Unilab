"""Retain both final PPO policies and every nominal-feed evaluation outcome."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import mjbatch.held_control
from report_g1_cricket_bimanual_contact import compare_resolution
from report_g1_cricket_bounced_delivery import summarize_bounced_row

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "ball_contact_observed_ppo_nominal_bounced_delivery_not_held_out"


def validate_training(summary, config, diagnostics):
    expected = {
        "status": "completed",
        "run_env_steps": 49152,
        "total_env_steps": 49152,
        "completed_iterations": 255,
        "global_num_envs": 8,
        "effective_seed": 1,
        "task": "G1CricketBimanualLearning",
        "algo": "ppo",
        "sim_backend": "mujoco",
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("training differs from the complete declared pilot")
    if config["algo"]["resume"] or config["algo"]["algorithm"]["disable_finite_checks"]:
        raise ValueError("pilot requires fresh actors and enabled finite checks")
    action = config["env"]["actions"]["reference"]
    if any(
        action[key] != value
        for key, value in {
            "scale": 0.05,
            "lookahead_frames": 0,
            "balance_gain": 4,
            "waist_tracking_gain": 1,
            "root_position_gain": 4,
        }.items()
    ):
        raise ValueError("controller differs from the learning protocol")
    if config["env"]["sim_dt"] != 0.00003125:
        raise ValueError("training requires the declared contact timestep")
    for group in ("actor", "critic"):
        terms = config["env"]["observations"][group]["terms"]
        if not {"ball", "contact", "bat_error"} <= terms.keys():
            raise ValueError("pilot requires ball, contact and bat-error observations")
    if (
        config["reward"]["bat_tracking"]["weight"] != 2
        or config["reward"]["bat_tracking"]["params"]["std"] != 0.08
    ):
        raise ValueError("bat tracking reward differs from the declared task")
    if not diagnostics["scalar_tags"] or not all(
        tag["all_finite"] for tag in diagnostics["scalar_tags"].values()
    ):
        raise ValueError("training scalars are missing or non-finite")


def build_report(directory):
    training, rows, hashes = {}, [], {}
    for hand in ("right", "left"):
        run = directory / f"ppo_{hand}"
        summary = json.loads((run / "run_summary.json").read_text())
        config = json.loads((run / "run_config.json").read_text())["config"]
        diagnostics = json.loads((run / "training_diagnostics.json").read_text())
        validate_training(summary, config, diagnostics)
        with (run / "training_scalars.csv").open(newline="") as stream:
            scalars = list(csv.DictReader(stream))
        if [int(row["iteration"]) for row in scalars] != list(range(256)):
            raise ValueError("all 256 training iterations must be retained")
        if any(
            not math.isfinite(float(value)) for row in scalars for value in row.values() if value
        ):
            raise ValueError("training scalar CSV is non-finite")
        checkpoint = run / "model_255.pt"
        if Path(summary["last_checkpoint"]).resolve() != checkpoint.resolve():
            raise ValueError("evaluation must use the declared final checkpoint")
        if config["env"]["handedness"] != hand:
            raise ValueError("training hand differs from declared case")
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        training[hand] = {**summary, "checkpoint_sha256": checkpoint_hash}
        for name in (
            "run_summary.json",
            "run_config.json",
            "training_diagnostics.json",
            "training_scalars.csv",
            "model_255.pt",
        ):
            path = run / name
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for resolution, dt in (("fine", 0.00003125), ("finest", 0.000015625)):
            path = directory / f"{hand}_{resolution}" / "evaluation.json"
            report = json.loads(path.read_text())
            if report["scope"] != SCOPE or report["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("evaluation is not the new task's final checkpoint")
            if report["evaluation_overrides"] != {
                "contact_dt": dt,
                "soft_toss": False,
                "bounced_delivery": True,
                "compact_substeps": True,
            }:
                raise ValueError("evaluation differs from the declared nominal-feed protocol")
            for source, digest in report["input_sha256"].items():
                if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"evaluation input changed: {source}")
            runtime_hash = hashlib.sha256(
                Path(mjbatch.held_control.__file__).read_bytes()
            ).hexdigest()
            if report["runtime_source_sha256"] != {"mjbatch.held_control": runtime_hash}:
                raise ValueError("recorder source changed")
            if [row["controller"] for row in report["rows"]] != ["reference_only", "ppo"]:
                raise ValueError("both reference and PPO controls are required")
            hashes[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in report["rows"]:
                if row["hand"] != hand:
                    raise ValueError("evaluation hand differs from declared case")
                rows.append({"resolution": resolution, **summarize_bounced_row(row)})
    pairs = []
    qualification = {}
    for hand in ("right", "left"):
        for controller in ("reference_only", "ppo"):
            matched = [
                row for row in rows if row["hand"] == hand and row["controller"] == controller
            ]
            pair = {"hand": hand, "controller": controller, **compare_resolution(*matched)}
            pairs.append(pair)
            if controller == "ppo":
                qualification[hand] = (
                    all(row["all_checks_pass"] for row in matched) and pair["all_pass"]
                )
    return {
        "scope": SCOPE,
        "training": training,
        "rows": rows,
        "resolution_comparisons": pairs,
        "trained_policy_qualification": qualification,
        "all_trained_policies_qualify": all(qualification.values()),
        "input_sha256": hashes,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = build_report(args.directory)
    (args.directory / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "rows": len(report["rows"]),
                "trained_policy_qualification": report["trained_policy_qualification"],
            }
        )
    )
