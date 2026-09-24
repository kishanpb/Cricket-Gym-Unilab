"""Apply the unchanged complete-pool v1 shot gate to the reward-only v2 run."""

import argparse
import json
from pathlib import Path

from evaluate_g1_cricket_residual import ROOT, evaluate
from evaluate_g1_cricket_smoke import sha256


def evaluate_v2(run_dir):
    report = evaluate(run_dir)
    parent_path = ROOT / "g1_cricket_results/residual_v1/evaluation.json"
    parent = json.loads(parent_path.read_text())
    for row, old in zip(report["rows"], parent["rows"], strict=True):
        for key in ("hand", "controller", "offset_m", "seed"):
            if row[key] != old[key]:
                raise ValueError(f"evaluation pool changed: {key}")
        if row["controller"] == "zero_residual":
            if {k: v for k, v in row.items() if k != "return"} != {
                k: v for k, v in old.items() if k != "return"
            }:
                raise ValueError("reward-only change altered baseline physics or shot gate")
    report["scope"] = (
        "reward_only_residual_v2_same_development_pool_not_generalization_or_full_cricket"
    )
    report["parent_report_sha256"] = sha256(parent_path)
    for name in (
        "scripts/evaluate_g1_cricket_residual_v2.py",
        "src/unilab/tasks/manipulation/g1_cricket/separation_reward.py",
        "src/unilab/conf/ppo/task/g1_cricket_residual_v2/mujoco.yaml",
        "docs/g1_cricket_residual_v2.md",
    ):
        report["source_sha256"][name] = sha256(ROOT / name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "g1_cricket_results/residual_v2/right"
    )
    args = parser.parse_args()
    report = evaluate_v2(args.run_dir)
    (args.run_dir.parent / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["aggregates"], indent=2))
