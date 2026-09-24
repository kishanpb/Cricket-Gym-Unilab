"""Use the frozen full shot gate and baseline check for residual reward v3."""

import argparse
import json
from pathlib import Path

from evaluate_g1_cricket_residual_v2 import ROOT, evaluate_v2, sha256


def evaluate_v3(run_dir):
    report = evaluate_v2(run_dir)
    report["scope"] = "nonnegative_separation_residual_v3_same_development_pool_not_full_cricket"
    report["predecessor_v2_report_sha256"] = sha256(
        ROOT / "g1_cricket_results/residual_v2/evaluation.json"
    )
    for name in (
        "scripts/evaluate_g1_cricket_residual_v3.py",
        "src/unilab/tasks/manipulation/g1_cricket/forward_exit_reward.py",
        "src/unilab/conf/ppo/task/g1_cricket_residual_v3/mujoco.yaml",
        "docs/g1_cricket_residual_v3.md",
    ):
        report["source_sha256"][name] = sha256(ROOT / name)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "g1_cricket_results/residual_v3/right"
    )
    args = parser.parse_args()
    report = evaluate_v3(args.run_dir)
    (args.run_dir.parent / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report["aggregates"], indent=2))
