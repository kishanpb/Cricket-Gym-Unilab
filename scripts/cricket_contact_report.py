"""Fixed reference-control contact diagnostics for the optional cricket tasks.

This is instrumentation validation, not a learned-policy benchmark.
"""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from cricket_gym.integrations.unilab import CricketMotorSwingCfg, CricketMotorSwingEnv
from cricket_gym.integrations.unilab_bowling import (
    CricketDeliveryStrideCfg,
    CricketDeliveryStrideUniLabEnv,
)
from humanoid_cricket import contact_telemetry


def collect():
    rows = []
    for task in ("batting", "bowling"):
        for hand in ("right", "left"):
            if task == "batting":
                cfg = CricketMotorSwingCfg(base_seed=17000, handedness=hand, contact_telemetry=True)
                env = CricketMotorSwingEnv(cfg, num_envs=2)
                actions = np.zeros((2, 7), dtype=np.float32)
            else:
                cfg = CricketDeliveryStrideCfg(
                    base_seed=4101, handedness=hand, contact_telemetry=True
                )
                env = CricketDeliveryStrideUniLabEnv(cfg, num_envs=2)
                actions = np.array([[0, -0.15, 0.23], [0, -0.15, 0]], dtype=np.float32)
            try:
                state = env.init_state()
                finished = np.zeros(2, dtype=bool)
                returns = np.zeros(2)
                for _ in range(200):
                    state = env.step(actions)
                    for i in range(2):
                        if finished[i]:
                            continue
                        returns[i] += float(state.reward[i])
                        if state.terminated[i] or state.truncated[i]:
                            rows.append(
                                {
                                    "task": task,
                                    "handedness": hand,
                                    "seed": cfg.base_seed + i,
                                    "action": actions[i].tolist(),
                                    "return": float(returns[i]),
                                    "observation_size": state.obs["obs"].shape[1],
                                    "contact_telemetry": state.info["contact_telemetry"][i],
                                }
                            )
                            finished[i] = True
                    if finished.all():
                        break
                if not finished.all():
                    raise RuntimeError("unfinished cricket diagnostic episode")
            finally:
                env.close()
    return {
        "scope": "Fixed reference controls; no training or policy-ranking claim",
        "backend": "UniLab external Gymnasium adapter / serial MuJoCo CPU",
        "runtime": {name: version(name) for name in ("unilab", "mujoco", "cricket-gym")},
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            "humanoid_cricket/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(contact_telemetry.__file__).parent.glob("*.py"))
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(report['rows'])} complete reference-control episodes: {args.output}")


if __name__ == "__main__":
    main()
