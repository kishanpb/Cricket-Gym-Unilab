"""Same eight reaches with bounded prior motor targets; no new learning."""

import json
from importlib.metadata import version

import numpy as np
from audit_g1_cricket_overarm_failures import FailureAudit, audit_inputs
from evaluate_g1_cricket_residual import sha256
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from probe_g1_cricket_overarm import DIRECTORY as PARENT
from probe_g1_cricket_overarm import PLAN, reach_trial
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, VERSIONS, make_env

from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK

DIRECTORY = ROOT / "g1_cricket_results/overarm_guard_v1"


def owner_config(engine="mjbatch"):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_overarm_guard_v1/{engine}"])


class GuardAudit(FailureAudit):
    def __init__(self, env):
        super().__init__(env)
        self.term = env.action_manager.get_term("residual")
        self.neutral = env.scene["robot"].data.default_joint_pos[0].copy()
        self.clip_ticks = 0
        self.clip_counts = np.zeros(29, dtype=int)
        self.maximum_correction = 0.0
        self.names = [env.get_playback_model().joint(j).name for j in self.joints]

    def __call__(self, tick, m, d):
        if tick not in self.context:
            sdk = np.empty(29, np.float32)
            sdk[POLICY_TO_SDK] = self.term.baseline_action[0]
            raw = self.neutral + 0.25 * sdk
            ids, limits = self.term.prior_ids, self.term.prior_target_limits
            expected = np.clip(raw[ids], limits[:, 0], limits[:, 1])
            np.testing.assert_array_equal(
                d.ctrl[ids].astype(np.float32), expected.astype(np.float32)
            )
            clipped = (raw[ids] < limits[:, 0]) | (raw[ids] > limits[:, 1])
            self.clip_ticks += int(clipped.any())
            self.clip_counts[ids] += clipped
            self.maximum_correction = max(
                self.maximum_correction, float(np.abs(raw[ids] - expected).max())
            )
        super().__call__(tick, m, d)

    def result(self):
        return dict(
            **super().result(),
            clipped_control_ticks=self.clip_ticks,
            clipped_joint_control_counts={
                name: int(n) for name, n in zip(self.names, self.clip_counts, strict=True) if n
            },
            maximum_target_correction_rad=self.maximum_correction,
        )


def inputs():
    result = audit_inputs()
    for path in (
        ROOT / "scripts/probe_g1_cricket_overarm_guard.py",
        ROOT / "tests/scripts/test_g1_cricket_overarm_guard.py",
        ROOT / "docs/g1_cricket_overarm_guard_v1.md",
        *(ROOT / "src/unilab/conf/ppo/task/g1_cricket_overarm_guard_v1").glob("*.yaml"),
    ):
        result["sources"][str(path.relative_to(ROOT))] = sha256(path)
    result.update(
        parent_attribution_sha256=sha256(PARENT / "failure_attribution.json"),
        versions={name: version(name) for name in VERSIONS},
        config=OmegaConf.to_container(owner_config(), resolve=True),
        plan=dict(PLAN, changed_axis="prior_target_limit_fraction_0.9"),
    )
    return result


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained guarded comparison instead of restarting")
    record = inputs()
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)
    parents = json.loads((PARENT / "evaluation.json").read_text())["rows"]
    rows = []
    for parent in parents:
        env = make_env(owner_config(), parent["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
        try:
            audit = GuardAudit(env)
            row = reach_trial(env, parent["pitch"], parent["elbow"], observer=audit)
            row["attribution"] = audit.result()
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"trace", "attribution"}}),
                flush=True,
            )
        finally:
            env.close()
    if record != inputs():
        raise ValueError("guarded reach inputs changed")
    assert len(rows) == PLAN["rows"]
    write_json(
        DIRECTORY / "evaluation.json",
        dict(
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            rows=rows,
            reach_witnesses=sum(r["reach_ready"] for r in rows),
            scope="scripted_sampled_reach_not_delivery_or_learning",
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    run()
