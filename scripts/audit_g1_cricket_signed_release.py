"""Replay every retained release with an additive signed hinge-extension audit."""

import json

import numpy as np
from evaluate_g1_cricket_residual import sha256
from g1_cricket_signed_delivery import SignedDeliveryReplay
from probe_g1_cricket_overarm_guard import owner_config
from probe_g1_cricket_overarm_release import (
    DIRECTORY,
    PLAN,
    release_trial,
)
from probe_g1_cricket_overarm_release import (
    inputs as release_inputs,
)
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

OUTPUT = DIRECTORY / "signed_elbow_audit.json"


def inputs():
    record = release_inputs()
    for name in (
        "scripts/audit_g1_cricket_signed_release.py",
        "scripts/g1_cricket_signed_delivery.py",
        "tests/scripts/test_g1_cricket_signed_delivery.py",
        "tests/scripts/test_g1_cricket_bowling_reward_v2.py",
        "docs/g1_cricket_signed_release_v1.md",
        "src/unilab/conf/ppo/task/g1_cricket_overarm_reward_v2/mujoco.yaml",
        "src/unilab/conf/ppo/task/g1_cricket_overarm_reward_v2/mjbatch.yaml",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record["parent_release_evaluation_sha256"] = sha256(DIRECTORY / "evaluation.json")
    return record


def run():
    if OUTPUT.exists():
        raise FileExistsError("inspect retained signed audit instead of restarting")
    record = inputs()
    parents = json.loads((DIRECTORY / "evaluation.json").read_text())["rows"]
    rows = []
    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:
        for parent in parents:
            replays = []

            def factory(env):
                replay = SignedDeliveryReplay(env)
                replays.append(replay)
                return replay

            actual = release_trial(env, parent["candidate"], replay_factory=factory)
            if actual != parent:
                raise ValueError("signed audit changed a legacy result or motion trace")
            outcome = actual["outcome"]
            release = outcome["release"]
            height = np.clip((release["position"][2] - 0.9) / 0.4, 0, 1)
            speed = np.clip(release["velocity"][0], 0, 12) / 6
            lateral = np.exp(-np.square(release["velocity"][1] / 2))
            bonus = float(10 * height * speed * lateral)
            row = dict(
                candidate=parent["candidate"],
                signed_audit=replays[0].audit.result(outcome),
                exact_legacy_outcome_and_full_trace=True,
                release_reward_audit=dict(
                    release_ball_x_m=release["position"][0],
                    legacy_integrated_release_bonus=bonus * (release["position"][0] < 0),
                    v2_counterfactual_integrated_release_bonus=bonus,
                    scope="release_component_only_not_recomputed_return_or_new_learning",
                ),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    if record != inputs():
        raise ValueError("signed replay audit inputs changed")
    write_json(OUTPUT, dict(inputs=record, rows=rows, new_training=False, policy_promoted=False))


if __name__ == "__main__":
    run()
