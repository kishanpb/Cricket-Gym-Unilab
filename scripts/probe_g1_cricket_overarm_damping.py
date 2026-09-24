"""Paired controller comparison on all retained overhead drive/release schedules."""

import json

from audit_g1_cricket_signed_release import inputs as signed_inputs
from evaluate_g1_cricket_residual import sha256
from g1_cricket_signed_delivery import SignedDeliveryReplay
from omegaconf import OmegaConf
from probe_g1_cricket_overarm_release import DIRECTORY as PARENT
from probe_g1_cricket_overarm_release import PLAN, candidates, release_trial
from probe_g1_cricket_shoulder_damping import owner_config
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

DIRECTORY = ROOT / "g1_cricket_results/overarm_damping_v1"


def inputs():
    record = signed_inputs()
    for name in (
        "scripts/probe_g1_cricket_overarm_damping.py",
        "tests/scripts/test_g1_cricket_overarm_damping.py",
        "docs/g1_cricket_overarm_damping_v1.md",
        "scripts/probe_g1_cricket_shoulder_damping.py",
        "scripts/probe_g1_cricket_positive_arc.py",
        "scripts/search_g1_cricket_shoulder.py",
        "src/unilab/tasks/manipulation/g1_cricket/impedance.py",
        "src/unilab/tasks/__init__.py",
        "src/unilab/conf/ppo/task/g1_cricket_shoulder_damping_v1/mujoco.yaml",
        "src/unilab/conf/ppo/task/g1_cricket_shoulder_damping_v1/mjbatch.yaml",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record.update(
        config=OmegaConf.to_container(owner_config(), resolve=True),
        changed_axis="selected_shoulder_pitch_controller_kd_10_to_2",
        candidate_selection="all_six_retained_overarm_release_schedules_in_original_order",
        checkpoint_scope="distinct_action_config_strict_contract_no_old_checkpoint_reuse",
    )
    return record


def paired_candidates():
    retained = json.loads((PARENT / "evaluation.json").read_text())["rows"]
    planned = candidates()
    if [row["candidate"] for row in retained] != planned:
        raise ValueError("paired study requires all six original release schedules")
    return planned


def signed_trial(env, candidate):
    replays = []

    def factory(env):
        replay = SignedDeliveryReplay(env)
        replays.append(replay)
        return replay

    row = release_trial(env, candidate, replay_factory=factory)
    row["signed_audit"] = replays[0].audit.result(row["outcome"])
    return row


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained overarm damping study instead of restarting")
    planned, record = paired_candidates(), inputs()
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)
    rows = []

    def save(status):
        write_json(
            DIRECTORY / "evaluation.json",
            dict(
                preflight_sha256=sha256(DIRECTORY / "preflight.json"),
                status=status,
                rows=rows,
                full_signed_gate_passes=sum(r["signed_audit"]["passed"] for r in rows),
                scope="six_paired_scripted_left_overarm_cases_not_learning",
                policy_promoted=False,
            ),
        )

    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:
        for candidate in planned:
            row = signed_trial(env, candidate)
            rows.append(row)
            save("running")
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"trace", "attribution"}}),
                flush=True,
            )
    finally:
        env.close()
    if record != inputs():
        raise ValueError("overarm damping inputs changed")
    assert len(rows) == PLAN["rows"] == 6
    save("completed")


if __name__ == "__main__":
    run()
