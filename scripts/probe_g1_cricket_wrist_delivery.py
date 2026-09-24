"""Fixed wrist-posture/release-window study, not a learned bowling policy."""

import itertools
import json

from evaluate_g1_cricket_residual import sha256
from probe_g1_cricket_overarm import smooth
from probe_g1_cricket_overarm_damping import DIRECTORY as PARENT
from probe_g1_cricket_overarm_damping import inputs as parent_inputs
from probe_g1_cricket_overarm_guard import GuardAudit
from probe_g1_cricket_overarm_release import release_attribution, target_and_release
from probe_g1_cricket_shoulder_damping import owner_config
from search_g1_cricket_shoulder import trajectory_trial
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

DIRECTORY = ROOT / "g1_cricket_results/wrist_delivery_v1"
PLAN = dict(
    hand="left",
    seed=6301,
    dt=0.0000625,
    engine="mjbatch",
    drive_pitch=1.0,
    drive_tick=110,
    wrist_pitches=[0.0, -0.8, -1.2],
    release_ticks=[114, 115, 116],
    recovery=[150, 190],
    rows=9,
    new_training=False,
    policy_promoted=False,
)


def candidates():
    return list(itertools.product(PLAN["wrist_pitches"], PLAN["release_ticks"]))


def wrist_target(neutral, candidate, tick):
    wrist, release = candidate
    parent = dict(pitch=-2.8, elbow=1.4, drive_pitch=PLAN["drive_pitch"], delay=release - 110)
    target, _ = target_and_release(neutral, parent, tick)
    target[5] += smooth(tick, 10, 30) * wrist * (1 - smooth(tick, *PLAN["recovery"]))
    return target


def verify_baseline(row):
    parent = json.loads((PARENT / "evaluation.json").read_text())
    if parent["status"] != "completed" or len(parent["rows"]) != 6:
        raise ValueError("neutral-wrist comparison requires the completed six-case parent")
    baseline = parent["rows"][3]
    if baseline["candidate"] != dict(pitch=-2.8, elbow=1.4, drive_pitch=1.0, delay=4):
        raise ValueError("neutral-wrist parent candidate differs")
    for key in ("outcome", "signed_audit", "steps", "episode_return"):
        if row[key] != baseline[key]:
            raise ValueError(f"neutral-wrist baseline differs: {key}")
    for actual, expected in zip(row["trace"], baseline["trace"], strict=True):
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("neutral-wrist baseline trace differs")


def inputs():
    record = parent_inputs()
    record.pop("changed_axis")
    for name in (
        "scripts/probe_g1_cricket_wrist_delivery.py",
        "tests/scripts/test_g1_cricket_wrist_delivery.py",
        "docs/g1_cricket_wrist_delivery_v1.md",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record.update(
        plan=PLAN,
        changed_axes=["wrist_pitch_preload", "release_tick"],
        candidate_selection="all_nine_wrist_posture_by_release_tick_combinations",
        parent_overarm_damping_sha256=sha256(PARENT / "evaluation.json"),
    )
    return record


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained wrist delivery study instead of restarting")
    record = inputs()
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
                scope="nine_scripted_left_wrist_release_development_cases_not_learning",
                policy_promoted=False,
            ),
        )

    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:
        for candidate in candidates():
            audits = []

            def factory(env):
                audit = GuardAudit(env)
                audits.append(audit)
                return audit

            row = trajectory_trial(
                env,
                candidate,
                target_fn=wrist_target,
                seed=PLAN["seed"],
                release_tick=candidate[1],
                audit_factory=factory,
            )
            row["attribution"] = release_attribution(audits[0].result())
            if candidate == (0.0, 114):
                verify_baseline(row)
            rows.append(row)
            save("running")
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"trace", "attribution"}}),
                flush=True,
            )
    finally:
        env.close()
    if record != inputs():
        raise ValueError("wrist delivery inputs changed")
    assert len(rows) == PLAN["rows"] == 9
    save("completed")


if __name__ == "__main__":
    run()
