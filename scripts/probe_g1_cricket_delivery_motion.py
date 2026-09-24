"""Bounded physical overarm demonstration search, never a learned-policy claim."""

import argparse
import itertools
import json

import numpy as np
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryReplay, trial
from omegaconf import OmegaConf
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import DIRECTORY as PARENT
from train_g1_cricket_delivery import ROOT, contract, make_env

from unilab.tasks.manipulation.g1_cricket.bowling import ARM_LIMITS

DIRECTORY = ROOT / "g1_cricket_results/delivery_motion_v1"
PLAN = dict(
    hands=["right", "left"],
    seed=6301,
    dt=0.00025,
    engine="mjbatch",
    settle_tick=10,
    raised_tick=45,
    swing_tick=55,
    recovery_tick=65,
    recovery_end_tick=95,
    windup_pitch_offset=-2.45,
    outward_roll_offset=0.15,
    elbow_offsets=[0.35, 0.6],
    swing_pitch_offsets=[1.0, 2.3],
    release_delays=[2, 3, 4, 5],
    rows=32,
    new_training=False,
    policy_promoted=False,
)


def candidates():
    return [
        dict(id=f"e{elbow}_p{pitch}_d{delay}", elbow=elbow, pitch=pitch, delay=delay)
        for elbow, pitch, delay in itertools.product(
            PLAN["elbow_offsets"], PLAN["swing_pitch_offsets"], PLAN["release_delays"]
        )
    ]


def action_at(candidate, hand, tick):
    raised = np.array(
        [
            PLAN["windup_pitch_offset"],
            PLAN["outward_roll_offset"] * (1 if hand == "left" else -1),
            0,
            candidate["elbow"],
            0,
            0,
            0,
        ],
        dtype=np.float32,
    )
    swing = raised.copy()
    swing[0] = candidate["pitch"]
    if tick < PLAN["swing_tick"]:
        fraction = np.clip(
            (tick - PLAN["settle_tick"]) / (PLAN["raised_tick"] - PLAN["settle_tick"]), 0, 1
        )
        offset = raised * (fraction * fraction * (3 - 2 * fraction))
    else:
        recovery = np.clip(
            (tick - PLAN["recovery_tick"]) / (PLAN["recovery_end_tick"] - PLAN["recovery_tick"]),
            0,
            1,
        )
        offset = swing * (1 - recovery * recovery * (3 - 2 * recovery))
    action = np.zeros((1, 8), dtype=np.float32)
    action[0, :7] = np.arctanh(offset / ARM_LIMITS)
    action[0, 7] = float(tick >= PLAN["swing_tick"] + candidate["delay"])
    return action


def preflight():
    contract()
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained motion experiment instead of restarting")
    names = [
        "scripts/probe_g1_cricket_delivery_motion.py",
        "tests/scripts/test_g1_cricket_delivery_motion.py",
        "docs/g1_cricket_delivery_motion_v1.md",
        str((PARENT / "preflight.json").relative_to(ROOT)),
        str((PARENT / "evaluation.json").relative_to(ROOT)),
    ]
    record = dict(plan=PLAN, input_sha256={n: sha256(ROOT / n) for n in names})
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)


def check():
    parent = contract()
    record = json.loads((DIRECTORY / "preflight.json").read_text())
    if record["plan"] != PLAN:
        raise ValueError("motion plan changed")
    for name, expected in record["input_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"motion input changed: {name}")
    return parent


def run():
    parent = check()
    output = DIRECTORY / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for hand in PLAN["hands"]:
        env = make_env(OmegaConf.create(parent["config"]), hand)
        try:
            replay = DeliveryReplay(env)
            for candidate in candidates():
                outcome = trial(
                    env, replay, lambda tick: action_at(candidate, hand, tick), PLAN["seed"]
                )
                row = dict(hand=hand, candidate=candidate, outcome=outcome)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    assert len(rows) == PLAN["rows"]
    check()
    write_json(
        output,
        dict(
            plan=PLAN,
            scope="scripted_motor_search_not_learned_or_qualified",
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            rows=rows,
            full_gate_passes=sum(r["outcome"]["passed"] for r in rows),
            qualification="requires_additional_seeds_and_paired_timestep_executor_checks",
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    args = parser.parse_args()
    preflight() if args.stage == "preflight" else run()
