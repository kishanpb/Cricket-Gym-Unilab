"""Fixed drive/release study from retained left-hand preloads; not learned motion."""

import itertools
import json
from copy import deepcopy

import mujoco
import numpy as np
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay, arm_geometry
from probe_g1_cricket_overarm import smooth, target_at
from probe_g1_cricket_overarm_guard import DIRECTORY as PARENT
from probe_g1_cricket_overarm_guard import GuardAudit, owner_config
from probe_g1_cricket_overarm_guard import inputs as guard_inputs
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

from unilab.tasks.manipulation.g1_cricket.overarm import actions_for_targets

DIRECTORY = ROOT / "g1_cricket_results/overarm_release_v1"
PLAN = dict(
    hand="left",
    parents=[[-2.80, 1.40]],
    parent_selection="near_straight_elbow_and_overhead_before_baseline_legal_stride_window",
    release_delays=[4, 8, 12],
    drive_tick=110,
    drive_pitches=[0.3, 1.0],
    recovery_begin=150,
    recovery_end=190,
    seed=6301,
    dt=0.0000625,
    engine="mjbatch",
    rows=6,
    new_training=False,
    policy_promoted=False,
)


def candidates():
    return [
        dict(pitch=pose[0], elbow=pose[1], drive_pitch=drive_pitch, delay=delay)
        for pose, drive_pitch, delay in itertools.product(
            PLAN["parents"], PLAN["drive_pitches"], PLAN["release_delays"]
        )
    ]


def target_and_release(neutral, candidate, tick):
    target = target_at(neutral, PLAN["hand"], candidate["pitch"], candidate["elbow"], tick)
    if tick >= PLAN["drive_tick"]:
        target = np.array([candidate["drive_pitch"], 0.35, 0, candidate["elbow"], 0, 0, 0])
        target += smooth(tick, PLAN["recovery_begin"], PLAN["recovery_end"]) * (neutral - target)
    return target, tick >= PLAN["drive_tick"] + candidate["delay"]


def release_attribution(result):
    result = deepcopy(result)
    result["first_contact_excluding_foot_pitch"] = result.pop("first_forbidden_contact")
    events = [
        result[name]
        for name in ("first_joint", "worst_joint", "first_contact_excluding_foot_pitch")
    ]
    for event in events + result["context"]:
        if event is None:
            continue
        for end, phase in (
            (10, "neutral"),
            (30, "settle"),
            (80, "raise"),
            (PLAN["drive_tick"], "hold"),
            (PLAN["recovery_begin"], "drive"),
            (PLAN["recovery_end"], "recover"),
            (200, "final_settle"),
        ):
            if event["tick"] < end:
                event["phase"] = phase
                break
    result["contact_scope"] = (
        "first_contact_excluding_foot_pitch_not_a_violation_gate_use_full_outcome"
    )
    return result


def release_trial(env, candidate):
    env.reset(seed=PLAN["seed"])
    term = env.action_manager.get_term("residual")
    neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids].copy()
    limits = term.joint_limits[term.arm_ids]
    replay, events, audit = DeliveryReplay(env), DeliveryEvents(PLAN["hand"]), GuardAudit(env)
    m, pose = replay.model, replay.pose
    arm_qadr = replay.joint_qadr[term.arm_ids]
    arm_vadr = m.jnt_dofadr[replay.joints[term.arm_ids]]
    trace, total = [], 0.0
    for tick in range(env.max_episode_length):
        target, release = target_and_release(neutral, candidate, tick)
        action = np.zeros((1, 8), np.float32)
        action[0, :7] = actions_for_targets(target, neutral, limits)
        action[0, 7] = release
        state = replay.step(env, action, events, observer=lambda m, d: audit(tick, m, d))
        snapshot = env.get_physics_state_snapshot()[0]
        mujoco.mj_setState(m, pose, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_kinematics(m, pose)
        upper, angle = arm_geometry(*pose.xpos[replay.arm])
        trace.append(
            dict(
                tick=tick,
                time=float(snapshot[0]),
                target=term.processed_action[0, term.arm_ids].tolist(),
                achieved=pose.qpos[arm_qadr].tolist(),
                joint_velocity=pose.qvel[arm_vadr].tolist(),
                last_substep_torque_nm=replay.data.actuator_force[term.arm_ids].tolist(),
                upper_z=upper,
                elbow_angle_rad=angle,
                ball_position=pose.qpos[replay.ball_qadr : replay.ball_qadr + 3].tolist(),
                ball_velocity=pose.qvel[replay.ball_vadr : replay.ball_vadr + 3].tolist(),
                ball_above_shoulder_m=float(
                    pose.qpos[replay.ball_qadr + 2] - pose.xpos[replay.arm[0], 2]
                ),
                released=bool(term.released[0]),
            )
        )
        total += float(state.reward[0])
        if state.terminated[0] or state.truncated[0]:
            break
    outcome = events.finish(bool(state.truncated[0] and not state.terminated[0]))
    return dict(
        hand=PLAN["hand"],
        candidate=candidate,
        outcome=outcome,
        steps=tick + 1,
        episode_return=total,
        trace=trace,
        landings_before_release=events.landings,
        attribution=release_attribution(audit.result()),
        exact_endpoint_state_and_sensor_replay=True,
    )


def inputs():
    result = guard_inputs()
    for name in (
        "scripts/probe_g1_cricket_overarm_release.py",
        "tests/scripts/test_g1_cricket_overarm_release.py",
        "docs/g1_cricket_overarm_release_v1.md",
    ):
        result["sources"][name] = sha256(ROOT / name)
    result.update(
        plan=PLAN,
        parent_preflight_sha256=sha256(PARENT / "preflight.json"),
        parent_evaluation_sha256=sha256(PARENT / "evaluation.json"),
    )
    return result


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained release study instead of restarting")
    parents = json.loads((PARENT / "evaluation.json").read_text())["rows"]
    ready = [
        [r["pitch"], r["elbow"]] for r in parents if r["hand"] == PLAN["hand"] and r["reach_ready"]
    ]
    if not all(parent in ready for parent in PLAN["parents"]):
        raise ValueError("release parents must be retained preload witnesses")
    record = inputs()
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)
    rows = []
    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:
        for candidate in candidates():
            row = release_trial(env, candidate)
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"trace", "attribution"}}),
                flush=True,
            )
    finally:
        env.close()
    if record != inputs():
        raise ValueError("release study inputs changed")
    assert len(rows) == PLAN["rows"]
    write_json(
        DIRECTORY / "evaluation.json",
        dict(
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            rows=rows,
            full_gate_passes=sum(r["outcome"]["passed"] for r in rows),
            scope="scripted_development_drive_release_not_learning_or_showcase",
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    run()
