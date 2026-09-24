"""Fixed absolute-reference reach study; no release, learning or policy promotion."""

import itertools
import json
from importlib.metadata import version

import mujoco
import numpy as np
from audit_g1_cricket_pitch_contact import source_hashes as contact_sources
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay, arm_geometry
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, VERSIONS, make_env, runtime_sources

from unilab.tasks.manipulation.g1_cricket.overarm import actions_for_targets

DIRECTORY = ROOT / "g1_cricket_results/overarm_v1"
PLAN = dict(
    hands=["right", "left"],
    pitches=[-2.45, -2.80],
    elbows=[1.10, 1.40],
    outward_roll=0.35,
    seed=6301,
    dt=0.0000625,
    engine="mjbatch",
    rows=8,
    new_training=False,
    policy_promoted=False,
)


def owner_config(engine="mjbatch"):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_overarm_v1/{engine}"])


def smooth(tick, begin, end):
    t = np.clip((tick - begin) / (end - begin), 0, 1)
    return t * t * (3 - 2 * t)


def target_at(neutral, hand, pitch, elbow, tick):
    raised = np.array([pitch, 0.35 if hand == "left" else -0.35, 0, elbow, 0, 0, 0])
    settled = raised.copy()
    settled[0] = neutral[0]
    target = neutral + smooth(tick, 10, 30) * (settled - neutral)
    target[0] += smooth(tick, 30, 80) * (pitch - neutral[0])
    return target + smooth(tick, 120, 170) * (neutral - target)


def reach_sample(upper, ball_above_shoulder, elbow_error):
    return upper >= 0.75 and ball_above_shoulder >= 0.25 and elbow_error <= 0.1


def cold_pose(replay, targets, hand):
    m, d = replay.model, replay.pose
    mujoco.mj_resetDataKeyframe(m, d, 0)
    arm_ids = np.arange(22, 29) if hand == "right" else np.arange(15, 22)
    d.qpos[replay.joint_qadr[arm_ids]] = targets
    wrist = m.body(f"{hand}_wrist_yaw_link").id
    mujoco.mj_kinematics(m, d)
    offset = [0.15, 0.06 if hand == "left" else -0.06, 0]
    # Offline FK only; these writes never reach the running environment.
    d.qpos[replay.ball_qadr : replay.ball_qadr + 3] = (
        d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ offset
    )
    d.qpos[replay.ball_qadr + 3 : replay.ball_qadr + 7] = d.xquat[wrist]
    mujoco.mj_forward(m, d)
    upper, angle = arm_geometry(*d.xpos[replay.arm])
    collisions = [
        [m.geom(int(g)).name for g in contact.geom]
        for contact in d.contact
        if contact.dist < 0 and replay.pitch not in contact.geom
    ]
    return dict(
        upper_z=upper,
        elbow_angle_rad=angle,
        ball_above_shoulder_m=float(d.qpos[replay.ball_qadr + 2] - d.xpos[replay.arm[0], 2]),
        non_ground_overlaps=collisions,
        dynamic_evidence=False,
    )


def reach_trial(env, pitch, elbow, *, observer=None):
    env.reset(seed=PLAN["seed"])
    term = env.action_manager.get_term("residual")
    neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids].copy()
    limits = term.joint_limits[term.arm_ids]
    replay = DeliveryReplay(env)
    proposed = target_at(neutral, env.cfg.handedness, pitch, elbow, 80)
    actions_for_targets(proposed, neutral, limits)
    cold = cold_pose(replay, proposed, env.cfg.handedness)
    events = DeliveryEvents(env.cfg.handedness)
    trace, consecutive, longest = [], 0, 0
    total = 0.0
    for tick in range(env.max_episode_length):
        target = target_at(neutral, env.cfg.handedness, pitch, elbow, tick)
        action = np.zeros((1, 8), dtype=np.float32)
        action[0, :7] = actions_for_targets(target, neutral, limits)
        state = replay.step(
            env,
            action,
            events,
            observer=None if observer is None else lambda m, d: observer(tick, m, d),
        )
        m, pose = replay.model, replay.pose
        snapshot = env.get_physics_state_snapshot()[0]
        mujoco.mj_setState(m, pose, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_kinematics(m, pose)
        upper, angle = arm_geometry(*pose.xpos[replay.arm])
        achieved = pose.qpos[replay.joint_qadr[term.arm_ids]]
        height = float(pose.qpos[replay.ball_qadr + 2] - pose.xpos[replay.arm[0], 2])
        ready = 80 <= tick < 120 and reach_sample(upper, height, abs(achieved[3] - elbow))
        consecutive = consecutive + 1 if ready else 0
        longest = max(longest, consecutive)
        trace.append(
            dict(
                tick=tick,
                upper_z=upper,
                elbow_angle_rad=angle,
                ball_above_shoulder_m=height,
                target=term.processed_action[0, term.arm_ids].tolist(),
                achieved=achieved.tolist(),
                sampled_reach=bool(ready),
            )
        )
        total += float(state.reward[0])
        if state.terminated[0] or state.truncated[0]:
            break
    complete = bool(state.truncated[0] and not state.terminated[0])
    outcome = events.finish(complete)
    safe = (
        complete
        and not events.failures
        and events.height >= 0.48
        and events.up >= 0.65
        and events.limit_excess <= 1e-6
        and events.force_fraction <= 1 + 1e-6
        and events.peak_penetration <= 0.006
    )
    return dict(
        hand=env.cfg.handedness,
        pitch=pitch,
        elbow=elbow,
        cold_pose=cold,
        trace=trace,
        longest_sampled_reach_ticks=longest,
        reach_ready=bool(safe and longest >= 4),
        outcome=outcome,
        steps=tick + 1,
        episode_return=total,
        exact_endpoint_replay=True,
    )


def sources():
    result = contact_sources()
    paths = [
        ROOT / "scripts/probe_g1_cricket_overarm.py",
        ROOT / "tests/scripts/test_g1_cricket_overarm.py",
        ROOT / "docs/g1_cricket_overarm_v1.md",
        *(ROOT / "src/unilab/conf/ppo/task/g1_cricket_overarm_v1").glob("*.yaml"),
    ]
    result.update({str(p.relative_to(ROOT)): sha256(p) for p in paths})
    return result


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained overarm study instead of restarting")
    inputs = dict(
        plan=PLAN,
        config=OmegaConf.to_container(owner_config(), resolve=True),
        versions={name: version(name) for name in VERSIONS},
        sources=sources(),
        runtime_sources=runtime_sources(),
        executor=executor_manifest(),
    )
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", inputs)
    rows = []
    for hand in PLAN["hands"]:
        env = make_env(owner_config(), hand, dt=PLAN["dt"], engine=PLAN["engine"])
        try:
            for pitch, elbow in itertools.product(PLAN["pitches"], PLAN["elbows"]):
                row = reach_trial(env, pitch, elbow)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
        finally:
            env.close()
    if inputs["sources"] != sources() or inputs["runtime_sources"] != runtime_sources():
        raise ValueError("overarm study inputs changed")
    if inputs["executor"] != executor_manifest():
        raise ValueError("overarm study executor changed")
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
