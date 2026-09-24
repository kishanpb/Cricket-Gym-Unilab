"""Bounded development trajectory optimization; not a learned bowling policy."""

import json
from importlib.metadata import version

import mujoco
import numpy as np
from audit_g1_cricket_signed_release import inputs as signed_inputs
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryEvents, arm_geometry
from g1_cricket_signed_delivery import SignedDeliveryReplay
from probe_g1_cricket_overarm import smooth, target_at
from probe_g1_cricket_overarm_guard import GuardAudit, owner_config
from scipy.optimize import differential_evolution
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

from unilab.tasks.manipulation.g1_cricket.overarm import actions_for_targets

DIRECTORY = ROOT / "g1_cricket_results/shoulder_search_v1"
PLAN = dict(
    hand="left",
    seed=6301,
    optimizer_seed=7351,
    dt=0.0000625,
    engine="mjbatch",
    knots=[98, 106, 114],
    release_tick=114,
    recovery=[150, 190],
    bounds=[[-2.85, -2.45], [0.10, 0.55], [-0.45, 0.45], [-0.3, 1.3], [-0.05, 0.55], [-0.6, 0.6]],
    population=8,
    generations=3,
    maximum_trials=32,
    strategy="best1bin",
    mutation=[0.5, 1.0],
    recombination=0.7,
    new_rl_training=False,
    policy_promoted=False,
)
SAFETY_FAILURES = {
    "episode_incomplete",
    "pelvis_height",
    "pelvis_orientation",
    "joint_limit",
    "actuator_limit",
    "ball_penetration_above_6_mm",
    "nonfoot_ground_contact",
    "robot_self_or_wicket_contact",
}


def shoulder_target(neutral, parameters, tick):
    start, middle, end = PLAN["knots"]
    if tick < start:
        return target_at(neutral, PLAN["hand"], -2.8, 1.4, tick)
    preload = np.array([-2.8, 0.35, 0, 1.4, 0, 0, 0])
    first, second = preload.copy(), preload.copy()
    first[:3], second[:3] = np.asarray(parameters).reshape(2, 3)
    if tick < middle:
        return preload + smooth(tick, start, middle) * (first - preload)
    target = first + smooth(tick, middle, end) * (second - first)
    return target + smooth(tick, *PLAN["recovery"]) * (neutral - target)


def initial_population():
    bounds = np.asarray(PLAN["bounds"])
    rng = np.random.default_rng(PLAN["optimizer_seed"])
    population = rng.uniform(bounds[:, 0], bounds[:, 1], (PLAN["population"], len(bounds)))
    population[0] = [-2.8, 0.35, 0, 1.0, 0.35, 0]
    return population


def search_cost(outcome, signed, trace):
    failures = set(signed["failures"])
    safety = sorted(f for f in failures if f in SAFETY_FAILURES or f.startswith("ball_contact:"))
    release, bounce, crossing = (
        outcome["release"],
        outcome["first_bounce"],
        outcome["target_crossing"],
    )
    positions = np.array([row["ball_position"] for row in trace if row["released"]])
    furthest = positions[np.argmax(positions[:, 0])] if len(positions) else None
    terms = dict(
        forward_speed=1.0 if release is None else max(0, (10 - release["velocity"][0]) / 10),
        lateral_speed=1.0 if release is None else abs(release["velocity"][1]) / 2,
        downward_speed=1.0 if release is None else max(0, -release["velocity"][2]) / 6,
        bounce_length=1.0 if bounce is None else abs(bounce["position"][0] - 10) / 10,
        bounce_lateral=1.0 if bounce is None else abs(bounce["position"][1]) / 1.32,
        target_shortfall=1.0 if furthest is None else max(0, (17.68 - furthest[0]) / 17.68),
        flight_lateral=1.0
        if furthest is None
        else abs((furthest if crossing is None else crossing["position"])[1]) / 1.32,
    )
    continuous = sum(min(1.0, value) for value in terms.values())
    # Failure tiers dominate the bounded tie-breaker; only the independent gate qualifies.
    cost = 10000 * bool(safety) + 1000 * bool(failures) + 20 * len(failures) + continuous
    return dict(
        cost=float(cost),
        safety_failures=safety,
        normalized_raw_terms={k: float(v) for k, v in terms.items()},
        bounded_continuous_cost=float(continuous),
        furthest_sampled_ball_position=None if furthest is None else furthest.tolist(),
        scope="development_search_score_not_a_promotion_metric",
    )


def trajectory_trial(env, parameters):
    env.reset(seed=PLAN["seed"])
    term = env.action_manager.get_term("residual")
    neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids].copy()
    limits = term.joint_limits[term.arm_ids]
    replay, events, audit = SignedDeliveryReplay(env), DeliveryEvents(PLAN["hand"]), GuardAudit(env)
    m, pose = replay.model, replay.pose
    qadr = replay.joint_qadr[term.arm_ids]
    vadr = m.jnt_dofadr[replay.joints[term.arm_ids]]
    trace, total = [], 0.0
    for tick in range(env.max_episode_length):
        target = shoulder_target(neutral, parameters, tick)
        action = np.zeros((1, 8), np.float32)
        action[0, :7] = actions_for_targets(target, neutral, limits)
        action[0, 7] = tick >= PLAN["release_tick"]
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
                achieved=pose.qpos[qadr].tolist(),
                joint_velocity=pose.qvel[vadr].tolist(),
                last_substep_torque_nm=replay.data.actuator_force[term.arm_ids].tolist(),
                upper_z=upper,
                elbow_angle_rad=angle,
                ball_position=pose.qpos[replay.ball_qadr : replay.ball_qadr + 3].tolist(),
                ball_velocity=pose.qvel[replay.ball_vadr : replay.ball_vadr + 3].tolist(),
                released=bool(term.released[0]),
            )
        )
        total += float(state.reward[0])
        if state.terminated[0] or state.truncated[0]:
            break
    outcome = events.finish(bool(state.truncated[0] and not state.terminated[0]))
    signed = replay.audit.result(outcome)
    return dict(
        parameters=np.asarray(parameters).tolist(),
        outcome=outcome,
        signed_audit=signed,
        search_score=search_cost(outcome, signed, trace),
        steps=tick + 1,
        episode_return=total,
        trace=trace,
        prior_target_guard=dict(
            clipped_control_ticks=audit.clip_ticks,
            maximum_target_correction_rad=audit.maximum_correction,
        ),
        exact_endpoint_state_and_sensor_replay=True,
    )


def inputs():
    record = signed_inputs()
    for name in (
        "scripts/search_g1_cricket_shoulder.py",
        "tests/scripts/test_g1_cricket_shoulder_search.py",
        "docs/g1_cricket_shoulder_search_v1.md",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record.update(
        plan=PLAN, scipy_version=version("scipy"), initial_population=initial_population().tolist()
    )
    return record


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained shoulder search instead of restarting")
    record = inputs()
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)
    rows = []

    def save(status, **extra):
        write_json(
            DIRECTORY / "evaluation.json",
            dict(
                preflight_sha256=sha256(DIRECTORY / "preflight.json"),
                status=status,
                rows=rows,
                full_signed_gate_passes=sum(r["signed_audit"]["passed"] for r in rows),
                scope="single_left_development_context_optimized_references_not_learned_policy",
                policy_promoted=False,
                **extra,
            ),
        )

    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:

        def evaluate(parameters):
            row = trajectory_trial(env, parameters)
            row["index"] = len(rows)
            rows.append(row)
            save("running")
            print(json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
            return row["search_score"]["cost"]

        result = differential_evolution(
            evaluate,
            PLAN["bounds"],
            strategy=PLAN["strategy"],
            maxiter=PLAN["generations"],
            init=initial_population(),
            mutation=tuple(PLAN["mutation"]),
            recombination=PLAN["recombination"],
            rng=np.random.default_rng(PLAN["optimizer_seed"]),
            polish=False,
            tol=0,
            atol=0,
            updating="immediate",
            workers=1,
        )
    finally:
        env.close()
    if record != inputs():
        raise ValueError("shoulder search inputs changed")
    assert result.nfev == len(rows) <= PLAN["maximum_trials"]
    best = min(rows, key=lambda row: row["search_score"]["cost"])
    save(
        "completed",
        optimizer=dict(
            nfev=result.nfev,
            nit=result.nit,
            message=result.message,
            converged=bool(result.success),
            best_index=best["index"],
            parameters=result.x.tolist(),
            cost=float(result.fun),
        ),
    )


if __name__ == "__main__":
    run()
