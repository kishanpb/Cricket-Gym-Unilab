from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from probe_g1_cricket_overarm import target_at
from probe_g1_cricket_overarm_guard import owner_config
from search_g1_cricket_shoulder import (
    PLAN,
    initial_population,
    optimizer_summary,
    search_cost,
    shoulder_target,
)
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.overarm import absolute_targets, actions_for_targets


def test_seeded_budget_and_original_target_bounds():
    population = initial_population()
    np.testing.assert_array_equal(population, initial_population())
    assert population.shape == (8, 6)
    assert PLAN["population"] * (PLAN["generations"] + 1) == PLAN["maximum_trials"] == 32
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    try:
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        corners = np.asarray(PLAN["bounds"]).T
        for parameters in np.vstack((population, corners)):
            for tick in range(200):
                target = shoulder_target(neutral, parameters, tick)
                action = actions_for_targets(target, neutral, limits)
                np.testing.assert_allclose(
                    absolute_targets(action, neutral, limits), target, atol=1e-7
                )
                if tick < 98:
                    np.testing.assert_array_equal(
                        target, target_at(neutral, "left", -2.8, 1.4, tick)
                    )
                if 98 <= tick <= 150:
                    np.testing.assert_array_equal(target[3:], [1.4, 0, 0, 0])
                if tick >= 190:
                    np.testing.assert_allclose(target, neutral, atol=1e-15)
    finally:
        env.close()


def test_knot_endpoints_and_pre_release_action_indexing():
    neutral = np.array([0.2, 0.2, 0, 1.28, 0, 0, 0])
    parameters = initial_population()[0]
    np.testing.assert_allclose(shoulder_target(neutral, parameters, 106)[:3], parameters[:3])
    np.testing.assert_allclose(shoulder_target(neutral, parameters, 114)[:3], parameters[3:])
    assert shoulder_target(neutral, parameters, 113)[0] < parameters[3]
    assert PLAN["release_tick"] == 114


def test_score_never_overrides_full_gate_or_safety():
    outcome = dict(
        release=dict(velocity=[10, 0, 0]),
        first_bounce=dict(position=[10, 0, 0]),
        target_crossing=dict(position=[17.68, 0, 0.5]),
    )
    trace = [dict(ball_position=[17.68, 0, 0.5], released=True)]
    passed = search_cost(outcome, dict(failures=[]), trace)
    failure = search_cost(outcome, dict(failures=["target_corridor_missed"]), trace)
    unsafe = search_cost(outcome, dict(failures=["joint_limit"]), trace)
    assert passed["cost"] < failure["cost"] < unsafe["cost"]
    poor = deepcopy(outcome)
    poor["release"] = poor["first_bounce"] = poor["target_crossing"] = None
    score = search_cost(
        poor, dict(failures=["no_release"]), [dict(ball_position=[100, 0, 1], released=False)]
    )
    assert score["bounded_continuous_cost"] == 7
    assert score["cost"] < unsafe["cost"]
    assert score["furthest_sampled_ball_position"] is None
    assert outcome["release"]["velocity"] == [10, 0, 0]


def test_tied_optimizer_minima_keep_index_and_parameters_from_same_row():
    rows = [dict(index=i, parameters=[i], search_score=dict(cost=1.0)) for i in range(2)]
    result = SimpleNamespace(nfev=2, nit=0, message="test", success=True, x=np.array([1]), fun=1.0)
    summary = optimizer_summary(rows, result)
    assert summary["best_index"] == 0
    assert summary["parameters"] == rows[0]["parameters"]
    assert summary["cost"] == rows[0]["search_score"]["cost"]
