"""Scripted screening cannot weaken full-shot or first-separation semantics."""

import copy
import itertools
import json

import numpy as np
import pytest
from g1_cricket_trial import BallEvents, ImpactReplay, finish_gate
from probe_g1_cricket_reachability import (
    PARENT,
    PLAN,
    action_at,
    execute,
    make_env,
    prefix_safe,
    rank,
    schedule,
)


def contact(name="bat_blade", depth=0.003):
    return {"geom": name, "distance_m": -depth, "force_norm_n": 12.0}


def gate(ball, **overrides):
    args = dict(complete=True, guarded=False, height=0.7, up=1, excess=0, fraction=0.9)
    return finish_gate(ball, **(args | overrides))


def clean_ball():
    ball = BallEvents()
    ball.observe([contact()], 0.1, -2)
    ball.observe([], 0.101, 1.1)
    return ball


def test_fixed_schedule_family_and_tanh_mapping():
    candidates = [schedule(signs) for signs in itertools.product((-1, 1), repeat=7)]
    assert len({c["id"] for c in candidates}) == 128
    assert 1 + 128 + 8 * 4 == PLAN["maximum_prefixes"]
    candidate = schedule([-1, 1, -1, 1, -1, 1, -1])
    for tick in range(100):
        expected = np.array(candidate["signs"], dtype=np.float32) * np.float32(0.95)
        if 10 <= tick < 20:
            expected *= -1
        elif tick >= 20:
            expected[:] = 0
        raw = action_at(candidate, tick)
        assert raw.shape == (1, 7) and raw.dtype == np.float32
        np.testing.assert_allclose(np.tanh(raw[0]), expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_simultaneous_first_pitch_and_blade_fails(reverse):
    ball = BallEvents()
    records = [contact(), contact("pitch")]
    ball.observe(records[::-1] if reverse else records, 0.1, -2)
    ball.observe([], 0.101, 2)
    failures, _ = gate(ball)
    assert "first_contact_not_blade_only" in failures
    assert "ball_contact:pitch" in failures


@pytest.mark.parametrize("geom", ["bat_handle", "cricket_stump_middle", "torso_link"])
def test_later_forbidden_contact_still_fails(geom):
    ball = clean_ball()
    ball.observe([contact(geom)], 0.8, 4)
    failures, _ = gate(ball)
    assert f"ball_contact:{geom}" in failures
    assert ball.separation == 1.1


def test_recontact_cannot_replace_first_exit_or_hide_penetration():
    ball = clean_ball()
    ball.observe([contact(depth=0.007)], 0.8, 3)
    ball.observe([], 0.801, 4)
    failures, original = gate(ball)
    assert original
    assert failures == ["blade_penetration_above_6_mm"]
    assert ball.penetration == 0.007 and ball.separation == 1.1
    assert sum(e["event"] == "first_blade_separation" for e in ball.events) == 1


def test_contact_without_separation_and_strict_velocity_gate():
    ball = BallEvents()
    ball.observe([contact()], 0.1, 3)
    assert "outgoing_velocity_not_above_1_m_s" in gate(ball)[0]
    ball.observe([], 0.101, 1)
    assert "outgoing_velocity_not_above_1_m_s" in gate(ball)[0]


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"complete": False}, "native_episode_incomplete"),
        ({"guarded": True}, "guarded_contact"),
        ({"height": 0.479}, "pelvis_height"),
        ({"up": 0.649}, "pelvis_orientation"),
        ({"excess": 2e-6}, "joint_limit"),
        ({"fraction": 1.000002}, "actuator_limit"),
    ],
)
def test_full_physical_gates(overrides, failure):
    assert gate(clean_ball(), **overrides) == ([failure], False)


def prefix(identifier, vx=0.5):
    return {
        "candidate": {"id": identifier},
        "terminated": False,
        "truncated": False,
        "outcome": {
            "seconds": 0.5,
            "failures": ["native_episode_incomplete", "outgoing_velocity_not_above_1_m_s"],
            "first_separation_ball_vx_m_s": vx,
            "passed": False,
        },
    }


def test_ranking_preserves_failed_prefix_status_and_tie_order():
    rows = [prefix("b"), prefix("zero_residual", 8), prefix("a"), prefix("c", 2)]
    before = copy.deepcopy(rows)
    assert [r["candidate"]["id"] for r in rank(rows)] == ["c", "a", "b"]
    assert [r["candidate"]["id"] for r in rank(rows, require_forward=True)] == ["c"]
    assert rows == before
    assert rank([]) == []


@pytest.mark.parametrize(
    "failure", ["guarded_contact", "ball_contact:pitch", "blade_penetration_above_6_mm"]
)
def test_unsafe_prefix_not_selected(failure):
    result = prefix("a", 3)
    result["outcome"]["failures"].append(failure)
    assert not prefix_safe(result) and not rank([result])


@pytest.mark.parametrize("field", ["terminated", "truncated"])
def test_early_native_end_cannot_qualify(field):
    result = prefix("a", 3)
    result[field] = True
    assert not prefix_safe(result)


def test_missing_exit_and_wrong_prefix_length_rejected():
    result = prefix("a", None)
    assert not prefix_safe(result)
    result = prefix("a", 3)
    result["outcome"]["seconds"] = 0.48
    assert not prefix_safe(result)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_complete_zero_row_exactly_matches_retained_parent(hand, dt):
    parent = json.loads(PARENT.read_text())["reports"][int(dt == 0.000125)]
    reference = next(
        row
        for row in parent["rows"]
        if row["hand"] == hand
        and row["controller"] == "zero_residual"
        and row["offset_m"] == 0
        and row["seed"] == 4301
    )
    env = make_env(hand, dt, "mujoco")
    try:
        result = execute(
            env,
            ImpactReplay(env),
            dict(id="zero_residual", signs=[0] * 7, switch_tick=10),
            100,
            hand,
        )
        assert result["outcome"] == reference
        assert result["truncated"] and not result["terminated"]
        assert result["first_impact"] is not None
        assert (
            result["first_impact"]["end"]["post_ball_velocity_world_m_s"][0]
            == reference["first_separation_ball_vx_m_s"]
        )
    finally:
        env.close()
