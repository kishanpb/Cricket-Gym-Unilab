"""One-command attenuation keeps the parent trajectory contract and full gates."""

import copy

import numpy as np
import pytest
from probe_g1_cricket_reachability import action_at as parent_action
from probe_g1_cricket_terminal_residual import CANDIDATE, PLAN, action_at, compare_scale


@pytest.mark.parametrize("scale", PLAN["scales"])
def test_only_tick_19_changes(scale):
    for tick in range(100):
        action = action_at(scale, tick)
        parent = parent_action(CANDIDATE, tick)
        assert action.dtype == np.float32 and action.shape == (1, 7)
        if tick == 19 and scale != 1:
            expected = (
                np.asarray(CANDIDATE["signs"], dtype=np.float32)
                * np.float32(-0.95)
                * np.float32(scale)
            )
            np.testing.assert_allclose(np.tanh(action[0]), expected, rtol=0, atol=1e-7)
        else:
            np.testing.assert_array_equal(action, parent)


def rows():
    outcome = dict(
        hand="right",
        controller="test",
        offset_m=0.0,
        seed=4301,
        seconds=2,
        passed=True,
        failures=[],
        blade_contact_seen=True,
        maximum_blade_penetration_m=0.005,
        first_separation_ball_vx_m_s=1.2,
        ball_contact_peak_force_norm_n={"bat_blade": 250},
    )
    return [
        dict(
            scale=0.5,
            engine=engine,
            sim_dt=dt,
            outcome=copy.deepcopy(outcome),
            terminated=False,
            truncated=True,
            blade_episode_count=1,
            first_impact={},
            first_loaded_impact={},
        )
        for engine in PLAN["executors"]
        for dt in PLAN["physics_dt_seconds"]
    ]


def test_requires_complete_unique_pairing():
    data = rows()
    assert compare_scale(data, 0.5)["scripted_witness"]
    for malformed in (data[:-1], data + data[:1], data[:3] + data[:1]):
        with pytest.raises(ValueError, match="four unique"):
            compare_scale(malformed, 0.5)


@pytest.mark.parametrize(
    "failure", ["physical_gate", "incomplete", "divergent_impulse", "resolution"]
)
def test_no_single_passing_row_can_make_a_witness(failure):
    data = rows()
    if failure == "physical_gate":
        for r in data:
            if r["sim_dt"] == 0.000125:
                r["outcome"].update(passed=False, failures=["blade_penetration_above_6_mm"])
    elif failure == "incomplete":
        for r in data:
            r["truncated"] = False
    elif failure == "divergent_impulse":
        data[-1]["first_impact"] = {"impulse_on_ball_world_ns": [1, 0, 0]}
    else:
        for r in data:
            if r["sim_dt"] == 0.000125:
                r["outcome"]["maximum_blade_penetration_m"] = 0.0055
    assert not compare_scale(data, 0.5)["scripted_witness"]


def test_fixed_full_trial_budget():
    assert PLAN["scales"] == [0, 0.25, 0.5, 0.75, 0.875, 1]
    assert (
        len(PLAN["scales"]) * len(PLAN["executors"]) * len(PLAN["physics_dt_seconds"])
        == PLAN["full_trial_count"]
        == 24
    )
    assert PLAN["ticks"] == 100
