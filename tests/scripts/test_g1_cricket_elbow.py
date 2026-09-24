"""Elbow targets and measured motor traces cannot bypass the physical shot gate."""

import numpy as np
import pytest
from g1_cricket_motor_replay import motor_summary
from probe_g1_cricket_elbow import PLAN, action_at, compare_candidate
from probe_g1_cricket_reachability import action_at as parent_action
from probe_g1_cricket_terminal_residual import CANDIDATE


@pytest.mark.parametrize("scale", PLAN["scales"])
def test_changes_only_forward_phase_elbow(scale):
    for tick in range(100):
        action, parent = action_at(scale, tick), parent_action(CANDIDATE, tick)
        assert action.dtype == np.float32 and action.shape == (1, 7)
        np.testing.assert_array_equal(action[:, [0, 1, 2, 4, 5, 6]], parent[:, [0, 1, 2, 4, 5, 6]])
        if 10 <= tick < 20 and scale != 1:
            np.testing.assert_allclose(np.tanh(action[0, 3]), -0.95 * scale, rtol=0, atol=1e-7)
        else:
            np.testing.assert_array_equal(action, parent)


def test_fixed_trial_contract():
    assert PLAN["scales"] == [1, 0.75, 0.5, 0]
    assert PLAN["changed_ticks"] == list(range(10, 20))
    assert PLAN["arm_channel"] == 3 and PLAN["joint"] == "right_elbow_joint"
    assert PLAN["motor_limits_nm"] == [-25, 25]
    assert PLAN["ticks"] == 100
    assert (
        len(PLAN["scales"]) * len(PLAN["physics_dt_seconds"]) * len(PLAN["executors"])
        == PLAN["full_trial_count"]
        == 16
    )


@pytest.mark.parametrize("limits", [[-25, 25], [-5, 5]])
def test_named_motor_limits_and_signed_counts(limits):
    requested = [limits[0] - 2, limits[0], 0, limits[1], limits[1] + 2]
    applied = np.clip(requested, *limits)
    result = motor_summary(10, 0.2, requested, applied, [0.2, 0.3], [-1, 2], limits)
    assert result["negative_saturated_steps"] == result["positive_saturated_steps"] == 2
    assert result["physics_steps"] == 5 and result["maximum_clipped_torque_error_nm"] == 0
    assert result["applied_torque_range_nm"] == limits
    with pytest.raises(AssertionError):
        motor_summary(10, 0.2, requested, -applied, [0.2, 0.3], [-1, 2], limits)


@pytest.mark.parametrize("failure", [None, "mismatch", "missing", "substeps"])
def test_motor_trace_is_part_of_witness(failure):
    data = [
        dict(
            scale=0.5,
            engine=engine,
            sim_dt=dt,
            terminated=False,
            truncated=True,
            blade_episode_count=1,
            first_impact={},
            first_loaded_impact={},
            outcome=dict(
                hand="right",
                controller="test",
                offset_m=0,
                seed=4301,
                seconds=2,
                passed=True,
                failures=[],
                blade_contact_seen=True,
                maximum_blade_penetration_m=0.005,
                first_separation_ball_vx_m_s=1.2,
                ball_contact_peak_force_norm_n={"bat_blade": 250},
            ),
        )
        for engine in PLAN["executors"]
        for dt in PLAN["physics_dt_seconds"]
    ]
    for row in data:
        row["motor_trace"] = [
            {"tick": t, "physics_steps": round(0.02 / row["sim_dt"])} for t in range(100)
        ]
    if failure == "mismatch":
        data[-1]["motor_trace"][0]["target_rad"] = 0.3
    elif failure == "missing":
        for row in data:
            row["motor_trace"].pop()
    elif failure == "substeps":
        for row in data:
            row["motor_trace"][0]["physics_steps"] -= 1
    assert compare_candidate(data, 0.5)["scripted_witness"] == (failure is None)
