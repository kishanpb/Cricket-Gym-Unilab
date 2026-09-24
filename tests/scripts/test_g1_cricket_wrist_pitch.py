"""Wrist shaping changes one command channel, never the task or shot gate."""

import numpy as np
import pytest
from probe_g1_cricket_reachability import action_at as parent_action
from probe_g1_cricket_terminal_residual import CANDIDATE
from probe_g1_cricket_wrist_pitch import PLAN, action_at, impact_direction


@pytest.mark.parametrize("scale", PLAN["scales"])
def test_only_forward_phase_wrist_pitch_changes(scale):
    for tick in range(100):
        action = action_at(scale, tick)
        parent = parent_action(CANDIDATE, tick)
        assert action.shape == (1, 7) and action.dtype == np.float32
        np.testing.assert_array_equal(action[:, [0, 1, 2, 3, 4, 6]], parent[:, [0, 1, 2, 3, 4, 6]])
        if 10 <= tick < 20 and scale != 1:
            np.testing.assert_allclose(np.tanh(action[0, 5]), -0.95 * scale, rtol=0, atol=1e-7)
        else:
            np.testing.assert_array_equal(action, parent)
        assert np.isfinite(action).all() and (np.abs(np.tanh(action)) <= 0.950001).all()


def test_fixed_all_context_budget():
    assert PLAN["scales"] == [1, 0.75, 0.5, 0, -0.5, -0.75, -1]
    assert PLAN["changed_ticks"] == list(range(10, 20))
    assert PLAN["arm_channel"] == 5 and PLAN["joint"] == "right_wrist_pitch_joint"
    assert PLAN["ticks"] == 100
    assert len(PLAN["scales"]) * len(PLAN["executors"]) * len(PLAN["physics_dt_seconds"]) == 28
    assert PLAN["full_trial_count"] == 28


def test_impact_components_preserve_signed_velocity():
    sample = {
        "solve_seconds": 0.4,
        "contacts": [
            {
                "normal_on_ball_world": [0.8, 0, 0.6],
                "ball_point_velocity_world_m_s": [-2.5, 0, -3.8],
                "bat_point_velocity_world_m_s": [1.2, 0, 0.2],
                "relative_normal_velocity_m_s": -5.36,
            }
        ],
    }
    result = impact_direction({"first_loaded_impact": sample})
    np.testing.assert_allclose(
        result["contacts"][0]["relative_normal_components_m_s"], [-2.96, 0, -2.4]
    )
    assert result["solve_seconds"] == 0.4
    assert impact_direction({"first_loaded_impact": None}) is None
    sample["contacts"][0]["relative_normal_velocity_m_s"] *= -1
    with pytest.raises(AssertionError):
        impact_direction({"first_loaded_impact": sample})
