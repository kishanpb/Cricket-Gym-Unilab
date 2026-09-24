import numpy as np
import pytest
from probe_g1_cricket_delivery_motion import PLAN, action_at, candidates

from unilab.tasks.manipulation.g1_cricket.bowling import ARM_LIMITS


def test_fixed_search_keeps_each_hand_and_every_candidate():
    rows = candidates()
    assert len(rows) * len(PLAN["hands"]) == PLAN["rows"] == 32
    assert len({r["id"] for r in rows}) == 16
    for candidate in rows:
        for hand in PLAN["hands"]:
            actions = np.concatenate([action_at(candidate, hand, t) for t in range(200)])
            assert np.isfinite(actions).all()
            assert np.flatnonzero(actions[:, 7] > 0.5)[0] == 55 + candidate["delay"]
            np.testing.assert_array_equal(actions[:11, :7], 0)
            np.testing.assert_array_equal(actions[95:, :7], 0)
            offset = np.tanh(actions[45, :7]) * ARM_LIMITS
            assert offset[0] == pytest.approx(-2.45)
            assert offset[3] == pytest.approx(candidate["elbow"])
            assert offset[1] == pytest.approx(0.15 if hand == "left" else -0.15)
            assert (np.abs(np.tanh(actions[:, :7])) < 1).all()


def test_mirroring_only_changes_roll_not_release_or_pitch():
    candidate = candidates()[0]
    for tick in range(200):
        right = action_at(candidate, "right", tick)
        left = action_at(candidate, "left", tick)
        left[0, 1] *= -1
        np.testing.assert_array_equal(right, left)
