"""Aerial reference diagnostics distinguish ballistic motion from required support."""

import numpy as np
import pytest
from audit_g1_cricket_running_flight import aerial_residual

from unilab.tasks.manipulation.g1_cricket.running import BallisticRunupHeight


def test_runup_height_is_ballistic_in_flight_and_c1_at_phase_boundaries():
    times = np.arange(136) * 0.02
    parent = 0.7 + 0.02 * times
    height = BallisticRunupHeight(times, parent, 9.81)
    for cycle in range(4):
        t = cycle * 0.3 + 0.26
        acceleration = (height(t + 0.01) - 2 * height(t) + height(t - 0.01)) / 0.01**2
        assert acceleration == pytest.approx(-9.81, abs=1e-9)
    boundaries = [0.22, 0.3, 0.52, 0.6, 0.82, 0.9, 1.12, 1.2, 1.32]
    delta = 1e-7
    for t in boundaries:
        before = (height(t) - height(t - delta)) / delta
        after = (height(t + delta) - height(t)) / delta
        assert before == pytest.approx(after, abs=1e-5)
    for t in times[times >= 1.32]:
        assert height(t) == pytest.approx(0.7 + 0.02 * t)


def test_outward_lane_keeps_hand_mirror_and_other_targets():
    from unilab.tasks.manipulation.g1_cricket.running import RunningDeliveryTargets

    for hand, sign in (("right", 1), ("left", -1)):
        parent, shifted = RunningDeliveryTargets(hand), RunningDeliveryTargets(hand, 0.2)
        for t in np.arange(136) * 0.02:
            np.testing.assert_allclose(shifted.root(t) - parent.root(t), [0, sign * 0.2, 0])
            for side in ("right", "left"):
                np.testing.assert_allclose(
                    shifted.foot(side, t) - parent.foot(side, t), [0, sign * 0.2, 0]
                )
                for actual, expected in zip(shifted.arm(side, t), parent.arm(side, t), strict=True):
                    np.testing.assert_array_equal(actual, expected)


def test_ballistic_com_has_no_external_support_residual():
    times = np.arange(15) * 0.02
    gravity = np.array([0, 0, -9.81])
    com = times[:, None] * [1, 0.2, 0.5] + 0.5 * times[:, None] ** 2 * gravity
    rows = aerial_residual(times, com, np.ones(15, bool), 33.5, gravity)
    assert len(rows) == 13
    np.testing.assert_allclose([r["required_nongravity_force_n"] for r in rows], 0, atol=1e-10)


def test_upward_acceleration_requires_support_and_contact_stencils_are_excluded():
    times = np.arange(7) * 0.02
    com = np.zeros((7, 3))
    com[:, 2] = 2 * times**2
    airborne = np.array([False, False, True, True, True, False, False])
    rows = aerial_residual(times, com, airborne, 33.5, np.array([0, 0, -9.81]))
    assert len(rows) == 1 and rows[0]["time_s"] == pytest.approx(0.06)
    assert rows[0]["required_nongravity_force_n"][2] == pytest.approx(33.5 * 13.81)
    assert aerial_residual(times + 1.2, com, airborne, 33.5, np.zeros(3)) == []
