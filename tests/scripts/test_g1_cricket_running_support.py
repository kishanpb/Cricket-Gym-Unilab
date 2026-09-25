import numpy as np
import pytest
from audit_g1_cricket_running_support import ground_wrench

from unilab.tasks.manipulation.g1_cricket.running import BallisticRunupCOM
from unilab.tasks.manipulation.g1_cricket.running_support import LateralSupportCOM


def target(hand):
    times = np.arange(136) * 0.02
    lane = 0.7 if hand == "right" else -0.7
    centers = np.column_stack((times, np.full_like(times, lane), np.full_like(times, 0.65)))
    return LateralSupportCOM(BallisticRunupCOM(times, centers, 9.81), hand, lane)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_support_force_location_and_flight(hand):
    curve = target(hand)
    dt = 1e-5
    for time in np.arange(0.01, 1.2, 0.01):
        phase = round(time % 0.3, 8)
        if phase in (0, 0.22):
            continue
        center = curve(time)
        acceleration = (curve(time + dt) - 2 * center + curve(time - dt)) / dt**2
        if phase > 0.22:
            np.testing.assert_allclose(acceleration[1:], [0, -9.81], atol=2e-5)
        else:
            cop = center[1] - center[2] * acceleration[1] / (9.81 + acceleration[2])
            foot = curve.lane + curve.sign * (-1) ** int(time / 0.3) * 0.12
            assert cop == pytest.approx(foot, abs=2e-6)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_continuity_and_unchanged_axes(hand):
    curve = target(hand)
    for time in np.linspace(0, 2.7, 271):
        np.testing.assert_array_equal(curve(time)[[0, 2]], curve.parent(time)[[0, 2]])
        if time >= 1.32:
            np.testing.assert_array_equal(curve(time), curve.parent(time))
    dt = 1e-6
    for time in (0.22, 0.3, 0.52, 0.6, 0.82, 0.9, 1.12, 1.2, 1.32):
        before = (curve(time)[1] - curve(time - dt)[1]) / dt
        after = (curve(time + dt)[1] - curve(time)[1]) / dt
        assert before == pytest.approx(after, abs=2e-5)


def test_left_right_mirror():
    right, left = target("right"), target("left")
    for time in np.linspace(0, 2.7, 271):
        np.testing.assert_allclose(right(time) * [1, -1, 1], left(time), atol=1e-12)


def test_ground_wrench_static_accelerating_and_flight():
    gravity = np.array([0, 0, -9.81])
    force, cop = ground_wrench(np.array([1, 2, 0.6]), np.zeros(3), np.zeros(3), 40, gravity)
    np.testing.assert_allclose(force, [0, 0, 392.4])
    np.testing.assert_allclose(cop, [1, 2])
    force, cop = ground_wrench(
        np.array([1, 2, 0.6]), np.array([1, -2, 0]), np.array([3, 4, 5]), 40, gravity
    )
    np.testing.assert_allclose(cop, [1 - (24 + 4) / 392.4, 2 + (48 + 3) / 392.4])
    force, cop = ground_wrench(np.ones(3), gravity, np.zeros(3), 40, gravity)
    np.testing.assert_array_equal(force, np.zeros(3))
    assert cop is None
