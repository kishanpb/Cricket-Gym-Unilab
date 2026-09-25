import numpy as np
import pytest
from audit_g1_cricket_running_support import ground_wrench

from unilab.tasks.manipulation.g1_cricket.running import BallisticRunupCOM, RunningDeliveryTargets
from unilab.tasks.manipulation.g1_cricket.running_support import (
    ForeAftSupportCOM,
    LateralSupportCOM,
)


def target(hand, fore_aft=False):
    times = np.arange(136) * 0.02
    lane = 0.7 if hand == "right" else -0.7
    centers = np.column_stack((times, np.full_like(times, lane), np.full_like(times, 0.65)))
    if fore_aft:
        centers[:, 0] -= 2.05
    support = ForeAftSupportCOM if fore_aft else LateralSupportCOM
    return support(BallisticRunupCOM(times, centers, 9.81), hand, lane)


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


@pytest.mark.parametrize("hand", ["right", "left"])
def test_fore_aft_pendulum_uses_actual_planted_foot(hand):
    curve = target(hand, fore_aft=True)
    feet = RunningDeliveryTargets(hand, lane_offset=0.2)
    dt = 1e-5
    for time in np.arange(0.01, 1.2, 0.01):
        phase = round(time % 0.3, 8)
        if phase in (0, 0.22):
            continue
        acceleration = (curve(time + dt) - 2 * curve(time) + curve(time - dt)) / dt**2
        force, cop = ground_wrench(curve(time), acceleration, np.zeros(3), 40, [0, 0, -9.81])
        if phase > 0.22:
            np.testing.assert_allclose(force, 0, atol=0.002)
            assert cop is None
        else:
            front = "left" if hand == "right" else "right"
            side = front if int(time / 0.3) % 2 == 0 else hand
            np.testing.assert_allclose(cop, feet.foot(side, time)[:2], atol=5e-6)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_fore_aft_periodicity_and_gather_continuity(hand):
    curve = target(hand, fore_aft=True)
    lateral = LateralSupportCOM(curve.parent, hand, curve.lane)
    for time in np.linspace(0, 2.7, 271):
        np.testing.assert_array_equal(curve(time)[1:], lateral(time)[1:])
        if time >= 1.32:
            np.testing.assert_array_equal(curve(time), lateral(time))
    for time in np.linspace(0, 0.89, 90):
        np.testing.assert_allclose(
            curve.forward(time + 0.3) - curve.forward(time), [0.3025, 0], atol=1e-10
        )
    for time in (0.22, 0.3, 0.52, 0.6, 0.82, 0.9, 1.12):
        before, after = curve.forward(time - 1e-8), curve.forward(time + 1e-8)
        np.testing.assert_allclose(before, after, atol=1e-7)
    np.testing.assert_allclose(curve.forward_gather(1.2), curve.forward(1.2)[0])
    np.testing.assert_allclose(curve.forward_gather(1.2, 1), curve.forward(1.2)[1])
    np.testing.assert_allclose(curve.forward_gather(1.32), curve.parent(1.32)[0])
    np.testing.assert_allclose(
        curve.forward_gather(1.32, 1), curve.parent.parent.derivative()(1.32)[0]
    )
    assert curve.forward_initial[1] > 1  # A periodic moving start, not rest.
