import numpy as np
import pytest
from audit_g1_cricket_running_support import ground_wrench
from scipy.integrate import quad_vec

from unilab.tasks.manipulation.g1_cricket.running import BallisticRunupCOM
from unilab.tasks.manipulation.g1_cricket.running_ground_momentum import RunningGroundMomentum
from unilab.tasks.manipulation.g1_cricket.running_support import LateralSupportCOM


@pytest.fixture(params=["right", "left"])
def targets(request):
    hand = request.param
    lane = 0.7 if hand == "right" else -0.7
    times = np.arange(136) * 0.02
    centers = np.column_stack((times, np.full_like(times, lane), np.full_like(times, 0.65)))
    com = LateralSupportCOM(BallisticRunupCOM(times, centers, 9.81), hand, lane)
    mean = np.array([0.2, 0.4, -0.1])
    return com, RunningGroundMomentum(com, 40, mean), mean


def test_cycle_average_and_continuity(targets):
    _, momentum, mean = targets
    measured, _ = quad_vec(momentum, 0, 0.6, points=[0.22, 0.3, 0.52])
    np.testing.assert_allclose(measured / 0.6, mean, atol=1e-10)
    for time in (0.22, 0.3, 0.52, 0.6, 0.82, 0.9, 1.12, 1.2):
        np.testing.assert_allclose(momentum(time - 1e-10), momentum(time + 1e-10), atol=1e-7)


def test_stance_wrench_and_ballistic_flight(targets):
    com, momentum, _ = targets
    step = 1e-5
    for time in np.arange(0.01, 1.2, 0.01):
        phase = round(time % 0.3, 8)
        if phase in (0, 0.22):
            continue
        acceleration = (com(time + step) - 2 * com(time) + com(time - step)) / step**2
        torque = (momentum(time + step) - momentum(time - step)) / (2 * step)
        force, cop = ground_wrench(com(time), acceleration, torque, 40, np.array([0, 0, -9.81]))
        if phase > 0.22:
            np.testing.assert_allclose(force, 0, atol=0.001)
            np.testing.assert_array_equal(torque, np.zeros(3))
            assert cop is None
        else:
            cycle = int(time / 0.3)
            np.testing.assert_allclose(
                cop,
                [com(cycle * 0.3 + 0.11)[0], com.lane + com.sign * (-1) ** cycle * 0.12],
                atol=2e-6,
            )


def test_constant_momentum_does_not_add_stance_impulse(targets):
    com, _, mean = targets
    momentum = RunningGroundMomentum(com, 40, mean, constant=True)
    for time in np.linspace(0, 1.2, 241):
        np.testing.assert_array_equal(momentum(time), mean)
    np.testing.assert_array_equal(momentum.initial, mean)
