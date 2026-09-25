"""Offline lateral COM target for the existing alternating running supports."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

from .running import GATHER_TIME, RunningDeliveryTargets


class LateralSupportCOM:
    """Periodic variable-height pendulum with ballistic flights, not full dynamics."""

    def __init__(self, parent, hand, lane):
        self.parent, self.lane = parent, lane
        self.sign = 1 if hand == "right" else -1

        def dynamics(time, state):
            height = float(parent.stance(time))
            acceleration = float(parent.stance(time, 2))
            coefficient = (parent.gravity + acceleration) / height
            values = state.reshape(2, 3)
            return np.vstack((values[1], coefficient * (values[0] - [0, 0, 0.12]))).ravel()

        # Propagate two homogeneous basis states and one forced solution.
        basis = solve_ivp(
            dynamics,
            (0, 0.22),
            np.array([[1, 0, 0], [0, 1, 0]]).ravel(),
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
        )
        if not basis.success:
            raise RuntimeError(basis.message)
        endpoint = basis.y[:, -1].reshape(2, 3)
        flight = np.array([[1, 0.08], [0, 1]]) @ endpoint
        initial = np.linalg.solve(flight[:, :2] + np.eye(2), -flight[:, 2])
        self.coefficients = np.r_[initial, 1]
        self.basis = basis.sol
        self.takeoff = endpoint @ self.coefficients
        start = self.lateral(GATHER_TIME)
        end = parent(1.32)[1]
        self.gather = CubicHermiteSpline(
            [GATHER_TIME, 1.32],
            [lane + start[0], end],
            [start[1], parent.parent.derivative()(1.32)[1]],
        )

    def lateral(self, time):
        cycle = int(np.floor((time + 1e-12) / 0.3))
        phase = max(0.0, time - cycle * 0.3)
        if phase <= 0.22:
            state = self.basis(phase).reshape(2, 3) @ self.coefficients
        else:
            state = self.takeoff + np.array([(phase - 0.22) * self.takeoff[1], 0])
        return self.sign * (-1) ** cycle * state

    def __call__(self, time):
        result = self.parent(time).copy()
        if time < GATHER_TIME:
            result[1] = self.lane + self.lateral(time)[0]
        elif time < 1.32:
            result[1] = self.gather(time)
        return result


class ForeAftSupportCOM(LateralSupportCOM):
    """Moving-start periodic pendulum targets about the planted ankle, not rest."""

    def __init__(self, parent, hand, lane):
        super().__init__(parent, hand, lane)
        targets = RunningDeliveryTargets(hand)
        front = "left" if hand == "right" else "right"
        self.first_support_x = targets.foot(front, 0)[0]
        self.step_length = targets.foot(hand, 0.3)[0] - self.first_support_x
        endpoint = self.basis(0.22).reshape(2, 3)[:, :2]
        transfer = np.array([[1, 0.08], [0, 1]]) @ endpoint
        # The next foot advances one step while the relative state repeats.
        self.forward_initial = np.linalg.solve(transfer - np.eye(2), [self.step_length, 0])
        self.forward_takeoff = endpoint @ self.forward_initial
        start = self.forward(GATHER_TIME)
        self.forward_gather = CubicHermiteSpline(
            [GATHER_TIME, 1.32],
            [start[0], parent(1.32)[0]],
            [start[1], parent.parent.derivative()(1.32)[0]],
        )

    def forward(self, time):
        cycle = int(np.floor((time + 1e-12) / 0.3))
        phase = max(0.0, time - cycle * 0.3)
        if phase <= 0.22:
            state = self.basis(phase).reshape(2, 3)[:, :2] @ self.forward_initial
        else:
            state = self.forward_takeoff + np.array([(phase - 0.22) * self.forward_takeoff[1], 0])
        return state + [self.first_support_x + cycle * self.step_length, 0]

    def __call__(self, time):
        result = super().__call__(time)
        if time < GATHER_TIME:
            result[0] = self.forward(time)[0]
        elif time < 1.32:
            result[0] = self.forward_gather(time)
        return result
