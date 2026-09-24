"""Aerial reference diagnostics distinguish ballistic motion from required support."""

import numpy as np
import pytest
from audit_g1_cricket_running_flight import aerial_residual


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
