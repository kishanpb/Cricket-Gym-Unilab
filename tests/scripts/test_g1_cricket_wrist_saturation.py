"""Signed requested torque must explain every retained applied-torque sample."""

from types import SimpleNamespace

import numpy as np
import pytest
from audit_g1_cricket_wrist_saturation import summarize, torque_request


def test_affine_torque_uses_solve_position_and_velocity():
    model = SimpleNamespace(
        actuator_gainprm=np.array([[40.0]]), actuator_biasprm=np.array([[0.0, -40.0, -10.0]])
    )
    data = SimpleNamespace(ctrl=[0.1], actuator_length=[0.2], actuator_velocity=[0.3])
    assert torque_request(model, data, 0) == -7


def test_saturation_counts_both_signs_and_interior():
    r = summarize(10, 0.1, [-9, -5, -1, 0, 3, 5, 8], [-5, -5, -1, 0, 3, 5, 5], [0, 0.1], [-2, 2])
    assert r["saturated_steps"] == 4
    assert r["positive_saturated_steps"] == r["negative_saturated_steps"] == 2
    assert r["physics_steps"] == 7
    assert r["maximum_clipped_torque_error_nm"] == 0
    assert r["requested_torque_range_nm"] == [-9, 8]
    assert r["applied_torque_range_nm"] == [-5, 5]


def test_disagreeing_physical_force_fails_closed():
    with pytest.raises(AssertionError):
        summarize(0, 0, [-9], [5], [0], [0])
