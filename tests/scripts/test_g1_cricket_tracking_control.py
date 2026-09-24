"""Position-actuator encoding must reproduce the declared PD velocity term."""

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_tracking import evaluate, reference_bat_positions
from g1_cricket_tracking_control_audit import (
    ankle_balance,
    feasibility_checks,
    position_velocity_control,
)


def test_velocity_reference_cancels_damping_at_desired_motion():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody><body>
      <joint name="joint"/><geom type="sphere" size=".1"/>
    </body></worldbody><actuator><position joint="joint" kp="40" kv="10"/></actuator></mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = 0.4, 2.0
    data.ctrl[:] = position_velocity_control(model, data.qpos, data.qvel)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 0, atol=1e-12)
    np.testing.assert_array_equal(
        position_velocity_control(model, data.qpos, np.zeros(1)), data.qpos
    )
    data.qvel[:] = 1.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.actuator_force, 10, atol=1e-12)


def test_balance_feedback_has_correct_sign_and_is_yaw_equivariant():
    pitch = np.array([np.cos(0.1), 0, np.sin(0.1), 0])
    identity = np.array([1.0, 0, 0, 0])
    expected = np.array([0, 0.14])
    np.testing.assert_allclose(ankle_balance(identity, pitch, np.zeros(3), 1), expected)
    for yaw in (-np.pi / 2, np.pi / 2):
        base = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        current = np.empty(4)
        mujoco.mju_mulQuat(current, base, pitch)
        np.testing.assert_allclose(
            ankle_balance(base, current, np.zeros(3), 1), expected, atol=1e-15
        )
        np.testing.assert_array_equal(ankle_balance(base, current, np.ones(3), 0), [0, 0])
        assert np.max(np.abs(ankle_balance(base, current, np.ones(3), 100))) == 0.3


def test_completion_alone_cannot_clear_feasibility():
    frame = {
        "minimum_substep_pelvis_height_m": 0.75,
        "unexpected_contacts": [],
        "joint_limit_excess_rad": 0.0,
        "maximum_substep_grip_gap_m": 0.001,
        "motor_fraction_peak": 0.5,
        "root_translation_error_m": 0.01,
        "joint_rmse_rad": 0.01,
        "bat_tracking_error_m": 0.01,
    }
    row = {"completed_three_seconds": True, "trace": [frame]}
    assert all(feasibility_checks(row).values())
    frame["joint_limit_excess_rad"] = 0.00011
    frame["unexpected_contacts"] = ["left_hand/right_elbow"]
    checks = feasibility_checks(row)
    assert checks["complete"]
    assert not checks["hard_joint_limits"]
    assert not checks["no_unexpected_contact"]


def test_bat_reference_uses_forward_kinematics_without_changing_poses():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/><site name="bat_center" pos=".2 .3 .4"/>
    </body></worldbody></mujoco>
    """)
    poses = np.array([[0.0], [0.5], [-0.25]])
    original = poses.copy()
    np.testing.assert_allclose(
        reference_bat_positions(model, poses), [[0.2, 0.3, 0.4], [0.7, 0.3, 0.4], [-0.05, 0.3, 0.4]]
    )
    np.testing.assert_array_equal(poses, original)


@pytest.mark.parametrize("same_output", [False, True])
def test_controller_variant_cannot_overwrite_parent(tmp_path, same_output):
    with pytest.raises(ValueError, match="separate output"):
        evaluate(tmp_path, output=tmp_path if same_output else None, waist_tracking_gain=2)
    assert not list(tmp_path.iterdir())
