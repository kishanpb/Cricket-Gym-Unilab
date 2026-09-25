"""Position-actuator encoding must reproduce the declared PD velocity term."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_tracking import evaluate, reference_bat_positions
from g1_cricket_tracking_control_audit import (
    ankle_balance,
    feasibility_checks,
    position_velocity_control,
)

from unilab.tasks.manipulation.g1_cricket.tracking import (
    SupportedCricketReferenceAction,
    root_position_balance,
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
@pytest.mark.parametrize("term", ["waist_tracking_gain", "root_position_gain", "lookahead_frames"])
def test_controller_variant_cannot_overwrite_parent(tmp_path, same_output, term):
    with pytest.raises(ValueError, match="separate output"):
        evaluate(tmp_path, output=tmp_path if same_output else None, **{term: 2})
    assert not list(tmp_path.iterdir())


def test_root_feedback_uses_reference_axes_and_velocity_damping():
    position, velocity = np.array([0.1, 0.2, 0.0]), np.array([0.4, -0.2, 0.0])
    for yaw in (0, -np.pi / 2, np.pi / 2):
        quaternion = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        rotation = rotation.reshape(3, 3)
        np.testing.assert_allclose(
            root_position_balance(quaternion, rotation @ position, rotation @ velocity, 2),
            [-0.32, 0.36],
            atol=1e-15,
        )
        np.testing.assert_array_equal(root_position_balance(quaternion, position, velocity, 0), 0)


@pytest.mark.parametrize("lead", [0, 1])
def test_motor_lookahead_keeps_measurement_phase_and_clamps_at_clip_end(lead):
    position = np.arange(12, dtype=np.float32).reshape(4, 3)
    velocity = position / 10
    frames = np.array([0, 2, 3])
    queried = []

    def get_motion(indices):
        queried.append(indices.copy())
        return SimpleNamespace(joint_pos=position[indices], joint_vel=velocity[indices])

    action = object.__new__(SupportedCricketReferenceAction)
    action.cfg = SimpleNamespace(lookahead_frames=lead, scale=0.25)
    action.command = SimpleNamespace(
        time_steps=frames.copy(),
        joint_pos=position[frames].copy(),
        joint_vel=velocity[frames].copy(),
        motion=SimpleNamespace(get_motion_at_frame=get_motion),
    )
    action._raw = np.zeros((3, 3), dtype=np.float32)
    action.target = np.zeros_like(action._raw)
    action.velocity_gain = np.array([0.1, 0.2, 0.3])
    action.gravity_offset = position / 20
    actions = np.array([[2, -2, 0.1]] * 3, dtype=np.float32)
    expected_frames = np.minimum(frames + lead, 3)
    expected = position[expected_frames] + 0.25 * np.clip(actions, -1, 1)
    expected += action.velocity_gain * velocity[expected_frames]
    expected += action.gravity_offset[expected_frames]
    action._reference_with_feedforward(actions)
    np.testing.assert_array_equal(action.target, expected)
    np.testing.assert_array_equal(action.raw_action, actions)
    np.testing.assert_array_equal(action.command.time_steps, frames)
    np.testing.assert_array_equal(action.command.joint_pos, position[frames])
    np.testing.assert_array_equal(action.command.joint_vel, velocity[frames])
    if lead:
        np.testing.assert_array_equal(queried, [expected_frames])
    else:
        assert queried == []


@pytest.mark.parametrize("lead", [-1, 0.5, True])
def test_invalid_motor_lookahead_is_rejected_before_environment_access(lead):
    with pytest.raises(ValueError, match="nonnegative integer"):
        SupportedCricketReferenceAction(SimpleNamespace(lookahead_frames=lead), None)
