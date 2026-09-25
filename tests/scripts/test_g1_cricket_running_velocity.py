import mujoco
import numpy as np
from evaluate_g1_cricket_running_velocity import sample_curve_reference
from retarget_g1_cricket_running import running_control
from scipy.spatial.transform import Rotation

from unilab.tasks.manipulation.g1_cricket.tracking import ankle_balance


def test_exact_moving_target_has_no_rate_error():
    q = np.array([1.0, 0, 0, 0])
    velocity = np.array([0.38, 3.58, 0.3])
    assert ankle_balance(q, q, velocity, 4)[1] == 0.3
    np.testing.assert_array_equal(
        ankle_balance(q, q, velocity, 4, reference_angular_velocity=velocity), np.zeros(2)
    )


def test_reference_rate_is_in_reference_body_frame():
    reference = np.array([1.0, 0, 0, 0])
    rotation = Rotation.from_rotvec([0, 0, np.pi / 2])
    quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    reference_rate = np.array([0.2, 0.4, 0])
    current_rate = rotation.inv().apply(reference_rate)
    np.testing.assert_allclose(
        ankle_balance(
            reference, quaternion, current_rate, 4, reference_angular_velocity=reference_rate
        ),
        np.zeros(2),
        atol=1e-12,
    )


def test_zero_reference_preserves_legacy_result():
    reference = np.array([1.0, 0, 0, 0])
    q = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_quat()[[3, 0, 1, 2]]
    velocity = np.array([-0.3, 0.2, 0.1])
    np.testing.assert_array_equal(
        ankle_balance(reference, q, velocity, 4),
        ankle_balance(reference, q, velocity, 4, reference_angular_velocity=np.zeros(3)),
    )


def test_curve_initial_velocity_and_first_intervals_follow_positions():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint type="slide"/>'
        '<geom size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    times = np.array([0.0, 0.005, 0.01])

    def curve(time):
        return np.array([0.2 + 0.4 * time + 30 * time**2])

    sampled = sample_curve_reference(model, curve, times)
    np.testing.assert_allclose(sampled["qpos"][:, 0], 0.2 + 0.4 * times + 30 * times**2)
    np.testing.assert_allclose(sampled["qvel"][:, 0], 0.4 + 60 * times, atol=1e-12)
    forward_velocity = np.array([(curve(t + 0.005) - curve(t))[0] / 0.005 for t in times])
    np.testing.assert_allclose(forward_velocity - sampled["qvel"][:, 0], 0.15, atol=1e-12)


def test_joint_gain_changes_position_and_velocity_feedback_not_model():
    model = SimpleNamespace(
        actuator_gainprm=np.zeros((29, 10)), actuator_biasprm=np.zeros((29, 10))
    )
    model.actuator_gainprm[:, 0] = 20
    model.actuator_biasprm[:, 2] = -2
    data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), qfrc_bias=np.zeros(41))
    data.qpos[3] = 1
    qa, va = np.arange(7, 36), np.arange(6, 35)
    target, velocity = data.qpos.copy(), data.qvel.copy()
    target[qa], velocity[va] = np.linspace(-0.2, 0.2, 29), np.linspace(-0.5, 0.5, 29)
    data.qvel[va] = -velocity[va]
    original = (model.actuator_gainprm.copy(), model.actuator_biasprm.copy())
    base, _ = running_control(model, data, target, velocity, qa, va)
    stronger, _ = running_control(model, data, target, velocity, qa, va, joint_tracking_gain=4)
    expected_delta = 3 * (target[qa] - data.qpos[qa]) + 0.1 * (velocity[va] - data.qvel[va])
    np.testing.assert_allclose(stronger - base, expected_delta, atol=1e-15)
    np.testing.assert_array_equal(model.actuator_gainprm, original[0])
    np.testing.assert_array_equal(model.actuator_biasprm, original[1])


from types import SimpleNamespace
