import mujoco
import numpy as np
from evaluate_g1_cricket_running_velocity import sample_curve_reference
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
