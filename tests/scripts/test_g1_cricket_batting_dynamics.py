import mujoco
import numpy as np
import pytest

from unilab.tasks.manipulation.g1_cricket.batting_dynamics import (
    constraint_contributions,
    evaluate_modes,
    independent_ball_dofs,
    reference_derivatives,
)
from unilab.tasks.manipulation.g1_cricket.reference_dynamics import ReferenceDynamics


def model_with_ball():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1"/>
    <body pos="0 0 1"><joint name="slide" type="slide" axis="1 0 0" range="-1 1" damping=".3" frictionloss=".2"/>
      <geom type="sphere" size=".1" mass="2"/></body>
    <body name="ball" pos="0 0 2"><freejoint name="ball_free"/><geom size=".1" mass="1"/></body>
    </worldbody><actuator><position joint="slide" kp="10" kv="2" forcerange="-5 5"/></actuator></mujoco>""")


@pytest.mark.parametrize("stencil", [1, 2])
def test_reference_derivative_convention_and_endpoints(stencil):
    model = model_with_ball()
    times = np.arange(11) * 0.02
    poses = np.tile(model.qpos0, (len(times), 1))
    poses[:, 0] = 0.1 + 0.3 * times + 0.4 * times**2
    velocity, acceleration, endpoint = reference_derivatives(model, poses, times, stencil)
    np.testing.assert_allclose(
        velocity[stencil:-stencil, 0], 0.3 + 0.8 * times[stencil:-stencil], atol=1e-12
    )
    np.testing.assert_allclose(acceleration[~endpoint, 0], 0.8, atol=1e-12)
    assert np.count_nonzero(endpoint) == 4 * stencil
    assert endpoint[0] and endpoint[-1]


def test_rotating_root_rejected():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    poses = np.tile(model.qpos0, (5, 1))
    poses[2, 3:7] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    with pytest.raises(ValueError, match="root quaternion"):
        reference_derivatives(model, poses, np.arange(5) * 0.02)


def test_force_decomposition_excludes_independent_ball():
    model = model_with_ball()
    helper = ReferenceDynamics(model)
    velocity, acceleration = np.zeros(model.nv), np.zeros(model.nv)
    velocity[0], acceleration[0] = 0.2, 0.8
    report = evaluate_modes(helper, model.qpos0, velocity, acceleration)
    np.testing.assert_array_equal(report["robot_dof_indices"], [0])
    np.testing.assert_array_equal(independent_ball_dofs(model), np.arange(1, 7))
    np.testing.assert_allclose(report["increments"]["dynamic_minus_velocity"]["inertia"], [1.6])
    np.testing.assert_allclose(report["increments"]["velocity_minus_static"]["passive"], [-0.06])
    for mode in report["modes"].values():
        assert len(mode["root_residual"]) == 0
        assert len(mode["robot_required_generalized_force"]) == 1
        assert len(mode["constraint_contributions"]["friction_loss"]) == 1
    ball_acceleration = acceleration.copy()
    ball_acceleration[1:4] = [1, 2, -9.81]
    alternate = evaluate_modes(helper, model.qpos0, velocity, ball_acceleration)
    for mode in report["modes"]:
        np.testing.assert_allclose(
            report["modes"][mode]["robot_required_generalized_force"],
            alternate["modes"][mode]["robot_required_generalized_force"],
        )


def test_native_constraint_groups_sum_to_total():
    model = model_with_ball()
    data = mujoco.MjData(model)
    data.qpos[0] = 1.01
    data.qpos[3] = 0.09
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    parts = constraint_contributions(model, data)
    np.testing.assert_allclose(sum(parts.values()), data.qfrc_constraint, atol=1e-10)
    assert np.linalg.norm(parts["contact"]) > 0
    assert np.linalg.norm(parts["joint_limit"]) > 0


def test_existing_audit_is_not_overwritten(tmp_path):
    from audit_g1_cricket_batting_dynamics import audit

    output = tmp_path / "audit.json"
    output.write_text("retained")
    with pytest.raises(FileExistsError):
        audit(output)
    assert output.read_text() == "retained"


def test_equality_force_is_not_mixed_with_contact():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><joint name="slide" type="slide"/>
    <geom size=".1"/></body></worldbody>
    <equality><joint joint1="slide" polycoef="0 0 0 0 0"/></equality></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[:] = 0.01
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0
    mujoco.mj_inverse(model, data)
    parts = constraint_contributions(model, data)
    assert np.linalg.norm(parts["equality"]) > 0
    np.testing.assert_allclose(parts["equality"], data.qfrc_constraint)
    np.testing.assert_array_equal(parts["contact"], 0)
