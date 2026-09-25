from pathlib import Path

import mujoco
import numpy as np
import pytest
from g1_cricket_delivery_trial import capsule_bounds

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.reference_dynamics import ReferenceDynamics, curve_state

ROOT = Path(__file__).resolve().parents[2]


def slide_model():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody><body>
      <joint name="slide" type="slide" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size=".1" mass="2"/>
    </body></worldbody><actuator>
      <position joint="slide" kp="10" kv="2" forcerange="-5 5"/>
    </actuator></mujoco>""")


def test_curve_derivatives():
    model = slide_model()
    q, v, a = curve_state(model, lambda t: np.array([0.2 + 0.3 * t + 0.4 * t**2]), 0.1, 1e-4)
    np.testing.assert_allclose(q, [0.234])
    np.testing.assert_allclose(v, [0.38], atol=1e-10)
    np.testing.assert_allclose(a, [0.8], atol=1e-8)


def test_force_cap_and_position_command_reachability():
    helper = ReferenceDynamics(slide_model())
    result = helper.evaluate(np.array([0.0]), np.array([0.0]), np.array([4.0]))
    np.testing.assert_allclose(result["required_motor_force"], [8])
    np.testing.assert_allclose(result["motor_limit_excess"], [3])
    np.testing.assert_allclose(result["bounded_controls"], [0.5])
    np.testing.assert_allclose(result["unactuated_residual"], [0])
    np.testing.assert_allclose(result["bounded_force_residual"], [3])
    allocation = helper.support_allocation(
        np.array([4.0]), [], helper.model.actuator_forcerange, minimum_motor_scale=True
    )
    assert allocation["minimum_motor_scale"] == pytest.approx(1.6)
    # Near a joint stop, damping can prevent a force below the hardware cap.
    result = helper.evaluate(np.array([0.9]), np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(result["required_motor_force"], [2])
    np.testing.assert_allclose(result["motor_limit_excess"], [0])
    np.testing.assert_allclose(result["reachable_motor_force_bounds"], [[-5, -1]])
    np.testing.assert_allclose(result["reachable_force_excess"], [3])
    np.testing.assert_allclose(result["bounded_controls"], [1])


def test_unactuated_gravity_cannot_be_replaced_by_motors():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body pos="0 0 2"><freejoint/>
      <geom type="sphere" size=".1" mass="2"/>
      <body><joint name="hinge" range="-1 1"/><geom size=".05" mass="1"/></body>
    </body></worldbody><actuator>
      <position joint="hinge" kp="10" kv="2" forcerange="-5 5"/>
    </actuator></mujoco>""")
    helper = ReferenceDynamics(model)
    result = helper.evaluate(model.qpos0, np.zeros(model.nv), np.zeros(model.nv))
    np.testing.assert_allclose(result["unactuated_residual"][:3], [0, 0, 3 * 9.81])
    acceleration = np.zeros(model.nv)
    acceleration[2] = -9.81
    result = helper.evaluate(model.qpos0, np.zeros(model.nv), acceleration)
    np.testing.assert_allclose(result["required_generalized_force"], 0, atol=1e-12)
    bounded = result["reachable_motor_force_bounds"]
    assert helper.support_allocation(acceleration, [], bounded)["feasible"]
    acceleration[:] = 0
    assert not helper.support_allocation(acceleration, [], bounded)["feasible"]
    patch = (1, np.array([[-0.1, -0.1, 0], [0.1, 0.1, 0]]), 0.8)
    supported = helper.support_allocation(acceleration, [patch], bounded)
    assert supported["feasible"]
    np.testing.assert_allclose(supported["support_wrenches_world"][0][:3], [0, 0, 29.43])
    off_center = (1, patch[1] + [1, 0, 0], 0.8)
    assert not helper.support_allocation(acceleration, [off_center], bounded)["feasible"]
    acceleration[6] = 1e5
    assert helper.support_allocation(acceleration, [patch], [(None, None)])["feasible"]
    assert not helper.support_allocation(acceleration, [patch], bounded)["feasible"]


def test_friction_loss_is_bounded_in_optimistic_allocation():
    model = slide_model()
    model.dof_frictionloss[:] = 3
    helper = ReferenceDynamics(model)
    helper.evaluate(np.zeros(1), np.zeros(1), np.zeros(1))
    result = helper.support_allocation(np.array([4.0]), [], model.actuator_forcerange)
    assert result["feasible"]
    np.testing.assert_allclose(result["motor_forces"], [5])
    np.testing.assert_allclose(result["friction_forces"], [3])
    assert not helper.support_allocation(np.array([4.5]), [], model.actuator_forcerange)["feasible"]


@pytest.mark.parametrize("hand", ["right", "left"])
def test_native_forward_inverse_and_no_model_edits(hand, tmp_path, monkeypatch):
    path = tmp_path / "scene.xml"
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", path
    )
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 0.0000625
    data = mujoco.MjData(model)
    with np.load(
        ROOT / f"g1_cricket_results/running_ground_momentum_v2/{hand}_physical.npz"
    ) as saved:
        data.qpos[:] = saved["qpos"][5]
    data.qvel[:] = np.random.default_rng(2).normal(0, 0.1, model.nv)
    joints = model.actuator_trnid[:, 0]
    data.ctrl[:] = data.qpos[model.jnt_qposadr[joints]]
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    original = {
        name: getattr(model, name).copy()
        for name in (
            "body_mass",
            "body_inertia",
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "geom_friction",
            "eq_data",
            "eq_solref",
            "eq_solimp",
        )
    }
    helper = ReferenceDynamics(model)

    def no_step(*args):
        raise AssertionError("offline dynamics must not integrate physics")

    monkeypatch.setattr(mujoco, "mj_step", no_step)
    result = helper.evaluate(data.qpos, data.qvel, data.qacc)
    np.testing.assert_allclose(result["required_generalized_force"], data.qfrc_actuator, atol=1e-5)
    np.testing.assert_allclose(result["required_motor_force"], data.actuator_force, atol=1e-5)
    np.testing.assert_allclose(result["unactuated_residual"], 0, atol=1e-5)
    np.testing.assert_allclose(result["bounded_controls"], data.ctrl, atol=1e-6)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, helper.data, mass)
    np.testing.assert_allclose(
        result["required_generalized_force"],
        mass @ data.qacc
        + helper.data.qfrc_bias
        - helper.data.qfrc_passive
        - result["constraint_generalized_force"],
        atol=1e-9,
    )
    patches = []
    for side in ("left", "right"):
        body = model.body(f"{side}_ankle_roll_link").id
        geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
        box = capsule_bounds(
            data.geom_xpos[geoms],
            data.geom_xmat[geoms].reshape(-1, 3, 3),
            model.geom_size[geoms],
            model.geom_type[geoms],
        )
        if box[0, 2] <= 0.002:
            patches.append(
                (
                    body,
                    box,
                    float(
                        max(
                            model.geom_friction[geoms, 0].max(),
                            model.geom_friction[model.geom("pitch").id, 0],
                        )
                    ),
                )
            )
    allocation = helper.support_allocation(
        data.qacc,
        patches,
        result["reachable_motor_force_bounds"],
        (model.body(f"{hand}_wrist_yaw_link").id, model.body("cricket_ball").id),
    )
    assert allocation["feasible"]
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(model, name), value)
