"""Native contact-control predictions cannot mutate the physical rollout."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from retarget_g1_cricket_running import running_control

from unilab.tasks.manipulation.g1_cricket.contact_control import ContactAccelerationControl
from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["right", "left"])
def setup(request, tmp_path):
    hand = request.param
    scene = tmp_path / "scene.xml"
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.0000625
    with np.load(ROOT / f"g1_cricket_results/running_momentum_v1/{hand}_reference.npz") as saved:
        poses, velocity = saved["qpos"], saved["qvel"]
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = poses[0], velocity[0]
    mujoco.mj_forward(model, data)
    return model, data, poses, velocity


def test_control_keeps_native_state_and_original_limits(setup, monkeypatch):
    model, data, poses, velocity = setup
    controller = ContactAccelerationControl(model, velocity, running_control)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, before, spec)
    limits, force_limits, mass = (
        model.jnt_range.copy(),
        model.actuator_forcerange.copy(),
        model.body_mass.copy(),
    )

    def forbid_step(*args):
        raise AssertionError("controller may only predict on scratch data")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    command = controller(data, poses[0], velocity[0])
    after = np.empty_like(before)
    mujoco.mj_getState(model, data, after, spec)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(limits, model.jnt_range)
    np.testing.assert_array_equal(force_limits, model.actuator_forcerange)
    np.testing.assert_array_equal(mass, model.body_mass)
    assert np.all(command >= controller.limits[:, 0])
    assert np.all(command <= controller.limits[:, 1])
    assert controller.trace[-1]["final_score"] <= controller.trace[-1]["initial_score"]
    assert len(controller.trace[-1]["stages"]) == 2
    assert np.isfinite(command).all()


def test_acceleration_jacobian_predicts_native_directional_change(setup):
    model, data, poses, velocity = setup
    controller = ContactAccelerationControl(model, velocity, running_control)
    command, _ = running_control(model, data, poses[0], velocity[0], controller.qa, controller.va)
    command = np.clip(command, controller.limits[:, 0], controller.limits[:, 1])
    direction = np.random.default_rng(0).normal(size=model.nu)
    jacobian = controller.jacobian(data, command)
    actual = (
        controller.acceleration(data, command + 1e-6 * direction)
        - controller.acceleration(data, command - 1e-6 * direction)
    ) / 2e-6
    np.testing.assert_allclose(jacobian @ direction, actual, rtol=0.01, atol=0.01)
