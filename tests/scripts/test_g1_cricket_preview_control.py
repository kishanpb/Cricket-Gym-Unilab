"""Sensor-reduced batched prediction must match the original physics."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from retarget_g1_cricket_running import running_control

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.preview_control import PreviewControl, build_preview_model

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
    preview = build_preview_model(scene, tmp_path / "preview.xml", model.opt.timestep)
    with np.load(ROOT / f"g1_cricket_results/running_momentum_v1/{hand}_reference.npz") as saved:
        reference = {key: saved[key].copy() for key in saved.files}
    controller = PreviewControl(model, preview, reference, running_control, samples=4, horizon=2)
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = reference["qpos"][0], reference["qvel"][0]
    mujoco.mj_forward(model, data)
    yield model, preview, controller, data, reference
    controller.close()


def test_sensor_reduction_preserves_physical_model_and_native_trajectory(setup):
    model, preview, controller, data, _ = setup
    assert preview.nsensordata == 11
    for name in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "jnt_range",
        "jnt_axis",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "dof_damping",
        "dof_frictionloss",
        "mesh_vert",
        "mesh_face",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "actuator_ctrllimited",
        "eq_data",
        "eq_solref",
        "eq_solimp",
    ):
        np.testing.assert_array_equal(getattr(model, name), getattr(preview, name))
    for name in (
        "timestep",
        "gravity",
        "integrator",
        "solver",
        "iterations",
        "tolerance",
        "cone",
        "disableflags",
        "enableflags",
    ):
        np.testing.assert_array_equal(getattr(model.opt, name), getattr(preview.opt, name))
    plans = controller.nominal[:2][None]
    state, sensors = controller.predict(data, plans)
    native = mujoco.MjData(model)
    mujoco.mj_copyData(native, model, data)
    snapshot = np.empty(state.shape[-1])
    for i in range(640):
        native.ctrl[:] = plans[0, i // 320]
        mujoco.mj_step(model, native)
        mujoco.mj_getState(model, native, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_allclose(state[0, i], snapshot, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            sensors[0, i, :3], native.xpos[model.body("left_ankle_roll_link").id], atol=1e-12
        )


def test_preview_keeps_live_state_and_applies_only_bounded_first_command(setup):
    model, _, controller, data, reference = setup
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, before, spec)
    command = controller(data, reference["qpos"][0], reference["qvel"][0])
    after = np.empty_like(before)
    mujoco.mj_getState(model, data, after, spec)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(command, controller.plan[0])
    assert np.all(command >= controller.limits[:, 0]) and np.all(command <= controller.limits[:, 1])
    assert len(controller.trace[-1]["stages"]) == 2
    for stage in controller.trace[-1]["stages"]:
        assert len(stage["costs"]) == 4
        assert stage["selected"] == int(np.argmin(stage["costs"]))


def test_preview_release_matches_native_holder_transition(setup):
    model, _, controller, data, reference = setup
    data.time = 1.8
    data.qpos[:], data.qvel[:] = reference["qpos"][90], reference["qvel"][90]
    mujoco.mj_forward(model, data)
    plans = controller.nominal[90:92][None]
    state, _ = controller.predict(data, plans)
    native = mujoco.MjData(model)
    mujoco.mj_copyData(native, model, data)
    snapshot = np.empty(state.shape[-1])
    for i in range(640):
        native.ctrl[:] = plans[0, i // 320]
        native.eq_active[model.equality("ball_holder").id] = i < 320
        mujoco.mj_step(model, native)
        mujoco.mj_getState(model, native, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_allclose(state[0, i], snapshot, atol=1e-12, rtol=0)


def test_prediction_penalties_inspect_every_substep(setup, monkeypatch):
    model, _, controller, data, _ = setup
    plans = np.repeat(controller.nominal[None, :2], 4, axis=0)
    states, sensors = controller.predict(data, plans)
    states[0, 10, 1 + controller.qa[0]] = controller.limits[0, 1] + 0.1
    sensors[1, 10, 6] = -0.01
    states[2, 10, 3] = 0.4
    states[3, 10, 0] = states[3, 9, 0]
    monkeypatch.setattr(controller, "predict", lambda *_: (states, sensors))
    costs, audit = controller.costs(data, plans)
    assert all(audit["predicted_unsafe"])
    assert np.all(costs >= 1e4)
    assert audit["peak_joint_limit_excess_rad"][0] == pytest.approx(0.1)
    assert audit["minimum_leg_clearance_m"][1] == -0.01
    assert audit["minimum_pelvis_height_m"][2] == 0.4
