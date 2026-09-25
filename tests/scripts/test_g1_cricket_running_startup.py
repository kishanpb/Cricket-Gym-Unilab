from pathlib import Path

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_startup import COLUMNS, replay_startup

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running_startup import (
    grounded_start,
    held_support_feedforward,
    startup_reference,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["right", "left"])
def setup(request, tmp_path):
    hand = request.param
    scene = tmp_path / "scene.xml"
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.000125
    return model, hand


def test_neutral_start_has_two_grounded_feet_and_consistent_holder(setup):
    model, hand = setup
    data = mujoco.MjData(model)
    data.qpos[:] = grounded_start(model, hand)
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        gap = min(
            mujoco.mj_geomDistance(
                model,
                data,
                model.geom(f"{side}_foot{i}_collision").id,
                model.geom("pitch").id,
                1,
                None,
            )
            for i in range(1, 8)
        )
        assert abs(gap) < 1e-12
    np.testing.assert_allclose(data.efc_pos[:6], 0, atol=1e-12)
    np.testing.assert_array_equal(data.qvel, np.zeros(model.nv))
    _, residual, loads = held_support_feedforward(model, data.qpos, hand)
    assert np.linalg.norm(residual) < 1e-8
    assert loads.min() > 100
    assert loads.sum() == pytest.approx(-model.body_mass.sum() * model.opt.gravity[2], abs=1e-8)


def test_reference_and_native_transfer_preserve_hardware_and_start_from_rest(setup):
    model, hand = setup
    before = [
        a.copy()
        for a in (
            model.body_mass,
            model.body_inertia,
            model.jnt_range,
            model.actuator_forcerange,
            model.actuator_gainprm,
        )
    ]
    reference = startup_reference(model, hand)
    assert reference["qpos"].shape == (276, model.nq)
    np.testing.assert_array_equal(reference["qvel"][:101], np.zeros((101, model.nv)))
    np.testing.assert_array_equal(reference["qvel"][176:], np.zeros((100, model.nv)))
    assert reference["errors"][:, 0].max() < 0.00011
    assert reference["errors"][:, 1].max() < 0.0015
    assert reference["errors"][:, 2].max() < 1e-8
    joints = [model.joint(name).id for name in SDK_JOINTS]
    command = reference["qpos"][:, model.jnt_qposadr[joints]]
    command = command + reference["torque"] / model.actuator_gainprm[:, 0]
    assert (command >= model.jnt_range[joints, 0]).all()
    assert (command <= model.jnt_range[joints, 1]).all()
    summary, trace = replay_startup(model, reference, hand)
    assert summary["passed"], summary["failures"]
    assert not summary["released"]
    assert summary["front_load_fraction"] > 0.8
    assert summary["measured_com_shift_m"] == pytest.approx(0.08, abs=0.002)
    assert trace["steps"].shape == (44000, len(COLUMNS))
    assert trace["qpos"].shape == reference["qpos"].shape
    assert np.isfinite(trace["steps"]).all()
    for old, new in zip(
        before,
        (
            model.body_mass,
            model.body_inertia,
            model.jnt_range,
            model.actuator_forcerange,
            model.actuator_gainprm,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(old, new)
