"""Stance diagnostics preserve the prior motor controller and native trajectory."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from audit_g1_cricket_running_stance import COLUMNS, foot_loads, replay
from retarget_g1_cricket_running import running_control

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.tracking import ankle_balance, root_position_balance

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
    with np.load(ROOT / f"g1_cricket_results/running_momentum_v1/{hand}_reference.npz") as data:
        reference = {key: data[key].copy() for key in data.files}
    return model, hand, reference


def test_extracted_control_matches_original_expression(setup):
    model, _, reference = setup
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = reference["qpos"][4], reference["qvel"][4]
    data.qpos[0] += 0.02
    data.qvel[4] -= 0.4
    mujoco.mj_forward(model, data)
    joints = [model.joint(name).id for name in SDK_JOINTS]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    target, velocity = reference["qpos"][5], reference["qvel"][5]
    expected = (
        target[qa]
        + (-model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0]) * velocity[va]
        + data.qfrc_bias[va] / model.actuator_gainprm[:, 0]
    )
    correction = ankle_balance(target[3:7], data.qpos[3:7], data.qvel[3:6], 4)
    correction += root_position_balance(
        target[3:7], data.qpos[:3] - target[:3], data.qvel[:3] - velocity[:3], 4
    )
    correction = np.clip(correction, -0.3, 0.3)
    expected[[4, 10]] += correction[1]
    expected[[5, 11]] += correction[0]
    expected[14] += target[qa[14]] - data.qpos[qa[14]]
    actual, _ = running_control(model, data, target, velocity, qa, va)
    np.testing.assert_array_equal(actual, expected)
    no_balance, removed = running_control(model, data, target, velocity, qa, va, balance_gain=0)
    np.testing.assert_array_equal(removed, np.zeros(2))
    difference = np.zeros(model.nu)
    difference[[4, 10]], difference[[5, 11]] = correction[1], correction[0]
    np.testing.assert_allclose(actual - no_balance, difference, atol=1e-15)


def test_full_baseline_replay_matches_frozen_physics_and_records_first_crossing(setup):
    model, hand, reference = setup
    before = model.jnt_range.copy(), model.actuator_forcerange.copy(), model.body_mass.copy()
    summary, steps, poses = replay(model, reference, 4)
    with np.load(ROOT / f"g1_cricket_results/running_momentum_v1/{hand}_physical.npz") as frozen:
        np.testing.assert_array_equal(poses, frozen["qpos"])
    assert steps.shape == (10880, len(COLUMNS))
    assert np.isfinite(steps).all()
    assert summary["duration_s"] == pytest.approx(0.68)
    assert 0.146 < summary["first_joint_limit"]["time_s"] < 0.147
    assert not summary["released"] and not summary["completed_horizon"]
    crossing = np.flatnonzero(steps[:, COLUMNS.index("maximum_joint_limit_excess")] > 1e-6)[0]
    assert steps[crossing, 1] == summary["first_joint_limit"]["time_s"]
    assert np.max(steps[:, COLUMNS.index("motor_force_fraction")]) <= 1
    for original, current in zip(
        before, (model.jnt_range, model.actuator_forcerange, model.body_mass), strict=True
    ):
        np.testing.assert_array_equal(original, current)


def test_foot_loads_are_normal_contact_forces_only():
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><geom name="pitch" type="plane" size="2 2 .1"/>
      <body pos="0 0 .09"><freejoint/>
      <geom name="left_foot_collision" type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 1
    wrench = np.zeros(6)
    mujoco.mj_contactForce(model, data, 0, wrench)
    assert wrench[0] > 0
    np.testing.assert_allclose(foot_loads(model, data), [wrench[0], 0])
