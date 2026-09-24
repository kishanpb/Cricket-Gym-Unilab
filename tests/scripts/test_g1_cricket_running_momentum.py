"""Held-ball root rotation conserves discrete momentum without applying forces."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.running import retarget_running_delivery
from unilab.tasks.manipulation.g1_cricket.running_momentum import HeldBallMomentum

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["right", "left"])
def setup(request, tmp_path):
    scene = tmp_path / "scene.xml"
    hand = request.param
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(
        ROOT / f"g1_cricket_results/running_ballistic_com_v3/{hand}_reference.npz"
    ) as reference:
        poses = reference["qpos"].copy()
    return model, hand, poses


def test_reduced_matrix_matches_native_momentum_and_held_ball_velocity(setup):
    model, hand, poses = setup
    helper = HeldBallMomentum(model, hand)
    pose = poses[13].copy()
    pose[3:7] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_quat()[[3, 0, 1, 2]]
    matrix = helper.matrix(pose)
    rates = np.random.default_rng(7).normal(size=32)
    velocity = helper.mapping @ rates
    data = mujoco.MjData(model)
    data.qpos[:] = helper.data.qpos
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    np.testing.assert_allclose(matrix @ rates, data.subtree_angmom[0], atol=1e-12)
    advanced = data.qpos.copy()
    mujoco.mj_integratePos(model, advanced, velocity, 1e-7)
    data.qpos[:] = advanced
    mujoco.mj_kinematics(model, data)
    held = data.xpos[helper.wrist] + data.xmat[helper.wrist].reshape(3, 3) @ helper.offset
    np.testing.assert_allclose(data.qpos[helper.ball_q : helper.ball_q + 3], held, atol=1e-12)
    np.testing.assert_allclose(
        data.qpos[helper.ball_q + 3 : helper.ball_q + 7], data.xquat[helper.wrist], atol=1e-12
    )


def test_midpoint_rotation_preserves_momentum_without_physics_or_model_edits(setup, monkeypatch):
    model, hand, poses = setup
    helper = HeldBallMomentum(model, hand)
    original = {
        name: getattr(model, name).copy()
        for name in ("body_pos", "body_mass", "body_inertia", "jnt_range", "actuator_forcerange")
    }
    target = helper.measure(poses[10], poses[11], 0.02)

    def forbid_step(*args):
        raise AssertionError("offline momentum solver must not integrate physics")

    monkeypatch.setattr(mujoco, "mj_step", forbid_step)
    before = poses[11].copy()
    for index in range(12, 16):
        joints = poses[index, helper.qa]
        quaternion, omega = helper.advance(before, joints, target, 0.02)
        after = before.copy()
        after[helper.qa] = joints
        after[3:7] = quaternion
        np.testing.assert_allclose(helper.measure(before, after, 0.02), target, atol=1e-8)
        assert np.isfinite(omega).all()
        np.testing.assert_array_equal(after[helper.qa], joints)
        before = after
    assert Rotation.from_quat(before[[4, 5, 6, 3]]).magnitude() > 0.01
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(model, name), value)


def test_rotation_repair_requires_ballistic_com(setup):
    model, hand, _ = setup
    with pytest.raises(ValueError, match="requires a COM target"):
        retarget_running_delivery(model, np.arange(16) * 0.02, hand, conserve_momentum=True)
