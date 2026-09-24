"""Running reference targets remain separate from achieved physical motion."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from retarget_g1_cricket_running import velocity_reference

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running import (
    END_TIME,
    GATHER_TIME,
    RELEASE_TIME,
    RunningDeliveryTargets,
    retarget_running_delivery,
)

ROOT = Path(__file__).resolve().parents[2]


def test_targets_mirror_hands_and_keep_delivery_foot_behind_crease():
    right, left = RunningDeliveryTargets("right"), RunningDeliveryTargets("left")
    mirror = np.array([1, -1, 1])
    for time in np.linspace(0, END_TIME, 271):
        np.testing.assert_allclose(left.root(time), right.root(time) * mirror)
        for side, other in (("right", "left"), ("left", "right")):
            np.testing.assert_allclose(left.foot(side, time), right.foot(other, time) * mirror)
            lu, ll, lr = left.arm(side, time)
            ru, rl, rr = right.arm(other, time)
            np.testing.assert_allclose(lu, ru * mirror)
            np.testing.assert_allclose(ll, rl * mirror)
            np.testing.assert_allclose(lr, np.diag(mirror) @ rr @ np.diag(mirror))
            np.testing.assert_allclose(lr.T @ lr, np.eye(3), atol=1e-14)
    assert right.foot("left", 1.65)[0] < 0
    assert right.foot("right", 1.42)[0] < right.foot("left", 1.65)[0]
    assert right.root(END_TIME)[0] - right.root(0)[0] == pytest.approx(2.2)
    assert any(
        min(right.foot(s, t)[2] for s in ("right", "left")) > 0.04
        for t in np.linspace(0, GATHER_TIME, 121)
    )


@pytest.mark.parametrize("side", ["right", "left"])
def test_arm_targets_are_continuous_at_gather(side):
    targets = RunningDeliveryTargets("right")
    before, after = targets.arm(side, GATHER_TIME - 1e-8), targets.arm(side, GATHER_TIME)
    for a, b in zip(before, after, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    assert targets.arm("right", RELEASE_TIME)[0][2] > 0.8


@pytest.mark.parametrize("hand", ["right", "left"])
def test_retarget_preserves_robot_and_exports_full_body_velocity(hand, tmp_path):
    scene = tmp_path / "scene.xml"
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    original = {
        name: getattr(model, name).copy()
        for name in ("body_pos", "body_mass", "body_inertia", "jnt_range", "actuator_forcerange")
    }
    times = np.arange(4) * 0.02
    reference = retarget_running_delivery(model, times, hand)
    poses = reference["qpos"]
    assert poses.shape == (4, model.nq)
    joints = np.array([model.joint(n).id for n in SDK_JOINTS])
    q = poses[:, model.jnt_qposadr[joints]]
    assert (q >= model.jnt_range[joints, 0]).all()
    assert (q <= model.jnt_range[joints, 1]).all()
    for name, values in original.items():
        np.testing.assert_array_equal(getattr(model, name), values)
    velocity = velocity_reference(model, poses, 0.02)
    assert np.isfinite(velocity).all()
    assert np.max(np.abs(velocity[:, model.jnt_dofadr[joints]])) <= 12 + 1e-8
    assert (velocity[:, 0] > 0.9).all()
    for i in range(3):
        integrated = poses[i].copy()
        mujoco.mj_integratePos(model, integrated, velocity[i], 0.02)
        np.testing.assert_allclose(integrated, poses[i + 1], atol=1e-12)
