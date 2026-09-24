"""Running reference targets remain separate from achieved physical motion."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image
from retarget_g1_cricket_running import render_review, velocity_reference

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


def test_review_includes_terminal_failure_frames(monkeypatch, tmp_path):
    selected = {}

    class Reader:
        def __init__(self, path):
            self.name = path.name
            selected[self.name] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def count_frames(self):
            return 136 if "offline" in self.name else 32

        def get_data(self, frame):
            selected[self.name].append(frame)
            return np.full((8, 8, 3), frame, dtype=np.uint8)

    monkeypatch.setattr("retarget_g1_cricket_running.imageio.get_reader", Reader)
    render_review(tmp_path)
    for hand in ("right", "left"):
        assert selected[f"{hand}_offline_targets.mp4"] == [0, 30, 60, 83, 91, 135]
        assert selected[f"{hand}_pd_diagnostic.mp4"] == [0, 6, 12, 18, 24, 31]
    with Image.open(tmp_path / "running_motion_review.png") as sheet:
        assert sheet.size == (1440, 644)


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


def test_bowling_windup_stays_in_front_and_moves_forward_at_release():
    targets = RunningDeliveryTargets("right")
    for time in np.linspace(GATHER_TIME, RELEASE_TIME, 63):
        assert targets.arm("right", time)[0][0] > 0
    assert targets.arm_angle.derivative()(RELEASE_TIME) == pytest.approx(10)
    assert targets.flexion(RELEASE_TIME) == pytest.approx(np.radians(8))


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
    times = np.arange(136) * 0.02
    reference = retarget_running_delivery(model, times, hand)
    np.testing.assert_array_equal(reference["times"], times)
    assert [error["time_s"] for error in reference["errors"]] == times.tolist()
    poses = reference["qpos"]
    assert poses.shape == (136, model.nq)
    assert all(error["optimizer_success"] for error in reference["errors"])
    assert not any(error["unexpected_penetrations"] for error in reference["errors"])
    assert max(error["arm_segment_error_m"] for error in reference["errors"]) < 0.006
    assert max(error["foot_error_m"] for error in reference["errors"]) < 0.002
    joints = np.array([model.joint(n).id for n in SDK_JOINTS])
    q = poses[:, model.jnt_qposadr[joints]]
    assert (q >= model.jnt_range[joints, 0]).all()
    assert (q <= model.jnt_range[joints, 1]).all()
    for name, values in original.items():
        np.testing.assert_array_equal(getattr(model, name), values)
    velocity = velocity_reference(model, poses, 0.02)
    assert np.isfinite(velocity).all()
    assert np.max(np.abs(velocity[:, model.jnt_dofadr[joints]])) <= 12 + 1e-8
    assert velocity[0, 0] > 0.9
    for i in range(len(poses) - 1):
        integrated = poses[i].copy()
        mujoco.mj_integratePos(model, integrated, velocity[i], 0.02)
        np.testing.assert_allclose(integrated, poses[i + 1], atol=1e-12)
