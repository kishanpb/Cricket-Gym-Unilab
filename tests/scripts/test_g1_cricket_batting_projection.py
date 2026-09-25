"""Offline closure preserves the existing bat pose and original robot limits."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from unilab.tasks.manipulation.g1_cricket.batting_projection import (
    BattingProjection,
    rotation_log_jacobian,
)
from unilab.tasks.manipulation.g1_cricket.bimanual_contact import G1BimanualContactCfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["right", "left"])
def projection(request, tmp_path):
    hand = request.param
    scene = tmp_path / "scene.xml"
    G1BimanualContactCfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(ROOT / f"g1_cricket_results/bimanual_grounded_v2/{hand}_reference.npz") as saved:
        poses = saved["qpos"].copy()
    return BattingProjection(model, poses[0], hand), poses


@pytest.mark.parametrize("vector", [[0, 0, 0], [0.2, -0.3, 0.1], [1e-7, 2e-7, -1e-7]])
def test_rotation_log_derivative(vector):
    vector = np.asarray(vector, dtype=float)
    rotation = Rotation.from_rotvec(vector)
    numerical = np.column_stack(
        [
            (
                (Rotation.from_rotvec(axis * 1e-6) * rotation).as_rotvec()
                - (Rotation.from_rotvec(-axis * 1e-6) * rotation).as_rotvec()
            )
            / 2e-6
            for axis in np.eye(3)
        ]
    )
    np.testing.assert_allclose(rotation_log_jacobian(vector), numerical, atol=1e-9)


def test_projection_constraint_jacobian(projection):
    helper, poses = projection
    reference = poses[83]
    value = reference[helper.q] + np.linspace(-0.015, 0.015, 32)
    _, analytical = helper.constraint(value, reference)
    numerical = np.column_stack(
        [
            (
                helper.constraint(value + axis * 1e-6, reference)[0]
                - helper.constraint(value - axis * 1e-6, reference)[0]
            )
            / 2e-6
            for axis in np.eye(32)
        ]
    )
    np.testing.assert_allclose(analytical, numerical, atol=1e-8)


def test_projection_preserves_bat_root_orientation_ball_and_limits(projection):
    helper, poses = projection
    original = poses.copy()
    model_state = (helper.model.body_pos.copy(), helper.model.jnt_range.copy())
    for index in (0, 53, 70, 83, 149):
        pose, audit = helper.project(poses[index])
        assert audit["optimizer_success"], audit
        assert audit["max_constraint_error"] < 1e-8
        untouched = np.setdiff1d(np.arange(helper.model.nq), helper.q)
        np.testing.assert_array_equal(pose[untouched], poses[index, untouched])
        assert (pose[helper.q[:29]] >= helper.joint_limits[:, 0]).all()
        assert (pose[helper.q[:29]] <= helper.joint_limits[:, 1]).all()
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(helper.model.body_pos, model_state[0])
    np.testing.assert_array_equal(helper.model.jnt_range, model_state[1])
