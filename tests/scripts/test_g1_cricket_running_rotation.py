"""Temporal refinement preserves source poses and native angular-momentum meaning."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from audit_g1_cricket_running_rotation import ReferenceCurve, momentum_at
from scipy.spatial.transform import Rotation

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("reference_name", ["running_ballistic_com_v3", "running_momentum_v1"])
def test_curve_preserves_all_reference_knots_and_held_ball(hand, reference_name, tmp_path):
    scene = tmp_path / "scene.xml"
    G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
        ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    with np.load(ROOT / f"g1_cricket_results/{reference_name}/{hand}_reference.npz") as source:
        times, poses = source["times"], source["qpos"]
    curve = ReferenceCurve(model, times, poses, hand)
    np.testing.assert_allclose([curve(t) for t in times], poses, atol=1e-12, rtol=0)
    data = mujoco.MjData(model)
    for t in (0.253, 0.567, 0.861, 1.153):
        data.qpos[:] = curve(t)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.subtree_com[0], curve.com(t), atol=1e-12)
        wrist = data.body(f"{hand}_wrist_yaw_link")
        held = wrist.xpos + wrist.xmat.reshape(3, 3) @ curve.offset
        np.testing.assert_allclose(data.body("cricket_ball").xpos, held, atol=1e-12)


@pytest.mark.parametrize("acceleration", [0.0, 3.0])
def test_refined_native_sphere_torque_matches_analytic_result(acceleration):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1" mass="2"/>'
        "</body></worldbody></mujoco>"
    )

    def curve(time):
        angle = 2 * time + 0.5 * acceleration * time**2
        quaternion = Rotation.from_rotvec([0, angle, 0]).as_quat()[[3, 0, 1, 2]]
        return np.r_[[10 + time, 20, 30], quaternion]

    for step in (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625):
        for velocity_step in (1e-4, 1e-5):
            torque = (
                momentum_at(model, curve, 0.5 + step, velocity_step)
                - momentum_at(model, curve, 0.5 - step, velocity_step)
            ) / (2 * step)
            np.testing.assert_allclose(torque, [0, 0.4 * 2 * 0.1**2 * acceleration, 0], atol=1e-9)
