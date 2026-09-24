"""Constraint-force interpretation and immutable diagnostic outputs."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_tracking import visual_model
from retarget_g1_cricket_batting import grip_force, main

from unilab.tasks.manipulation.g1_cricket.bimanual import build_bimanual_scene


def test_connect_force_norm_and_inactive_rows():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody>
        <site name="anchor" pos="0 0 1"/>
        <body pos="0 0 1"><freejoint/>
          <geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
          <site name="grip"/>
        </body>
      </worldbody>
      <equality><connect site1="grip" site2="anchor" solref=".012 1"/></equality>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qfrc_applied[0] = 2
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert grip_force(data, 0) == pytest.approx(2, abs=1e-5)
    constraint = np.empty(model.nv)
    mujoco.mj_mulJacTVec(model, data, constraint, data.efc_force)
    np.testing.assert_allclose(constraint, data.qfrc_constraint, atol=1e-10)
    assert constraint[0] < 0
    data.eq_active[0] = 0
    mujoco.mj_forward(model, data)
    with pytest.raises(RuntimeError, match="three constraint rows"):
        grip_force(data, 0)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["retarget", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("hand", ["right", "left"])
def test_visual_playback_restores_meshes_without_changing_state(tmp_path, hand):
    root = Path(__file__).resolve().parents[2]
    scene = tmp_path / "scene.xml"
    build_bimanual_scene(root / "src/unilab/assets/robots/g1/g1.xml", scene, hand)
    spec = mujoco.MjSpec.from_file(str(scene))
    spec.compiler.discardvisual = True
    physics = spec.compile()
    free_bodies = physics.jnt_bodyid[physics.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
    physics.body_pos[free_bodies] += 3e-7
    restored = visual_model(scene, physics)
    assert physics.nmesh == 0 and restored.nmesh > 0
    np.testing.assert_array_equal(restored.body_pos, physics.body_pos)
    actual, display = mujoco.MjData(physics), mujoco.MjData(restored)
    mujoco.mj_resetDataKeyframe(physics, actual, 0)
    display.qpos[:] = actual.qpos
    mujoco.mj_forward(physics, actual)
    mujoco.mj_forward(restored, display)
    np.testing.assert_allclose(actual.xpos, display.xpos, atol=1e-12)
    physics.body_pos[physics.body("left_elbow_link").id, 0] += 3e-7
    with pytest.raises(AssertionError):
        visual_model(scene, physics)
