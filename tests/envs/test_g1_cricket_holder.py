"""The G1 holder clears both hands and releases only an equality constraint."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from unilab.tasks.manipulation.g1_cricket.holder import build_holder_scene
from unilab.tasks.manipulation.g1_cricket.prior import SDK_DEFAULT

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("hand", ["left", "right"])
def test_holder_preserves_robot_clearance_and_ball_state(tmp_path, hand):
    source = ROOT / "src/unilab/assets/robots/g1/g1.xml"
    original = source.read_bytes()
    path = tmp_path / "holder.xml"
    build_holder_scene(source, path, hand)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY], 0, atol=1e-12
    )
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert source.read_bytes() == original
    assert (model.nq, model.nv, model.nu, model.neq) == (43, 41, 29, 1)
    assert not model.body_gravcomp.any()
    np.testing.assert_array_equal(data.qpos[7:36], SDK_DEFAULT)
    ball = model.body("cricket_ball")
    geom = ball.geomadr[0]
    joint = ball.jntadr[0]
    qadr = model.jnt_qposadr[joint]
    vadr = model.jnt_dofadr[joint]
    assert model.body_mass[ball.id] == pytest.approx(0.156)
    assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
    assert model.geom_size[geom, 0] == pytest.approx(0.036)
    for other in range(model.ngeom):
        if other != geom and (model.geom_contype[other] or model.geom_conaffinity[other]):
            assert mujoco.mj_geomDistance(model, data, geom, other, 10, None) > 0.003
    eq_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    assert eq_rows.sum() == 6
    np.testing.assert_allclose(data.efc_pos[eq_rows], 0, atol=1e-12)
    assert data.eq_active[0]
    # Give the whole configuration a shared velocity, then let physics act.
    data.qvel[:3] = [0.4, 0, 0.1]
    data.qvel[vadr : vadr + 3] = [0.4, 0, 0.1]
    data.ctrl[:] = SDK_DEFAULT
    data.ctrl[model.actuator(f"{hand}_elbow_joint").id] += 0.1
    for _ in range(40):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.qfrc_constraint[vadr : vadr + 3]) > 0.1
    before = np.concatenate((data.qpos, data.qvel))
    position = data.qpos[qadr : qadr + 3].copy()
    velocity = data.qvel[vadr : vadr + 3].copy()
    data.eq_active[0] = False
    np.testing.assert_array_equal(np.concatenate((data.qpos, data.qvel)), before)
    for step in range(1, 11):
        mujoco.mj_step(model, data)
        assert not np.any(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        np.testing.assert_array_equal(data.qfrc_constraint[vadr : vadr + 6], 0)
        dt = model.opt.timestep
        np.testing.assert_allclose(
            data.qvel[vadr : vadr + 3], velocity + model.opt.gravity * step * dt, atol=1e-12
        )
        np.testing.assert_allclose(
            data.qpos[qadr : qadr + 3],
            position + velocity * step * dt + model.opt.gravity * dt**2 * step * (step + 1) / 2,
            atol=1e-12,
        )


@pytest.mark.parametrize("hand", ["left", "right"])
def test_g1_release_trajectory_matches_native_batch(tmp_path, hand):
    pytest.importorskip("mjbatch.held_control")
    from mjbatch.held_control import FULL, HeldControlRollout

    path = tmp_path / "holder.xml"
    build_holder_scene(ROOT / "src/unilab/assets/robots/g1/g1.xml", path, hand)
    model = mujoco.MjModel.from_xml_path(str(path))
    model.opt.timestep = 0.00025
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:3] = [0.4, 0, 0.1]
    ball_dof = model.jnt_dofadr[model.body("cricket_ball").jntadr[0]]
    data.qvel[ball_dof : ball_dof + 3] = [0.4, 0, 0.1]
    for _ in range(40):
        mujoco.mj_step(model, data)
    seed = np.empty(mujoco.mj_stateSize(model, FULL))
    mujoco.mj_getState(model, data, seed, FULL)
    initial = np.tile(seed, (2, 1))
    spec = int(mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE)
    control = np.zeros((2, 40, model.nu + model.neq))
    control[:, :, : model.nu] = SDK_DEFAULT
    control[0, :, -1] = 1
    recorder = HeldControlRollout([model] * 2)
    try:
        for _ in range(3):
            states, sensors = recorder.rollout(
                [model] * 2, [data], initial, control, control_spec=spec, nstep=40
            )
            for row in range(2):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, initial[row], FULL)
                mujoco.mj_setState(model, data, control[row, 0], spec)
                for step in range(40):
                    mujoco.mj_step(model, data)
                    expected = np.empty_like(seed)
                    mujoco.mj_getState(model, data, expected, FULL)
                    np.testing.assert_array_equal(states[row, step], expected)
                    np.testing.assert_array_equal(sensors[row, step], data.sensordata)
                    if row == 1:
                        assert not np.any(data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                        np.testing.assert_array_equal(
                            data.qfrc_constraint[ball_dof : ball_dof + 6], 0
                        )
            assert not np.array_equal(states[0], states[1])
            initial = states[:, -1].copy()
    finally:
        recorder.close()
