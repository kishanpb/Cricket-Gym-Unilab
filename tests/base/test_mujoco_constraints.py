"""Release changes a constraint, not the free ball's integrated state."""

import mujoco
import numpy as np
import pytest

from unilab.base.backend_constraints import EqualityConstraintBackend
from unilab.base.mujoco_substeps import SubstepMuJoCoBackend
from unilab.base.np_env import NpEnv
from unilab.base.scene import SceneCfg


@pytest.mark.parametrize("engine", ["rollout", "mjbatch"])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("observe", [False, True])
def test_release_and_partial_reset_match_direct_physics(tmp_path, engine, side, observe):
    if engine == "mjbatch":
        pytest.importorskip("mjbatch.held_control")
    path = tmp_path / "holder.xml"
    path.write_text(f"""
    <mujoco><option timestep=".001" integrator="implicitfast"/>
      <worldbody>
        <body name="wrist" pos="0 {side * 0.08} 1.2">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="box" size=".04 .03 .03" mass=".7" contype="0" conaffinity="0"/>
        </body>
        <body name="ball" pos="0 {side * 0.08} 1.11"><freejoint/>
          <geom type="sphere" size=".036" mass=".156"/>
        </body>
      </worldbody>
      <equality>
        <weld name="holder" body1="wrist" body2="ball" solref=".004 1"/>
        <joint name="unused" joint1="slide" active="false"/>
      </equality>
      <actuator><motor joint="slide"/></actuator>
      <sensor><framelinvel name="velocity" objtype="body" objname="ball"/></sensor>
    </mujoco>
    """)
    backend = SubstepMuJoCoBackend(
        SceneCfg(model_file=str(path)),
        3,
        0.001,
        adaptive_chunk_size=False,
        substep_engine=engine,
    )
    backend.materialize()
    assert isinstance(backend, EqualityConstraintBackend)
    model = backend.get_playback_model()
    data = mujoco.MjData(model)
    qpos = np.tile(model.qpos0, (3, 1))
    qvel = np.zeros((3, model.nv))
    qvel[:, [0, 1]] = 1.2
    backend.set_state(np.arange(3), qpos, qvel)
    samples = []
    if observe:
        backend.set_substep_observer(
            ("velocity",), "ball", lambda sensors, velocity: samples.append(sensors.copy())
        )
    full = mujoco.mjtState.mjSTATE_FULLPHYSICS
    try:
        assert backend.get_equality_names() == ("holder", "unused")
        snapshot = backend.get_equality_active()
        snapshot[:] = False
        np.testing.assert_array_equal(backend.get_equality_active(), [[True, False]] * 3)
        with pytest.raises(ValueError, match="boolean"):
            backend.set_equality_active(np.array([0]), np.array([[0.5]]))
        with pytest.raises(ValueError, match="constraints"):
            backend.set_equality_active(np.array([0]), np.zeros((1, 3), dtype=bool))
        for tick in range(6):
            if tick == 1:
                before = backend.get_physics_state().copy()
                sensed = backend._sensor_data.copy()
                backend.set_equality_active(np.array([1, 2]), np.zeros((2, 2), dtype=bool))
                np.testing.assert_array_equal(backend.get_physics_state(), before)
                np.testing.assert_array_equal(backend._sensor_data, sensed)
            if tick == 3:
                untouched = backend.get_physics_state()[[0, 2]].copy()
                backend.set_state(np.array([1]), qpos[1:2], qvel[1:2])
                np.testing.assert_array_equal(backend.get_physics_state()[[0, 2]], untouched)
            active = backend.get_equality_active()
            assert not active[:, 1].any()
            np.testing.assert_array_equal(active[:, 0], [True, tick == 0 or tick >= 3, tick == 0])
            initial = backend.get_physics_state().copy()
            force = np.zeros((3, 1, 3))
            force[:, 0, 0] = 0.1 if tick == 2 else 0
            backend.apply_body_force(np.array([1]), force, np.zeros_like(force))
            backend.step(np.full((3, 1), 0.2), 40)
            for row in range(3):
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, initial[row], full)
                data.eq_active[:] = active[row]
                data.ctrl[:] = 0.2
                data.xfrc_applied[1, :3] = force[row, 0]
                for step in range(40):
                    mujoco.mj_step(model, data)
                    if observe:
                        np.testing.assert_array_equal(samples[-1][row, step], data.sensordata[:3])
                    if not active[row, 0]:
                        assert data.nefc == 0
                        np.testing.assert_array_equal(data.qfrc_constraint[1:], 0)
                expected = np.empty(initial.shape[1])
                mujoco.mj_getState(model, data, expected, full)
                np.testing.assert_array_equal(
                    backend.get_physics_state()[row], expected.astype(initial.dtype)
                )
                np.testing.assert_array_equal(
                    backend._sensor_data[row], data.sensordata.astype(backend._sensor_data.dtype)
                )
                if active[row, 0]:
                    assert abs(data.qpos[3] - 1.11) < 0.001
                else:
                    assert data.qvel[3] < -0.3
                    assert data.qvel[1] > 1
            assert not backend._pending_xfrc_applied.any()
    finally:
        backend.cleanup_scene_assets()


def test_env_constraint_capability_is_explicit():
    from types import SimpleNamespace

    with pytest.raises(NotImplementedError, match="equality activation"):
        NpEnv.equality_constraints.fget(SimpleNamespace(_backend=object()))
    backend = object.__new__(SubstepMuJoCoBackend)
    assert NpEnv.equality_constraints.fget(SimpleNamespace(_backend=backend)) is backend
