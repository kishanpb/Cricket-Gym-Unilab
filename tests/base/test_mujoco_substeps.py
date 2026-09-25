"""Experimental recorder must observe, not change, the held-control dynamics."""

import mujoco
import numpy as np
import pytest
from unisim.backend.mujoco.backend import MuJoCoBackend

from unilab.base.backend_factory import create_backend, env_backend_kwargs
from unilab.base.base import EnvCfg
from unilab.base.mujoco_substeps import SubstepMuJoCoBackend
from unilab.base.scene import SceneCfg

XML = """
<mujoco><option integrator="implicitfast" gravity="0 0 0"/>
  <worldbody>
    <geom name="blade" type="box" size=".02 .2 .2"/>
    <body name="ball" pos=".055 0 0"><freejoint/>
      <geom name="ball_geom" type="sphere" size=".036" mass=".156"/>
    </body>
  </worldbody>
  <contact><pair geom1="blade" geom2="ball_geom" solref=".004 1"/></contact>
  <sensor>
    <contact name="hit" geom1="ball_geom" geom2="blade" num="4"
      data="found force torque dist pos normal tangent"/>
    <framelinvel name="velocity" objtype="body" objname="ball"/>
  </sensor>
</mujoco>
"""


def make_backends(tmp_path, **options):
    path = tmp_path / "contact.xml"
    path.write_text(XML)
    backends = [
        cls(SceneCfg(model_file=str(path)), 2, 0.00025, **options)
        for cls in (MuJoCoBackend, SubstepMuJoCoBackend)
    ]
    for backend in backends:
        backend.materialize()
        models = backend._pool.get_all_models()
        models[1].body_mass[1] *= 1.7
        mujoco.mj_setConst(models[1], mujoco.MjData(models[1]))
        qpos = np.tile(models[0].qpos0, (2, 1))
        qpos[1, 0] = 0.09
        qvel = np.zeros((2, models[0].nv))
        qvel[:, 0] = -2.5
        backend.set_state(np.arange(2), qpos, qvel)
    return backends


@pytest.mark.parametrize("nsteps", [1, 80, 160])
def test_recorded_path_exactly_matches_native_steps(tmp_path, nsteps):
    old, new = make_backends(tmp_path)
    samples = []

    def observe(sensors, velocity):
        assert not sensors.flags.writeable and not velocity.flags.writeable
        samples.append((sensors.copy(), velocity.copy()))

    new.set_substep_observer(("hit", "velocity"), "ball", observe)
    try:
        for tick in range(4):
            if tick == 1:
                for backend in (old, new):
                    backend.apply_body_force(
                        np.array([1]), np.full((2, 1, 3), 0.3), np.full((2, 1, 3), 0.01)
                    )
            if tick == 2:
                for backend in (old, new):
                    model = backend._pool.get_model(1)
                    velocity = np.zeros((1, model.nv))
                    velocity[0, 0] = -2
                    backend.set_state(np.array([1]), model.qpos0[None], velocity)
            old.step(np.empty((2, 0)), nsteps)
            new.step(np.empty((2, 0)), nsteps)
            np.testing.assert_array_equal(new.get_physics_state(), old.get_physics_state())
            np.testing.assert_array_equal(new._sensor_data, old._sensor_data)
            assert not new._pending_xfrc_applied.any() and not old._pending_xfrc_applied.any()
            sensors, velocity = samples[-1]
            assert sensors.shape == (2, nsteps, 71) and velocity.shape == (2, nsteps, 3)
            np.testing.assert_array_equal(sensors[:, -1].astype(np.float32), new._sensor_data)
            np.testing.assert_array_equal(
                velocity[:, -1].astype(np.float32), new.get_physics_state()[:, 8:11]
            )
        assert len(samples) == 4
        assert any((s[0][:, :, :68].reshape(2, nsteps, 4, 17)[..., 1] > 0).any() for s in samples)
        assert any(not np.array_equal(s[0][:, :, 68:], s[1]) for s in samples)
    finally:
        old.cleanup_scene_assets()
        new.cleanup_scene_assets()


def test_unobserved_adapter_keeps_default_path_and_rejects_control_callback(tmp_path):
    old, new = make_backends(tmp_path)
    try:
        old.step(np.empty((2, 0)), 80)
        new.step(np.empty((2, 0)), 80)
        np.testing.assert_array_equal(old.get_physics_state(), new.get_physics_state())
        with pytest.raises(NotImplementedError, match="held control"):
            new.set_pre_step_control(lambda backend, ctrl: ctrl)
        new.set_substep_observer(("hit",), "ball", lambda sensors, velocity: None)
        with pytest.raises(ValueError, match="only one"):
            new.set_substep_observer(("hit",), "ball", lambda sensors, velocity: None)
    finally:
        old.cleanup_scene_assets()
        new.cleanup_scene_assets()


def test_factory_is_opt_in_and_rejects_unsupported_combinations(tmp_path):
    path = tmp_path / "contact.xml"
    path.write_text(XML)
    cfg = EnvCfg(scene=SceneCfg(model_file=str(path)), adaptive_chunk_size=False)
    assert "mujoco_observe_substeps" not in env_backend_kwargs(cfg)
    cfg.mujoco_observe_substeps = True
    backend = create_backend("mujoco", cfg.scene, 1, 0.00025, **env_backend_kwargs(cfg))
    try:
        assert isinstance(backend, SubstepMuJoCoBackend)
        backend.materialize()
    finally:
        backend.cleanup_scene_assets()
    with pytest.raises(ValueError, match="MuJoCo-only"):
        create_backend("motrix", cfg.scene, 1, 0.00025, **env_backend_kwargs(cfg))
    cfg.post_step_forward_sensor = True
    backend = create_backend("mujoco", cfg.scene, 1, 0.00025, **env_backend_kwargs(cfg))
    try:
        with pytest.raises(ValueError, match="solved sensors"):
            backend.materialize()
    finally:
        backend.cleanup_scene_assets()


@pytest.mark.parametrize("group_identical_models", [False, True])
def test_mjbatch_selection_without_observer_and_reset_model_mutation(
    tmp_path, group_identical_models
):
    pytest.importorskip("mjbatch.held_control")
    from mjbatch.held_control import HeldControlRollout
    from unisim.dr.types import ResetRandomizationPayload

    path = tmp_path / "native.xml"
    path.write_text(XML)
    cfg = EnvCfg(scene=SceneCfg(model_file=str(path)), adaptive_chunk_size=False)
    cfg.mujoco_group_identical_models = group_identical_models
    if group_identical_models:
        with pytest.raises(ValueError, match="grouping requires mjbatch"):
            cfg.validate()
    cfg.mujoco_substep_engine = "wrong"
    with pytest.raises(ValueError, match="mujoco_substep_engine"):
        cfg.validate()
    cfg.mujoco_substep_engine = "mjbatch"
    with pytest.raises(ValueError, match="requires mujoco_observe_substeps"):
        cfg.validate()
    cfg.mujoco_observe_substeps = True
    cfg.validate()
    backends = [
        MuJoCoBackend(cfg.scene, 3, 0.00025, adaptive_chunk_size=False),
        create_backend("mujoco", cfg.scene, 3, 0.00025, **env_backend_kwargs(cfg)),
    ]
    try:
        for backend in backends:
            backend.materialize()
        assert isinstance(backends[1]._recorder, HeldControlRollout)
        assert len(backends[1]._recorder.groups) == (1 if group_identical_models else 3)
        for backend in backends:
            backend.step(np.empty((3, 0)), 80)
        np.testing.assert_array_equal(
            backends[0].get_physics_state(), backends[1].get_physics_state()
        )
        np.testing.assert_array_equal(backends[0]._sensor_data, backends[1]._sensor_data)
        model = backends[1].get_playback_model()
        with pytest.raises(NotImplementedError, match="reset randomization"):
            backends[1].set_state(
                np.array([0]),
                model.qpos0[None],
                np.zeros((1, model.nv)),
                randomization=ResetRandomizationPayload(body_mass=model.body_mass[None]),
            )
    finally:
        for backend in backends:
            backend.cleanup_scene_assets()
    with pytest.raises(ValueError, match="MuJoCo-only"):
        create_backend("motrix", cfg.scene, 1, 0.00025, **env_backend_kwargs(cfg))
    cfg.post_step_forward_sensor = True
    backend = create_backend("mujoco", cfg.scene, 1, 0.00025, **env_backend_kwargs(cfg))
    try:
        with pytest.raises(ValueError, match="solved sensors"):
            backend.materialize()
    finally:
        backend.cleanup_scene_assets()
