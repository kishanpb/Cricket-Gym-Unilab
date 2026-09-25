from pathlib import Path

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_first_step import FirstStepReplay
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.running_task import RunningSupportObservation

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["right", "left"])
def env(request):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=["task=g1_cricket_first_step/mjbatch", f"env.handedness={request.param}"],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    instance = registry.make(
        owner.training.task_name, num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        instance.reset(seed=1)
        yield instance
    finally:
        instance.close()


def test_first_step_uses_declared_support_torque_and_never_releases(env):
    action = env.action_manager.get_term("reference")
    command = env.command_manager.get_term("motion")
    model = env.get_playback_model()
    with np.load(action.cfg.reference_file) as saved:
        np.testing.assert_array_equal(
            action.bias_offset, saved["torque"] / model.actuator_gainprm[:, 0]
        )
    assert action.action_dim == 29
    assert env.max_episode_length == 525
    assert action.cfg.scale == 0.1
    state = env.get_physics_state_snapshot().copy()
    for frame in (0, 91, 225, 325, 525):
        command.time_steps[:] = frame
        command._refresh_motion()
        action.process_actions(np.zeros((2, 29), dtype=np.float32))
        assert not action.released.any()
        assert not action.just_released.any()
        np.testing.assert_array_equal(env.equality_constraints.get_equality_active(), True)
        np.testing.assert_array_equal(env.get_physics_state_snapshot(), state)


def test_motor_step_has_exact_native_replay_and_observed_foot_loads(env, monkeypatch):
    def forbid_write(*args, **kwargs):
        raise AssertionError("first-step controller must use motors, not state writes")

    replay = DeliveryReplay(env, action_name="reference")
    events = DeliveryEvents(env.cfg.handedness)
    monkeypatch.setattr(Entity, "write_root_state_to_sim", forbid_write)
    monkeypatch.setattr(Entity, "write_joint_state_to_sim", forbid_write)
    state = replay.step(env, np.zeros((2, 29), dtype=np.float32), events)
    assert not state.terminated.any()
    observation = RunningSupportObservation(None, env)(env)
    assert observation.shape == (2, 7)
    assert np.isfinite(observation).all()
    np.testing.assert_array_equal(observation[:, [0, 3]], 1)
    assert np.all(observation[:, [1, 4]] > 1)
    np.testing.assert_allclose(observation[:, -1], 1 / 525)
    assert not env.action_manager.get_term("reference").released.any()
    np.testing.assert_array_equal(replay.data.xfrc_applied, 0)
    np.testing.assert_array_equal(replay.data.qfrc_applied, 0)


def test_first_step_driver_compares_full_native_state_and_sensors(env):
    driver = FirstStepReplay(env, lambda: np.zeros((2, 29), dtype=np.float32))
    model = env.get_playback_model()
    data = mujoco.MjData(model)
    driver.initialize(model, data)
    data.ctrl[:] = driver.begin(model, data)
    for _ in range(env.cfg.sim_substeps):
        mujoco.mj_step(model, data)
    driver.finish(model, data)
    assert len(driver.states) == 2
    assert len(driver.controls) == len(driver.actions) == 1
    assert not driver.done
    data.qpos[0] += 0.001
    with pytest.raises(AssertionError):
        driver.finish(model, data)
