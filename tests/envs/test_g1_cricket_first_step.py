from pathlib import Path

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_first_step import FirstStepReplay, evaluation_override
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

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


def test_uniform_config_changes_only_training_sampling_and_eval_starts_at_zero():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        parent = compose("config", overrides=["task=g1_cricket_first_step/mjbatch"])
        uniform = compose("config", overrides=["task=g1_cricket_first_step_uniform/mjbatch"])
    original = OmegaConf.to_container(uniform, resolve=True)
    assert uniform.env.commands.motion.params.sampling_mode == "uniform"
    restored = OmegaConf.create(original)
    restored.env.commands.motion.params.sampling_mode = "start"
    assert OmegaConf.to_container(restored, resolve=True) == OmegaConf.to_container(
        parent, resolve=True
    )
    registry.ensure_registries()
    instance = registry.make(
        uniform.training.task_name,
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=evaluation_override(uniform, 0.00003125),
    )
    try:
        for seed in (1, 2, 3):
            instance.reset(seed=seed)
            np.testing.assert_array_equal(instance.command_manager.get_term("motion").time_steps, 0)
            np.testing.assert_array_equal(instance.episode_length_buf, 0)
        assert instance.cfg.sim_dt == 0.00003125
        assert not instance.cfg.auto_reset
    finally:
        instance.close()
    assert OmegaConf.to_container(uniform, resolve=True) == original


def test_sampled_reset_matches_wrist_kinematics_and_preserves_other_row(env, monkeypatch):
    command = env.command_manager.get_term("motion")
    model = env.get_playback_model()
    data = mujoco.MjData(model)
    wrist = model.body(f"{env.cfg.handedness}_wrist_yaw_link").id
    ball = model.body("cricket_ball").id
    velocity = np.empty(6)
    ball_velocity = np.empty(6)
    for frame in (0, 100, 200, 225, 250, 275, 300, 325, 400, 500, 525):
        monkeypatch.setattr(command.sampler, "sample_frames", lambda ids: np.full(len(ids), frame))
        # The sampler normally owns time_steps as well as returning its frames.
        command.time_steps[0] = frame
        untouched = env.get_physics_state_snapshot()[1].copy()
        env.reset(env_ids=np.array([0], dtype=np.int32))
        state = env.get_physics_state_snapshot()
        np.testing.assert_array_equal(state[1], untouched)
        mujoco.mj_setState(model, data, state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        offset = data.xmat[wrist].reshape(3, 3) @ command.offset
        np.testing.assert_allclose(data.xpos[ball], data.xpos[wrist] + offset, atol=1e-6, rtol=0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, wrist, velocity, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, ball, ball_velocity, 0)
        np.testing.assert_allclose(ball_velocity[:3], velocity[:3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            ball_velocity[3:], velocity[3:] + np.cross(velocity[:3], offset), atol=1e-6, rtol=0
        )
        assert env.equality_constraints.get_equality_active()[0].all()


def test_uniform_sampling_covers_all_phases_and_motor_steps_replay(env):
    command = env.command_manager.get_term("motion")
    command.sampler.mode = "uniform"
    frames = []
    for seed in range(128):
        env.reset(seed=seed)
        frames.extend(command.time_steps.copy())
        np.testing.assert_allclose(
            env.scene["robot"].data.joint_pos, command.joint_pos, atol=1e-6, rtol=0
        )
        if seed % 8 == 0:
            driver = FirstStepReplay(env, lambda: np.zeros((2, 29), dtype=np.float32))
            model = env.get_playback_model()
            data = mujoco.MjData(model)
            driver.initialize(model, data)
            data.ctrl[:] = driver.begin(model, data)
            for _ in range(env.cfg.sim_substeps):
                mujoco.mj_step(model, data)
            driver.finish(model, data)
            assert np.isfinite(env.get_physics_state_snapshot()).all()
    assert set(np.asarray(frames) // 50) == set(range(11))
