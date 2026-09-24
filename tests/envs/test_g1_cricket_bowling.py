"""Bowling learns arm targets/release; the fixture never supplies launch velocity."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import SDK_DEFAULT

ROOT = Path(__file__).resolve().parents[2]
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


def config(engine):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_bowling_v1/{engine}"])


def make_env(engine="mujoco", hand="right", nenv=2):
    if engine == "mjbatch":
        pytest.importorskip("mjbatch.held_control")
    cfg = config(engine)
    override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    env = create_env(cfg, num_envs=nenv, env_cfg_override=override)
    env.reset(seed=5301)
    return env


def test_owner_changes_only_executor():
    configs = [
        OmegaConf.to_container(config(engine), resolve=True) for engine in ("mujoco", "mjbatch")
    ]
    assert configs[1]["env"].pop("mujoco_substep_engine") == "mjbatch"
    assert configs[0] == configs[1]


@pytest.mark.parametrize("hand", ["right", "left"])
def test_fixture_force_sign_and_free_release(hand):
    env = make_env(hand=hand)
    try:
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        ball = model.body("cricket_ball").id
        dof = model.jnt_dofadr[model.body_jntadr[ball]]
        site = model.site("holder_site").id
        assert model.body("holder_loadcell").mass[0] == 0
        assert (model.nq, model.nv, model.nu, model.neq) == (43, 41, 29, 1)
        assert model.body_mass[ball] == pytest.approx(0.156)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "holder_torque") == -1
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = SDK_DEFAULT
        data.ctrl[model.actuator(f"{hand}_elbow_joint").id] += 0.1
        for _ in range(40):
            mujoco.mj_step(model, data)
            force = data.site_xmat[site].reshape(3, 3) @ data.sensor("holder_force").data
            np.testing.assert_allclose(force, data.qfrc_constraint[dof : dof + 3], atol=1e-10)
            assert not data.sensor("ball_hand").data.reshape(4, 17)[:, 0].any()
        assert np.linalg.norm(force) > 0.1
        before = np.r_[data.qpos, data.qvel]
        data.eq_active[0] = False
        np.testing.assert_array_equal(np.r_[data.qpos, data.qvel], before)
        velocity = data.qvel[dof : dof + 3].copy()
        for step in range(1, 11):
            mujoco.mj_step(model, data)
            np.testing.assert_array_equal(data.sensor("holder_force").data, 0)
            np.testing.assert_array_equal(data.qfrc_constraint[dof : dof + 6], 0)
            np.testing.assert_allclose(
                data.qvel[dof : dof + 3],
                velocity + model.opt.gravity * step * model.opt.timestep,
                atol=1e-12,
            )
    finally:
        env.close()


@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
@pytest.mark.parametrize("hand", ["right", "left"])
def test_release_latch_integrated_snapshot_and_partial_reset(engine, hand):
    env = make_env(engine, hand)
    try:
        term = env.action_manager.get_term("residual")
        action = np.zeros((2, 8), np.float32)
        action[:, 7] = 0.5
        env.step(action)
        assert not term.released.any()
        assert term.peak_load.min() > 0.1
        np.testing.assert_array_equal(term.touch_fraction, 0)
        before = env.get_physics_state_snapshot().copy()
        term.process_actions(action)
        targets_before_release = term.processed_action.copy()
        action[0, 7] = 0.5001
        term.process_actions(action)
        np.testing.assert_array_equal(term.processed_action, targets_before_release)
        held_target = term.processed_action[1].copy()
        np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
        np.testing.assert_array_equal(term.release_physics[0], before[0])
        np.testing.assert_array_equal(
            term.release_position[0], before[0, term.qadr : term.qadr + 3]
        )
        np.testing.assert_array_equal(
            term.release_velocity[0], before[0, term.vadr : term.vadr + 3]
        )
        np.testing.assert_array_equal(term.released, [True, False])
        np.testing.assert_array_equal(
            env.equality_constraints.get_equality_active(), [[False], [True]]
        )
        term.process_actions(np.zeros((2, 8), np.float32))
        np.testing.assert_array_equal(term.processed_action[1], held_target)
        assert not term.just_released.any()
        np.testing.assert_array_equal(term.release_physics[0], before[0])
        env.step(np.zeros((2, 8), np.float32))
        np.testing.assert_array_equal(term.peak_load[0], 0)
        np.testing.assert_array_equal(term.impulse_world[0], 0)
        survivor = env.get_physics_state_snapshot()[0].copy()
        survivor_action = term.raw_action[0].copy()
        env.reset(env_ids=np.array([1]))
        np.testing.assert_array_equal(env.get_physics_state_snapshot()[0], survivor)
        np.testing.assert_array_equal(term.raw_action[0], survivor_action)
        np.testing.assert_array_equal(term.released, [True, False])
        env.reset(env_ids=np.array([0]))
        assert not term.released.any()
        np.testing.assert_array_equal(term.release_physics, 0)
        assert env.equality_constraints.get_equality_active().all()
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        for state in env.get_physics_state_snapshot():
            mujoco.mj_setState(model, data, state, FULL)
            mujoco.mj_forward(model, data)
            rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            assert rows.sum() == 6
            np.testing.assert_allclose(data.efc_pos[rows], 0, atol=2e-7)
    finally:
        env.close()


def test_interval_impulse_frame_touch_occupancy_and_invalid_signals():
    env = make_env()
    try:
        term = env.action_manager.get_term("residual")
        sensors = np.zeros((2, 4, 7 + 4 * 17))
        sensors[..., 3] = np.sqrt(0.5)
        sensors[..., 6] = np.sqrt(0.5)
        sensors[..., 0] = 2
        sensors[1, 0, 7] = 1
        term.observe(sensors, None)
        np.testing.assert_allclose(term.peak_load, 2)
        np.testing.assert_allclose(term.impulse_world, [[0, 8 * env.physics_dt, 0]] * 2, atol=1e-15)
        np.testing.assert_array_equal(term.touch_fraction, [0, 0.25])
        sensors[..., :3] = 0
        sensors[..., 7:] = 0
        term.observe(sensors, None)
        np.testing.assert_array_equal(term.peak_load, 0)
        np.testing.assert_array_equal(term.impulse_world, 0)
        np.testing.assert_array_equal(term.touch_fraction, 0)
        sensors[0, 0, 7] = 5
        with pytest.raises(RuntimeError, match="capacity"):
            term.observe(sensors, None)
        sensors[0, 0, 0] = np.nan
        with pytest.raises(RuntimeError, match="non-finite"):
            term.observe(sensors, None)
    finally:
        env.close()


@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_training_autoreset_reattaches_only_finished_row(engine):
    env = make_env(engine)
    try:
        env.set_autoreset(True)
        term = env.action_manager.get_term("residual")
        action = np.zeros((2, 8), np.float32)
        action[:, 7] = 1
        env.step(action)
        assert term.released.all()
        env.set_episode_length_buf(np.array([env.max_episode_length - 1, 1]))
        state = env.step(np.zeros((2, 8), np.float32))
        np.testing.assert_array_equal(state.truncated, [True, False])
        np.testing.assert_array_equal(term.released, [False, True])
        np.testing.assert_array_equal(
            env.equality_constraints.get_equality_active(), [[True], [False]]
        )
        np.testing.assert_array_equal(term.release_physics[0], 0)
        assert term.release_physics[1].any()
        assert np.isfinite(state.obs["obs"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_full_interval_parity_no_pose_writes_and_load_reduction(monkeypatch, hand):
    envs = [make_env(engine, hand) for engine in ("mujoco", "mjbatch")]
    trajectories = [None, None]
    try:
        for index, env in enumerate(envs):
            original = env._backend._recorder.rollout

            def record(*args, index=index, original=original, **kwargs):
                result = original(*args, **kwargs)
                trajectories[index] = result
                return result

            monkeypatch.setattr(env._backend._recorder, "rollout", record)
        for tick in range(20):
            if tick == 12:
                for env in envs:
                    env.reset(env_ids=np.array([0]))
            action = np.zeros((2, 8), np.float32)
            action[:, :7] = 0.1 * np.sin(0.17 * tick + np.arange(7))
            action[0, 7] = float(tick == 6)
            action[1, 7] = float(tick == 15)
            with monkeypatch.context() as patch:

                def forbid(*args, **kwargs):
                    raise AssertionError("step must not write body or joint state")

                for env in envs:
                    for name in ("robot", "ball"):
                        entity = env.scene[name]
                        patch.setattr(entity, "write_root_state_to_sim", forbid)
                        patch.setattr(entity, "write_joint_state_to_sim", forbid)
                states = [env.step(action.copy()) for env in envs]
            for a, b in zip(trajectories[0], trajectories[1], strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            for name in ("reward", "terminated", "truncated"):
                np.testing.assert_array_equal(getattr(states[0], name), getattr(states[1], name))
            for key in states[0].obs:
                np.testing.assert_array_equal(states[0].obs[key], states[1].obs[key])
            assert states[0].obs["obs"].shape == (2, 122)
            for env, trajectory in zip(envs, trajectories, strict=True):
                term = env.action_manager.get_term("residual")
                model = env.get_playback_model()
                sensor = model.sensor("holder_force")
                force = trajectory[1][:, :, sensor.adr[0] : sensor.adr[0] + 3]
                np.testing.assert_allclose(
                    term.peak_load, np.linalg.norm(force, axis=-1).max(axis=1)
                )
                np.testing.assert_array_equal(term.raw_action, action)
                assert np.isfinite(states[0].obs["obs"]).all()
                assert np.all(
                    term.processed_action[:, term.arm_ids] >= term.joint_limits[term.arm_ids, 0]
                )
                assert np.all(
                    term.processed_action[:, term.arm_ids] <= term.joint_limits[term.arm_ids, 1]
                )
    finally:
        for env in envs:
            env.close()
