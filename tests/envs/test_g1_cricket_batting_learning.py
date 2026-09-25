"""Ball-observed batting keeps reset-only dynamics and the original tracking target."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.batting_learning import (
    BatReferenceObservation,
    BatTrackingReward,
    LoadedForwardExitReward,
)

ROOT = Path(__file__).resolve().parents[2]


def test_bat_error_and_reward_use_current_frame_and_environment_origin():
    state = object.__new__(BatReferenceObservation)
    state.command = SimpleNamespace(time_steps=np.array([0, 1]))
    state.positions = np.array([[1, 2, 3], [2, 3, 4]], dtype=float)
    origins = np.array([[0, 0, 0], [10, -5, 0]])
    actual = state.positions + origins + [[0, 0, 0], [0.08, 0, 0]]
    state.bat = SimpleNamespace(read=lambda: actual)
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    error = state(env, "unused.npz")
    np.testing.assert_allclose(error, [[0, 0, 0], [0.08, 0, 0]], atol=1e-14)
    reward = object.__new__(BatTrackingReward)
    reward.__dict__.update(state.__dict__)
    np.testing.assert_allclose(reward(env, "unused.npz"), [1, np.exp(-1)])
    np.testing.assert_array_equal(state.command.time_steps, [0, 1])


def test_loaded_exit_ignores_margin_contact_and_stale_force_then_scores_once():
    reward = object.__new__(LoadedForwardExitReward)
    reward.hit_seen = np.zeros(2, dtype=bool)
    reward.scored = np.zeros(2, dtype=bool)
    reward.pending = np.zeros(2, dtype=bool)
    reward.separation_velocity = np.zeros(2)
    sensors = np.zeros((2, 3, 4, 17))
    sensors[:, 0, 0, 0] = 1
    sensors[:, 0, 1, 1] = 50
    velocities = np.zeros((2, 3, 3))
    reward.observe(sensors.reshape(2, 3, -1), velocities)
    assert not reward.hit_seen.any() and not reward.pending.any()
    sensors[1, 1, 0, :2] = [1, 40]
    velocities[1, 2, 0] = 2.5
    reward.observe(sensors.reshape(2, 3, -1), velocities)
    np.testing.assert_array_equal(reward.pending, [False, True])
    np.testing.assert_array_equal(reward.separation_velocity, [0, 2.5])
    reward.ball = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=np.zeros((2, 3)), root_link_lin_vel_w=np.ones((2, 3)))
    )
    reward.bat = SimpleNamespace(read=lambda: np.zeros((2, 3)))
    env = SimpleNamespace(step_dt=0.02)
    np.testing.assert_allclose(reward(env), [0, 5 * (1 + np.tanh(1.5)) / 0.02])
    reward.observe(sensors.reshape(2, 3, -1), velocities)
    np.testing.assert_array_equal(reward(env), [0, 0])
    reward.reset(np.array([1]))
    assert not reward.hit_seen.any() and not reward.scored.any()
    sensors[0, 0, 0, 0] = 5
    with pytest.raises(RuntimeError, match="capacity"):
        reward.observe(sensors.reshape(2, 3, -1), velocities)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_batting_learning_owner_has_ball_tactile_bat_state_and_reset_only_physics(
    hand, engine, monkeypatch
):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[f"task=g1_cricket_batting_learning_v1/{engine}", f"env.handedness={hand}"],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make(
        "G1CricketBimanualLearning", num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        obs, _ = env.reset(seed=1)
        assert obs["obs"].shape[1] > 160 and obs["critic"].shape[1] > 286
        for group in ("actor", "critic"):
            assert {"ball", "contact", "bat_error"} <= set(
                env.observation_manager.active_terms[group]
            )
            assert np.isfinite(obs["obs" if group == "actor" else group]).all()
        assert {"bat_tracking", "loaded_exit"} <= set(env.reward_manager.active_terms)
        np.testing.assert_allclose(env.scene["ball"].data.root_link_pos_w, [[4, 0, 1.3]])
        np.testing.assert_allclose(env.scene["ball"].data.root_link_lin_vel_w, [[-3, 0, 4]])
        assert env.action_manager.get_term("reference").cfg.lookahead_frames == 0
        assert env.action_manager.get_term("reference").cfg.scale == 0.05

        def forbidden(*args, **kwargs):
            pytest.fail("state write after reset")

        monkeypatch.setattr(Entity, "write_root_state_to_sim", forbidden)
        monkeypatch.setattr(Entity, "write_joint_state_to_sim", forbidden)
        for _ in range(2):
            env.step(np.zeros((1, 29), dtype=np.float32))
            assert np.isfinite(env.state.reward).all()
            assert all(np.isfinite(value).all() for value in env.state.obs.values())
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        mujoco.mj_setState(
            model, data, env.get_physics_state_snapshot()[0], mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        va = int(model.joint("ball_free").dofadr[0])
        np.testing.assert_allclose(data.qvel[va : va + 3], [-3, 0, 4 - 9.81 * 0.04], atol=1e-6)
        # Entity velocities are solve-phase snapshots, one physics substep earlier.
        np.testing.assert_allclose(
            env.scene["ball"].data.root_link_lin_vel_w,
            [[-3, 0, 4 - 9.81 * (0.04 - env.physics_dt)]],
            atol=1e-6,
        )
    finally:
        env.close()
