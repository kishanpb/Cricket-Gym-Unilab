"""The native Batch executor must preserve the complete G1 policy/sensor contract."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("mjbatch.held_control")
from mjbatch.held_control import HeldControlRollout

from unilab.base.config_adapter import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[2]


def make_env(engine, hand, dt, lane, nenv=1):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task=g1_cricket_tanh_v1/{engine}"])
    override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False, sim_dt=dt)
    env = create_env(cfg, num_envs=nenv, env_cfg_override=override)
    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [lane]
    env.reset(seed=4301)
    return env


def test_owner_changes_only_executor():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        configs = [
            OmegaConf.to_container(
                compose("config", overrides=[f"task=g1_cricket_tanh_v1/{engine}"]), resolve=True
            )
            for engine in ("mujoco", "mjbatch")
        ]
    assert configs[1]["env"].pop("mujoco_substep_engine") == "mjbatch"
    assert configs[0] == configs[1]


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("dt", [0.00025, 0.000125])
@pytest.mark.parametrize("lane", [0.0, -0.12])
@pytest.mark.parametrize("nonzero", [False, True])
def test_full_g1_interval_parity(monkeypatch, hand, dt, lane, nonzero):
    envs = [make_env(engine, hand, dt, lane) for engine in ("mujoco", "mjbatch")]
    trajectories = [None, None]
    try:
        assert isinstance(envs[1]._backend._recorder, HeldControlRollout)
        for index, env in enumerate(envs):
            original = env._backend._recorder.rollout

            def record(*args, index=index, original=original, **kwargs):
                result = original(*args, **kwargs)
                trajectories[index] = result
                return result

            monkeypatch.setattr(env._backend._recorder, "rollout", record)
        contact_seen = False
        rewards = [env.reward_manager.get_term_cfg("batting").func for env in envs]
        sensor = envs[0].get_playback_model().sensor("ball_bat")
        for tick in range(100):
            action = np.zeros((1, 7), dtype=np.float32)
            if nonzero:
                action[0] = 1.6 * np.sin(0.17 * tick + np.arange(7))
            states = [env.step(action.copy()) for env in envs]
            for a, b in zip(trajectories[0], trajectories[1], strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            np.testing.assert_array_equal(
                envs[0]._backend._sensor_data, envs[1]._backend._sensor_data
            )
            for name in ("reward", "terminated", "truncated"):
                np.testing.assert_array_equal(getattr(states[0], name), getattr(states[1], name))
            for key in states[0].obs:
                np.testing.assert_array_equal(states[0].obs[key], states[1].obs[key])
            terms = [env.action_manager.get_term("residual") for env in envs]
            for name in ("baseline_action", "raw_action", "processed_action"):
                np.testing.assert_array_equal(getattr(terms[0], name), getattr(terms[1], name))
            for name in ("hit_seen", "scored", "pending", "separation_velocity"):
                np.testing.assert_array_equal(getattr(rewards[0], name), getattr(rewards[1], name))
            contact_seen |= bool(np.any(trajectories[0][1][:, :, sensor.adr[0]] > 0))
            if states[0].terminated[0] or states[0].truncated[0]:
                break
        assert states[0].obs["obs"].shape == (1, 115)
        if not nonzero and lane == 0:
            assert contact_seen
    finally:
        for env in envs:
            env.close()


def test_two_row_partial_reset_and_external_wrench():
    envs = [make_env(engine, "right", 0.00025, 0, nenv=2) for engine in ("mujoco", "mjbatch")]
    try:
        assert not envs[1]._backend.get_dr_capabilities().supported_reset_terms
        for tick in range(30):
            if tick == 5:
                for env in envs:
                    body = env.get_playback_model().body("cricket_ball").id
                    env._backend.apply_body_force(
                        np.array([body]), np.full((2, 1, 3), 0.03), np.full((2, 1, 3), 0.001)
                    )
            if tick == 16:
                for env in envs:
                    before = env.get_physics_state_snapshot()[1].copy()
                    term = env.action_manager.get_term("residual")
                    prior = term.baseline_action[1].copy()
                    env.reset(env_ids=np.array([0]))
                    np.testing.assert_array_equal(env.get_physics_state_snapshot()[1], before)
                    np.testing.assert_array_equal(term.baseline_action[1], prior)
            states = [env.step(np.zeros((2, 7), dtype=np.float32)) for env in envs]
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            np.testing.assert_array_equal(
                envs[0]._backend._sensor_data, envs[1]._backend._sensor_data
            )
            np.testing.assert_array_equal(states[0].reward, states[1].reward)
            for key in states[0].obs:
                np.testing.assert_array_equal(states[0].obs[key], states[1].obs[key])
    finally:
        for env in envs:
            env.close()
