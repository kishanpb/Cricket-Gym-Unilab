"""Substep acquisition preserves the reward curve and scores the first separation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.scene import BALL_CONTACT_NAMES
from unilab.tasks.manipulation.g1_cricket.substep_reward import SubstepForwardExitReward

ROOT = Path(__file__).resolve().parents[2]


def make_term(nenv=4):
    velocity = np.zeros((nenv, 3))
    ball = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=np.ones((nenv, 3)), root_link_lin_vel_w=velocity)
    )

    class Scene(dict):
        def bind_sensor_data(self, names):
            size = 3 if names == ("bat_center_world",) else 68
            return SimpleNamespace(read=lambda: np.zeros((nenv, size)))

    def bind(names, body, observer):
        assert names == ("ball_bat",) and body == "cricket_ball"
        env.observer = observer

    env = SimpleNamespace(
        scene=Scene(ball=ball), num_envs=nenv, step_dt=0.02, set_substep_observer=bind
    )
    return env, SubstepForwardExitReward(None, env)


def test_between_sample_contacts_first_exit_and_partial_reset():
    env, term = make_term()
    sensors, velocity = np.zeros((4, 8, 68)), np.zeros((4, 8, 3))
    sensors[0, 1:3, 0] = 1
    sensors[0, 5:7, 0] = 1
    velocity[0, 3, 0], velocity[0, 7, 0] = 2, 5
    sensors[1, -1, 0] = 1
    sensors[2, :4, 0] = 1
    velocity[2, 4, 0] = -1
    env.observer(sensors, velocity)
    expected = np.array([5 * (1 + np.tanh(1)), 0, 5 * (1 + np.tanh(-2)), 0])
    np.testing.assert_allclose(term(env) * env.step_dt, expected)
    np.testing.assert_array_equal(term(env), np.zeros(4))
    assert term.hit_seen.tolist() == [True, True, True, False]
    assert term.scored.tolist() == [True, False, True, False]
    sensors[:] = 0
    velocity[:] = 0
    velocity[1, 0, 0] = 1
    env.observer(sensors, velocity)
    np.testing.assert_allclose(term(env) * env.step_dt, [0, 5, 0, 0])
    term.reset(np.array([0]))
    assert term.scored.tolist() == [False, True, True, False]
    sensors[:, 0, 0] = 1
    velocity[:, 1, 0] = 3
    env.observer(sensors, velocity)
    np.testing.assert_allclose(
        term(env) * env.step_dt, [5 * (1 + np.tanh(2)), 0, 0, 5 * (1 + np.tanh(2))]
    )
    term.reset()
    assert not term.hit_seen.any() and not term.scored.any() and not term.pending.any()


def test_final_substep_separation_and_pending_reset():
    env, term = make_term(1)
    sensors = np.zeros((1, 4, 68))
    velocity = np.zeros((1, 4, 3))
    sensors[:, :-1, 0] = 1
    velocity[:, -1, 0] = 2
    env.observer(sensors, velocity)
    assert term.pending[0] and term.separation_velocity[0] == 2
    term.reset()
    assert term(env)[0] == 0
    sensors[:] = 0
    env.observer(sensors, velocity)
    assert term(env)[0] == 0


def test_one_substep_cross_interval_and_capacity():
    env, term = make_term(1)
    sensors, velocity = np.zeros((1, 1, 68)), np.zeros((1, 1, 3))
    sensors[0, 0, 0] = 1
    env.observer(sensors, velocity)
    assert term(env)[0] == 0
    sensors[:] = 0
    velocity[:, :, 0] = 1
    env.observer(sensors, velocity)
    assert term(env)[0] * env.step_dt == 5
    sensors[0, 0, 0] = 5
    with pytest.raises(RuntimeError, match="capacity"):
        env.observer(sensors, velocity)


def test_owner_preserves_every_other_training_setting():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        old, new = [
            OmegaConf.to_container(
                compose(
                    "config", overrides=[f"task=g1_cricket_{name}/mujoco", "env.sim_dt=0.00025"]
                ),
                resolve=True,
            )
            for name in ("impact_v1", "impact_events_v1")
        ]
    assert new["env"].pop("mujoco_observe_substeps") is True
    assert new["reward"]["batting"].pop("func").endswith("SubstepForwardExitReward")
    assert old["reward"]["batting"].pop("func").endswith("ForwardExitReward")
    assert old == new


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_real_g1_batch_keeps_physics_and_policy_observations_exact(hand, dt):
    envs = []
    try:
        for task in ("impact_v1", "impact_events_v1"):
            with initialize_config_dir(
                config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"
            ):
                owner = compose("config", overrides=[f"task=g1_cricket_{task}/mujoco"])
            cfg = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            cfg.update(handedness=hand, auto_reset=False, sim_dt=dt)
            env = create_env(owner, num_envs=2, env_cfg_override=cfg)
            envs.append(env)
            env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [0.0]
            env.reset(seed=4301)
        old, new = envs
        contacts = [env.scene.bind_sensor_data(BALL_CONTACT_NAMES) for env in envs]
        for _ in range(30):
            a, b = [env.step(np.zeros((2, 7), dtype=np.float32)) for env in envs]
            np.testing.assert_array_equal(
                old.get_physics_state_snapshot(), new.get_physics_state_snapshot()
            )
            np.testing.assert_array_equal(contacts[0].read(), contacts[1].read())
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key], b.obs[key])
            assert not a.terminated.any() and not b.terminated.any()
        term = new.reward_manager.get_term_cfg("batting").func
        assert term.hit_seen.all() and term.scored.all()
    finally:
        for env in envs:
            env.close()
