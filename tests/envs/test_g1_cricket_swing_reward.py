"""Forward shaping uses existing solved-phase kinematics, never post-hit motion."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.swing_reward import ForwardSwingReward

ROOT = Path(__file__).resolve().parents[2]


def make_term(nenv=5):
    position = np.tile([0.0, 0.0, -0.4], (nenv, 1))
    ball = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=position.copy(),
            root_link_lin_vel_w=np.tile([-1.0, 0, 0], (nenv, 1)),
        )
    )
    robot = SimpleNamespace(
        body_names=("pelvis", "cricket_bat"),
        data=SimpleNamespace(
            body_link_pos_w=np.zeros((nenv, 2, 3)),
            body_link_lin_vel_w=np.zeros((nenv, 2, 3)),
            body_link_ang_vel_w=np.zeros((nenv, 2, 3)),
        ),
    )

    class Scene(dict):
        def bind_sensor_data(self, names):
            return SimpleNamespace(
                read=lambda: position if names == ("bat_center_world",) else np.zeros((nenv, 68))
            )

    def bind(names, body, observer):
        assert names == ("ball_bat",) and body == "cricket_ball"
        env.observer = observer

    env = SimpleNamespace(
        scene=Scene(ball=ball, robot=robot),
        num_envs=nenv,
        step_dt=0.02,
        set_substep_observer=bind,
    )
    return env, ForwardSwingReward(None, env)


def test_blade_velocity_includes_angular_offset():
    _, term = make_term(1)
    term.robot.data.body_link_lin_vel_w[0, 1] = [0.1, 0.2, 0.3]
    term.robot.data.body_link_ang_vel_w[0, 1] = [0, -2, 0]
    np.testing.assert_allclose(term.blade_velocity(), [[0.9, 0.2, 0.3]])


def test_approach_direction_speed_saturation_distance_and_incoming_mask():
    env, term = make_term()
    term.robot.data.body_link_lin_vel_w[:, 1, 0] = [-1, 0, 0.5, 1, 2]
    np.testing.assert_array_equal(term(env), [0, 0, 2.5, 5, 5])
    term.ball.data.root_link_pos_w[2, 1] += 0.18
    term.ball.data.root_link_lin_vel_w[3:, 0] = [0, 1]
    np.testing.assert_allclose(term(env), [0, 0, 2.5 / np.e, 0, 0])


def test_contact_interval_mask_first_bonus_recontact_and_partial_reset():
    env, term = make_term(2)
    term.robot.data.body_link_lin_vel_w[:, 1, 0] = 1
    sensors, velocity = np.zeros((2, 4, 68)), np.zeros((2, 4, 3))
    sensors[0, 1, 0] = 1
    velocity[0, 2, 0] = 2
    sensors[1, -1, 0] = 1
    env.observer(sensors, velocity)
    np.testing.assert_allclose(term(env) * env.step_dt, [5 * (1 + np.tanh(1)), 0])
    np.testing.assert_array_equal(term(env), [0, 0])
    sensors[:] = 0
    velocity[:, :, 0] = 1
    env.observer(sensors, velocity)
    np.testing.assert_allclose(term(env) * env.step_dt, [0, 5])
    sensors[:, 1, 0] = 1
    velocity[:, 2, 0] = 6
    env.observer(sensors, velocity)
    np.testing.assert_array_equal(term(env), [0, 0])
    term.reset(np.array([0]))
    np.testing.assert_array_equal(term(env), [5, 0])
    term.reset()
    np.testing.assert_array_equal(term(env), [5, 5])
    assert not term.pending.any() and not term.hit_seen.any() and not term.scored.any()


def owner(task):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_{task}/mujoco"])


def test_owner_changes_only_batting_reward_function():
    old, new = [
        OmegaConf.to_container(owner(name), resolve=True)
        for name in ("impact_events_v1", "swing_v1")
    ]
    assert old["reward"]["batting"].pop("func").endswith("SubstepForwardExitReward")
    assert new["reward"]["batting"].pop("func").endswith("ForwardSwingReward")
    assert old == new


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_native_physics_observation_parity_and_solved_blade_velocity(hand, dt):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate_g1_cricket_residual import CricketReplay

    envs = []
    try:
        for name in ("impact_events_v1", "swing_v1"):
            cfg = owner(name)
            override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
            override.update(handedness=hand, auto_reset=False, sim_dt=dt)
            env = create_env(cfg, num_envs=1, env_cfg_override=override)
            envs.append(env)
            env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [0.0]
            env.reset(seed=4301)
        old, new = envs
        replay = CricketReplay(new)
        term = new.reward_manager.get_term_cfg("batting").func
        site = replay.model.site("bat_center").id
        velocity = np.empty(6)
        for tick in range(30):
            action = np.full((1, 7), 0.03 * np.sin(tick), dtype=np.float32)
            a = old.step(action)
            b = replay.step(new, action)[0]
            np.testing.assert_array_equal(
                old.get_physics_state_snapshot(), new.get_physics_state_snapshot()
            )
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key], b.obs[key])
            mujoco.mj_objectVelocity(
                replay.model, replay.data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0
            )
            np.testing.assert_allclose(term.blade_velocity()[0], velocity[3:], atol=2e-6, rtol=0)
        assert term.hit_seen.all() and term.scored.all()
        assert replay.state_error == replay.sensor_error == 0
    finally:
        for env in envs:
            env.close()
