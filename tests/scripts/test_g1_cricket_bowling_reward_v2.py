from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from probe_g1_cricket_overarm_guard import owner_config
from train_g1_cricket_delivery import ROOT

from unilab.tasks.manipulation.g1_cricket.bowling import bowling_release_reward_v2, bowling_reward


def test_reward_v2_changes_only_ball_position_proxy():
    term = SimpleNamespace(
        release_position=np.array([[-0.2, 0, 1.3], [0.2, 0, 1.3], [0.2, 0, 1.3], [0.2, 0, 1.3]]),
        release_velocity=np.array([[3.0, 0, 0], [3.0, 0, 0], [-3.0, 0, 0], [3.0, 0, 0]]),
        just_released=np.array([True, True, True, False]),
        released=np.array([True, True, True, True]),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=np.zeros((4, 3, 3)),
            root_link_lin_vel_w=np.tile([0.4, 0, 0], (4, 1)),
        )
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: term),
        scene={"robot": robot},
        cfg=SimpleNamespace(handedness="left"),
        step_dt=0.02,
    )
    old, new = bowling_reward(env), bowling_release_reward_v2(env)
    np.testing.assert_allclose(old, [250.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(new, [250.5, 250.5, 0.5, 0.5])


def test_versioned_owner_preserves_dynamics_observations_and_actions():
    for engine in ("mujoco", "mjbatch"):
        with initialize_config_dir(
            config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"
        ):
            new = OmegaConf.to_container(
                compose("config", overrides=[f"task=g1_cricket_overarm_reward_v2/{engine}"]),
                resolve=True,
            )
        old = OmegaConf.to_container(owner_config(engine), resolve=True)
        assert new["reward"]["bowling"].pop("func").endswith("bowling_release_reward_v2")
        assert old["reward"]["bowling"].pop("func").endswith("bowling_reward")
        assert old == new
