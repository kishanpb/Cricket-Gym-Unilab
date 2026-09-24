"""The separation objective changes reward only, not the simulator contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.tasks.manipulation.g1_cricket.forward_exit_reward import ForwardExitReward
from unilab.tasks.manipulation.g1_cricket.separation_reward import SeparationReward

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "candidate,suffix",
    [("v2", "separation_reward.SeparationReward"), ("v3", "forward_exit_reward.ForwardExitReward")],
)
def test_only_reward_changes_in_owner(candidate, suffix):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        configs = [
            OmegaConf.to_container(
                compose("config", overrides=[f"task=g1_cricket_residual_{v}/mujoco"]), resolve=True
            )
            for v in ("v1", candidate)
        ]
    old_func = configs[0]["reward"]["batting"].pop("func")
    new_func = configs[1]["reward"]["batting"].pop("func")
    assert old_func.endswith("residual.BattingReward")
    assert new_func.endswith(suffix)
    assert configs[0] == configs[1]


@pytest.mark.parametrize("reward_cls", [SeparationReward, ForwardExitReward])
def test_separation_scores_once_with_partial_reset(reward_cls):
    records = np.zeros((4, 68))
    velocity = np.zeros((4, 3))
    velocity[:, 0] = [-1, 0.5, 1, 2]
    ball = SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=np.ones((4, 3)), root_link_lin_vel_w=velocity)
    )
    views = {
        "bat_center_world": SimpleNamespace(read=lambda: np.zeros((4, 3))),
        "ball_bat": SimpleNamespace(read=lambda: records),
    }

    class Scene(dict):
        def bind_sensor_data(self, names):
            return views[names[0]]

    env = SimpleNamespace(scene=Scene(ball=ball), num_envs=4, step_dt=0.02)
    term = reward_cls(None, env)
    shaping = term(env)
    records[:, 0] = 1
    np.testing.assert_array_equal(term(env), shaping)
    records[:, 0] = 0
    event = (term(env) - shaping) * env.step_dt
    expected = 10 * np.tanh(velocity[:, 0] - 1)
    if reward_cls is ForwardExitReward:
        expected = 5 + expected / 2
        assert (event >= 0).all() and (event <= 10).all()
        assert event[0] < event[1] < event[2] < event[3]
    else:
        assert event[0] < event[1] < 0 and event[2] == 0 and event[3] > 0
    np.testing.assert_allclose(event, expected)
    np.testing.assert_array_equal(term(env), shaping)
    records[:, 0] = 1
    term(env)
    records[:, 0] = 0
    np.testing.assert_array_equal(term(env), shaping)
    term.reset(np.array([3]))
    assert term.scored.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(term(env), shaping)
    records[3, 0] = 1
    term(env)
    records[3, 0] = 0
    np.testing.assert_allclose((term(env) - shaping) * env.step_dt, [0, 0, 0, expected[3]])
