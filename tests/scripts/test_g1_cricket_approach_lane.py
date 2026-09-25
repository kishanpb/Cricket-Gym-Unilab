from types import SimpleNamespace

import numpy as np

from unilab.tasks.manipulation.g1_cricket.approach import approach_command
from unilab.tasks.manipulation.g1_cricket.approach_lane import lane_command


def test_lane_sign_bound_forward_preservation_and_partial_reset():
    positions = np.array([[0, 0.8, 0.8], [0, -0.8, 0.8], [0, 2, 0.8]])
    defaults = np.array([[0, 0.7, 0.8], [0, -0.7, 0.8], [0, 0.7, 0.8]])
    env = SimpleNamespace(
        num_envs=3,
        episode_length_buf=np.array([75, 150, 250]),
        step_dt=0.02,
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(root_link_pos_w=positions, default_root_state=defaults)
            )
        },
    )
    baseline = approach_command(env)
    result = lane_command(env)
    np.testing.assert_array_equal(result[:, [0, 2]], baseline[:, [0, 2]])
    np.testing.assert_allclose(result[:, 1], [-0.1, 0.1, -0.25])
    positions[0] = defaults[0]
    env.episode_length_buf[0] = 0
    after = lane_command(env)
    np.testing.assert_array_equal(after[0], 0)
    np.testing.assert_array_equal(after[1:], result[1:])
