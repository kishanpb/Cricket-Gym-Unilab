from types import SimpleNamespace

import numpy as np

from integrations.g1_dynamics.batting_maximum_effort import MaximumEffortBattingEnv
from integrations.g1_dynamics.groot_batting_env import GrootBattingEnv


def test_swing_intervention_skips_static_reference_and_preserves_other_controls(monkeypatch):
    controller = SimpleNamespace(
        actuators=np.arange(29), q=np.arange(29), times=np.array([0., 1., 2., 3.]),
        poses=np.zeros((4, 29)), apply=lambda data: data.ctrl.fill(.01),
    )
    controller.poses[2:, 15:29] = .1
    env = MaximumEffortBattingEnv.__new__(MaximumEffortBattingEnv)
    env.controller, env.maximum_arm_effort, env.model = controller, True, None
    monkeypatch.setattr(GrootBattingEnv, "reset", lambda self, **kwargs: (None, {}))
    monkeypatch.setattr("integrations.g1_dynamics.batting_maximum_effort.servo_effort_target",
                        lambda model, data, ids: (np.full(14, .2), np.ones(14), np.full(14, 5)))
    env.reset()
    data = SimpleNamespace(time=.5, ctrl=np.zeros(29))
    controller.apply(data)
    assert env.effort_count == 0
    np.testing.assert_array_equal(data.ctrl, np.full(29, .01))
    data.time = 1.5
    controller.apply(data)
    np.testing.assert_array_equal(data.ctrl[:15], np.full(15, .01))
    np.testing.assert_array_equal(data.ctrl[15:], np.full(14, .2))
    assert env.effort_count == 1
    data.time = 2.5
    controller.apply(data)
    assert env.effort_count == 1
    env.maximum_arm_effort = False
    data.time = 1.5
    controller.apply(data)
    assert env.effort_count == 1
