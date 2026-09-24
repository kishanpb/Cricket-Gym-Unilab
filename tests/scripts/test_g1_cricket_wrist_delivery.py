from copy import deepcopy

import numpy as np
import pytest
from probe_g1_cricket_overarm_release import target_and_release
from probe_g1_cricket_shoulder_damping import owner_config
from probe_g1_cricket_wrist_delivery import PLAN, candidates, verify_baseline, wrist_target
from train_g1_cricket_delivery import make_env


def test_full_factorial_changes_only_wrist_reference_and_release_time():
    neutral = np.array([0.2, 0.2, 0, 1.28, 0, 0, 0])
    assert len(candidates()) == PLAN["rows"] == 9
    for candidate in candidates():
        wrist, release = candidate
        parent = dict(pitch=-2.8, elbow=1.4, drive_pitch=1.0, delay=release - 110)
        for tick in range(200):
            reference, _ = target_and_release(neutral, parent, tick)
            actual = wrist_target(neutral, candidate, tick)
            np.testing.assert_array_equal(np.delete(actual, 5), np.delete(reference, 5))
            if wrist == 0:
                np.testing.assert_array_equal(actual, reference)
            if 30 <= tick <= 150:
                assert actual[5] == wrist
            if tick >= 190:
                np.testing.assert_allclose(actual, neutral, atol=1e-15)


def test_wrist_motor_authority_and_original_joint_bounds():
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    try:
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        m = env.get_playback_model()
        actuator = term.arm_ids[5]
        np.testing.assert_array_equal(m.actuator_biasprm[actuator, 1:3], [-40, -10])
        np.testing.assert_array_equal(m.actuator_forcerange[actuator], [-5, 5])
        for candidate in candidates():
            for tick in range(200):
                target = wrist_target(neutral, candidate, tick)
                assert np.all((target > limits[:, 0]) & (target < limits[:, 1]))
    finally:
        env.close()


def test_baseline_guard_rejects_outcome_and_trace_drift():
    import json

    from probe_g1_cricket_wrist_delivery import PARENT

    baseline = json.loads((PARENT / "evaluation.json").read_text())["rows"][3]
    verify_baseline(baseline)
    changed = deepcopy(baseline)
    changed["outcome"]["passed"] = True
    with pytest.raises(ValueError, match="outcome"):
        verify_baseline(changed)
    changed = deepcopy(baseline)
    changed["trace"][0]["ball_velocity"][0] += 0.1
    with pytest.raises(ValueError, match="trace"):
        verify_baseline(changed)
