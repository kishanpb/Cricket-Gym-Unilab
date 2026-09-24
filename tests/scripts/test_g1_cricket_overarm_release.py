import numpy as np
import pytest
from probe_g1_cricket_overarm import target_at
from probe_g1_cricket_overarm_guard import owner_config
from probe_g1_cricket_overarm_release import (
    PLAN,
    candidates,
    release_attribution,
    target_and_release,
)
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.overarm import absolute_targets, actions_for_targets


def test_contact_diagnostic_is_not_a_forbidden_contact_gate():
    original = dict(
        first_joint=None,
        worst_joint=None,
        first_forbidden_contact=dict(tick=130, phase="recover", geoms=["ball_geom", "pitch"]),
        context=[dict(tick=tick, phase="old") for tick in (109, 110, 149, 150, 189, 190)],
    )
    corrected = release_attribution(original)
    assert "first_forbidden_contact" not in corrected
    assert corrected["first_contact_excluding_foot_pitch"]["phase"] == "drive"
    assert [r["phase"] for r in corrected["context"]] == [
        "hold",
        "drive",
        "drive",
        "recover",
        "recover",
        "final_settle",
    ]
    assert original["first_forbidden_contact"]["phase"] == "recover"
    assert original["context"][0]["phase"] == "old"


def test_six_fixed_candidates_and_identical_preload():
    rows = candidates()
    assert len(rows) == PLAN["rows"] == 6
    assert len({tuple(r.values()) for r in rows}) == 6
    neutral = np.array([0.2, 0.2, 0, 1.28, 0, 0, 0])
    for row in rows:
        for tick in range(PLAN["drive_tick"]):
            target, release = target_and_release(neutral, row, tick)
            np.testing.assert_array_equal(
                target, target_at(neutral, "left", row["pitch"], row["elbow"], tick)
            )
            assert not release


@pytest.mark.parametrize("delay", PLAN["release_delays"])
def test_release_is_irreversible_and_recovery_keeps_original_bounds(delay):
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    try:
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        for row in [c for c in candidates() if c["delay"] == delay]:
            for tick in range(200):
                target, release = target_and_release(neutral, row, tick)
                assert release == (tick >= 110 + delay)
                assert np.all((target > limits[:, 0]) & (target < limits[:, 1]))
                raw = actions_for_targets(target, neutral, limits)
                np.testing.assert_allclose(
                    absolute_targets(raw, neutral, limits), target, atol=1e-7
                )
                if tick >= PLAN["recovery_end"]:
                    np.testing.assert_allclose(target, neutral, atol=1e-15)
    finally:
        env.close()
