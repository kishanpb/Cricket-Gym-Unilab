import json

import numpy as np
import pytest
from probe_g1_cricket_overarm_damping import inputs, paired_candidates
from probe_g1_cricket_overarm_release import PLAN, candidates, target_and_release
from probe_g1_cricket_shoulder_damping import owner_config
from train_g1_cricket_delivery import make_env


def test_all_original_candidates_are_paired_without_filtering():
    assert paired_candidates() == candidates()
    assert len(paired_candidates()) == PLAN["rows"] == 6
    record = inputs()
    assert record["changed_axis"] == "selected_shoulder_pitch_controller_kd_10_to_2"
    assert record["config"]["training"]["task_name"] == "G1CricketShoulderDamping"
    assert record["plan"] == PLAN


def test_missing_parent_row_fails_closed(monkeypatch, tmp_path):
    import probe_g1_cricket_overarm_damping as study

    (tmp_path / "evaluation.json").write_text(
        json.dumps(dict(rows=[dict(candidate=c) for c in candidates()[:-1]]))
    )
    monkeypatch.setattr(study, "PARENT", tmp_path)
    with pytest.raises(ValueError, match="all six original"):
        paired_candidates()


def test_retuned_owner_preserves_reference_bounds_and_release_schedule():
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    try:
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        for candidate in paired_candidates():
            for tick in range(200):
                target, release = target_and_release(neutral, candidate, tick)
                assert np.all((target > limits[:, 0]) & (target < limits[:, 1]))
                assert release == (tick >= PLAN["drive_tick"] + candidate["delay"])
                if tick >= PLAN["recovery_end"]:
                    np.testing.assert_allclose(target, neutral, atol=1e-15)
    finally:
        env.close()
