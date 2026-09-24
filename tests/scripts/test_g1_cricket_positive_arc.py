import mujoco
import numpy as np
from probe_g1_cricket_overarm_guard import GuardAudit, owner_config
from probe_g1_cricket_positive_arc import (
    PLAN,
    LaunchAudit,
    candidates,
    corrected_attribution,
    launch_target,
    phase,
)
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.overarm import actions_for_targets


def test_six_fixed_cases_targets_and_original_compiled_gains():
    assert len(candidates()) == PLAN["rows"] == 6
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    try:
        m = env.get_playback_model()
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[0, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        actuator = term.arm_ids[0]
        assert m.actuator_gainprm[actuator, 0] == 40
        np.testing.assert_array_equal(m.actuator_biasprm[actuator, 1:3], [-40, -10])
        np.testing.assert_array_equal(m.actuator_forcerange[actuator], [-25, 25])
        for candidate in candidates():
            drive, brake = candidate
            for tick in range(200):
                target = launch_target(neutral, candidate, tick)
                assert np.all((target > limits[:, 0]) & (target < limits[:, 1]))
                assert np.isfinite(actions_for_targets(target, neutral, limits)).all()
                if drive <= tick < brake:
                    assert target[0] == 2.6
                if brake <= tick <= 150:
                    assert target[0] == 1.6
                if 30 <= tick <= 150:
                    np.testing.assert_allclose(target[3:], [1.4, 0, 0, 0], atol=1e-15)
                if tick >= 190:
                    np.testing.assert_allclose(target, neutral, atol=1e-15)
    finally:
        env.close()


def test_brake_at_release_has_identical_held_actions_but_earlier_brake_does_not():
    neutral = np.array([0.2, 0.2, 0, 1.28, 0, 0, 0])
    assert PLAN["release_tick"] == 114
    for drive in PLAN["drive_ticks"]:
        for tick in range(112):
            np.testing.assert_array_equal(
                launch_target(neutral, [drive, 112], tick),
                launch_target(neutral, [drive, 114], tick),
            )
        assert launch_target(neutral, [drive, 112], 112)[0] == 1.6
        assert launch_target(neutral, [drive, 114], 113)[0] == 2.6
        assert launch_target(neutral, [drive, 114], 114)[0] == 1.6


def test_contact_label_is_not_a_gate_and_phases_match_actual_controls():
    source = dict(
        first_joint=None,
        worst_joint=dict(tick=115, phase="hold"),
        first_forbidden_contact=dict(tick=140, phase="recover"),
        context=[dict(tick=100, phase="hold")],
    )
    result = corrected_attribution(source, [98, 112])
    assert result["worst_joint"]["phase"] == "brake"
    assert result["first_contact_excluding_foot_pitch"]["phase"] == "brake"
    assert result["context"][0]["phase"] == "drive"
    assert source["first_forbidden_contact"]["phase"] == "recover"
    assert phase(190, [98, 112]) == "final_settle"


def test_phase_torque_extrema_use_real_samples_not_an_unobserved_zero(monkeypatch):
    env = make_env(owner_config(), "left", dt=PLAN["dt"])
    monkeypatch.setattr(GuardAudit, "__call__", lambda *args: None)
    try:
        audit = LaunchAudit(env, [98, 112])
        m = env.get_playback_model()
        data = mujoco.MjData(m)
        for tick, torques in ((100, [5, 10, 25]), (114, [-10, -5])):
            for torque in torques:
                data.actuator_force[audit.index] = torque
                audit(tick, m, data)
        assert audit.torques["drive"] == dict(
            minimum_nm=5, maximum_nm=25, saturated_substeps=1, substeps=3
        )
        assert audit.torques["brake"] == dict(
            minimum_nm=-10, maximum_nm=-5, saturated_substeps=0, substeps=2
        )
    finally:
        env.close()
