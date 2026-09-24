import mujoco
import numpy as np
import pytest
from audit_g1_cricket_pitch_contact import drop, owner_config
from train_g1_cricket_delivery import make_env
from train_g1_cricket_delivery import owner_config as parent_config


def test_pitch_pair_does_not_change_robot_holder_or_other_contact_materials():
    old = make_env(parent_config(), "right", dt=0.0000625)
    new = make_env(owner_config(), "right", dt=0.0000625)
    try:
        a, b = old.get_playback_model(), new.get_playback_model()
        for name in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_range",
            "dof_armature",
            "dof_damping",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "geom_solref",
            "geom_solimp",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "eq_solref",
            "eq_solimp",
            "key_qpos",
        ):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        assert b.npair == a.npair + 1
        pair = b.pair("cricket_pitch_impact_v2")
        np.testing.assert_array_equal(pair.solref, [0.002, 0.3])
        assert {pair.geom1[0], pair.geom2[0]} == {b.geom("ball_geom").id, b.geom("pitch").id}
        assert b.opt.timestep == a.opt.timestep
        assert mujoco.mj_stateSize(a, mujoco.mjtState.mjSTATE_FULLPHYSICS) == mujoco.mj_stateSize(
            b, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("velocity", [(0, 0, 0), (8, 0, -4), (12, 0, -8)])
def test_pitch_impact_momentum_rebound_and_resolution(velocity):
    coarse, fine = [drop(dt, velocity, True) for dt in (0.0000625, 0.00003125)]
    for row in (coarse, fine):
        assert row["exact_native_state_sensors"]
        assert 0 < row["maximum_penetration_m"] < 0.006
        assert row["maximum_momentum_residual_ns"] < 1e-10
        assert 0 < row["first_rebound_energy_ratio"] < 0.95
        assert row["first_out_velocity"][2] > 0
    assert abs(coarse["peak_force_n"] - fine["peak_force_n"]) / fine["peak_force_n"] < 0.05
    np.testing.assert_allclose(
        coarse["impulse_world_ns"], fine["impulse_world_ns"], atol=0.01, rtol=0.01
    )
    np.testing.assert_allclose(
        coarse["first_out_velocity"], fine["first_out_velocity"], atol=0.1, rtol=0.01
    )


def test_parent_pitch_response_remains_a_failing_reference():
    old = drop(0.000125, (12, 0, -8), False)
    assert old["maximum_penetration_m"] > 0.006
    assert old["maximum_momentum_residual_ns"] < 1e-10


def test_revised_owner_rejects_unvalidated_coarse_step():
    with pytest.raises(ValueError, match="pitch impact v2 requires"):
        make_env(owner_config(), "right", dt=0.000125)
