"""The versioned pair changes compliance without altering robot or other contacts."""

import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import build_prior_scene
from unilab.tasks.manipulation.g1_cricket.scene import build_scene
from unilab.tasks.manipulation.g1_cricket.task import G1CricketCfg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_residual import CricketReplay


def make_env(version, hand, dt=None):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=[f"task=g1_cricket_{version}/mujoco"])
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    if dt is not None:
        override["sim_dt"] = dt
    return create_env(owner, num_envs=1, env_cfg_override=override)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_impact_pair_preserves_all_other_model_parameters(hand):
    old = make_env("residual_v3", hand)
    new = make_env("impact_v1", hand)
    try:
        a, b = CricketReplay(old), CricketReplay(new)
        assert a.model.npair == 0 and b.model.npair == 1
        assert old.step_dt == new.step_dt == 0.02
        assert new.cfg.sim_dt == 0.0005 and new.cfg.sim_substeps == 40
        assert a.names == b.names
        pair = b.model.pair("cricket_blade_impact_v1").id
        assert {int(b.model.pair_geom1[pair]), int(b.model.pair_geom2[pair])} == {
            b.model.geom("ball_geom").id,
            b.model.geom("bat_blade").id,
        }
        np.testing.assert_array_equal(b.model.pair_solref[pair], [0.004, 1])
        np.testing.assert_array_equal(b.model.pair_solimp[pair], [0.9, 0.95, 0.001, 0.5, 2])
        np.testing.assert_array_equal(b.model.pair_friction[pair], [0.6, 0.6, 0.01, 0.001, 0.001])
        for field in (
            "body_mass",
            "body_inertia",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_solref",
            "geom_solimp",
            "geom_friction",
            "geom_contype",
            "geom_conaffinity",
            "exclude_signature",
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "key_qpos",
            "key_ctrl",
        ):
            np.testing.assert_array_equal(getattr(a.model, field), getattr(b.model, field))
        old.reset(seed=4301)
        new.reset(seed=4301)
        np.testing.assert_array_equal(
            old.get_physics_state_snapshot(), new.get_physics_state_snapshot()
        )
        for _ in range(3):
            state, *_ = b.step(new, np.zeros((1, 7), dtype=np.float32))
            assert not state.terminated[0]
        assert b.state_error == b.sensor_error == 0
    finally:
        old.close()
        new.close()


def test_impact_rejects_underresolved_physics():
    with pytest.raises(ValueError, match="physics timestep <=0.5 ms"):
        make_env("impact_v1", "right", dt=0.002)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_fine_resolution_preserves_control_period_and_native_replay(hand):
    env = make_env("impact_v1", hand, dt=0.000125)
    try:
        replay = CricketReplay(env)
        assert env.step_dt == 0.02 and env.cfg.sim_substeps == replay.steps == 160
        assert replay.model.opt.timestep == 0.000125
        env.reset(seed=4301)
        state, *_ = replay.step(env, np.zeros((1, 7), dtype=np.float32))
        assert not state.terminated[0]
        assert replay.state_error == replay.sensor_error == 0
    finally:
        env.close()


@pytest.mark.parametrize(
    "hand,prior,mount",
    [
        ("right", False, "legacy"),
        ("left", False, "legacy"),
        *[(h, True, m) for h in ("none", "right", "left") for m in ("legacy", "forward_down")],
    ],
)
def test_default_builder_is_identical_to_original_dispatch(tmp_path, hand, prior, mount):
    source = ROOT / "src/unilab/assets/robots/g1/g1.xml"
    old, new = tmp_path / "old.xml", tmp_path / "new.xml"
    cfg = G1CricketCfg(handedness=hand, locomotion_prior=prior, prior_bat_mount=mount)
    expected = (
        build_prior_scene(source, old, hand, mount) if prior else build_scene(source, old, hand)
    )
    assert cfg.build_scene(source, new) == expected
    assert old.read_bytes() == new.read_bytes()
