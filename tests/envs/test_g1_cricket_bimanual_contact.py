"""Soft toss is reset-only; the two-hand robot retains its original limits."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from evaluate_g1_cricket_tracking import TrackingReplay, evaluate
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.base.scene import SceneCfg
from unilab.tasks.manipulation.g1_cricket.bimanual_contact import G1BimanualContactCfg

ROOT = Path(__file__).resolve().parents[2]


def test_contact_task_registered_in_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from unilab.base import registry; registry.ensure_registries(); "
            "assert registry.contains('G1CricketBimanualContact')",
        ],
        check=True,
        cwd=ROOT,
    )


def test_contact_model_rejects_coarse_physics():
    with pytest.raises(ValueError, match="<=0.0625 ms"):
        G1BimanualContactCfg(
            sim_dt=0.001,
            ctrl_dt=0.02,
            max_episode_seconds=3,
            scene=SceneCfg(model_file="unused.xml"),
        ).validate()


def test_soft_toss_cannot_use_dry_model(tmp_path):
    with pytest.raises(ValueError, match="fine-resolution"):
        evaluate(tmp_path, soft_toss=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_soft_toss_reset_free_flight_and_exact_replay(hand, engine, monkeypatch):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[f"task=g1_cricket_balanced_tracking/{engine}", f"env.handedness={hand}"],
        )
    OmegaConf.set_struct(owner, False)
    owner.env.sim_dt = 0.0000625
    owner.env.actions.reference.waist_tracking_gain = 1.0
    owner.env.actions.reference.root_position_gain = 4.0
    owner.env.scene.entities.ball = {
        "root_body_name": "cricket_ball",
        "body_names": ["cricket_ball"],
    }
    owner.env.events = {
        "soft_toss": {
            "func": "unilab.tasks.manipulation.g1_cricket.bimanual_contact.ResetSoftToss",
            "mode": "reset",
        }
    }
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make(
        "G1CricketBimanualContact", num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.reset(seed=1)
        model = env.get_playback_model()
        tree = ET.parse(Path(env.scene_directory.name) / "cricket.xml")
        assert model.neq == 1 and model.nu == 29
        assert tree.find("contact/pair[@geom2='bat_blade']").get("solref") == "0.002 1"
        assert tree.find("contact/pair[@geom2='pitch']").get("solref") == ".002 .3"
        original = ET.parse(ROOT / owner.env.scene.model_file)
        for tag in ("joint", "inertial"):
            assert [x.attrib for x in tree.findall(f".//{tag}")] == [
                x.attrib for x in original.findall(f".//{tag}")
            ]
        assert [x.get("forcerange") for x in tree.findall("actuator/*")] == [
            x.get("forcerange") for x in original.findall("actuator/*")
        ]
        ball = env.scene["ball"]
        np.testing.assert_allclose(ball.data.root_link_pos_w[0], [4, 0, 1.1])
        np.testing.assert_allclose(ball.data.root_link_lin_vel_w[0], [-2.8, 0, 6.5])

        def forbidden(*args, **kwargs):
            pytest.fail("state write after reset")

        monkeypatch.setattr(Entity, "write_root_state_to_sim", forbidden)
        monkeypatch.setattr(Entity, "write_joint_state_to_sim", forbidden)
        replay = TrackingReplay(env, ball_contact=True)
        for tick in range(5):
            initial = env.get_physics_state_snapshot()[0].copy()
            env.step(np.zeros((1, 29), dtype=np.float32))
            audit = replay.measure(env, initial)
            assert not audit["ball_contacts"]
            assert audit["first_force_free_blade_exit"] is None
            t = (tick + 1) * 0.02
            va = int(model.joint("ball_free").dofadr[0])
            np.testing.assert_allclose(
                replay.data.qvel[va : va + 3], [-2.8, 0, 6.5 - 9.81 * t], atol=1e-6
            )
        assert not replay.data.warning.number.any()
    finally:
        env.close()
