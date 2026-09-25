"""Soft toss is reset-only; the two-hand robot retains its original limits."""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
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
from unilab.tasks.manipulation.g1_cricket.pitch_contact import add_pitch_pair

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
    with pytest.raises(ValueError, match="fine-resolution"):
        evaluate(tmp_path, bounced_delivery=True)
    with pytest.raises(ValueError, match="only one incoming"):
        evaluate(tmp_path, soft_toss=True, bounced_delivery=True)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
@pytest.mark.parametrize(
    "delivery,position,velocity",
    [
        ("ResetSoftToss", [4, 0, 1.1], [-2.8, 0, 6.5]),
        ("ResetBouncedDelivery", [4, 0, 1.3], [-3, 0, 4]),
    ],
)
def test_soft_toss_reset_free_flight_and_exact_replay(
    hand, engine, delivery, position, velocity, monkeypatch
):
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
            "func": "unilab.tasks.manipulation.g1_cricket.bimanual_contact." + delivery,
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
        np.testing.assert_allclose(ball.data.root_link_pos_w[0], position)
        np.testing.assert_allclose(ball.data.root_link_lin_vel_w[0], velocity)

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
                replay.data.qvel[va : va + 3],
                [velocity[0], velocity[1], velocity[2] - 9.81 * t],
                atol=1e-6,
            )
        assert not replay.data.warning.number.any()
    finally:
        env.close()


@pytest.mark.parametrize("dt,expected_height", [(0.0000625, 0.461676), (0.00003125, 0.425088)])
def test_nominal_ball_only_delivery_bounces_before_swing(dt, expected_height):
    from evaluate_g1_cricket_tracking import BallContactSequence

    root = ET.fromstring("""
    <mujoco><option integrator="implicitfast"/><worldbody>
      <geom name="pitch" type="plane" size="25 25 .05" friction=".7 .01 .001" condim="3"/>
      <body pos="4 0 1.3"><freejoint/>
        <geom name="ball_geom" type="sphere" size=".036" mass=".156"
          friction=".5 .01 .001" condim="3"/>
      </body>
    </worldbody></mujoco>""")
    add_pitch_pair(root)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    data.qvel[:3] = [-3, 0, 4]
    sequence = BallContactSequence()
    force = np.zeros(6)
    for _ in range(round(1.4 / dt)):
        before = data.qvel[:3].copy()
        mujoco.mj_step(model, data)
        loaded = False
        for i, contact in enumerate(data.contact):
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(model, data, i, force)
                loaded |= force[0] > 0
        sequence.update(data.time, data.qpos[:3], before, data.qvel[:3], loaded, False)
    assert len(sequence.pitch_events) == 1
    event = sequence.pitch_events[0]
    assert 1.05 < event["start_s"] < event["end_s"] < 1.1
    assert event["incoming_velocity_m_s"][2] < 0 < event["outgoing_velocity_m_s"][2]
    assert 0.8 < event["position_m"][0] < 0.85
    assert data.qpos[2] == pytest.approx(expected_height, abs=1e-6)
    assert 0.02 < data.qpos[0] < 0.06
    assert data.qvel[0] < -2
    assert sequence.first_blade_contact is None
    assert not data.warning.number.any()
