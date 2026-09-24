"""Native G1 cricket construction and dynamics contracts, not policy quality."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.scene import build_scene
from unilab.tasks.manipulation.g1_cricket.task import IncidentalBatSupport

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "src/unilab/assets/robots/g1/g1.xml"


def make_env(handedness: str, task: str = "g1_cricket_batting/mujoco", rotation=None):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=[f"task={task}", f"env.handedness={handedness}"])
    if rotation is not None:
        OmegaConf.update(
            owner,
            "env.events.audit_rotation",
            {
                "func": "unilab.envs.mdp.reset_root_state_uniform",
                "mode": "reset",
                "params": {
                    "pose_range": {
                        name: [float(angle), float(angle)]
                        for name, angle in zip(
                            ("roll", "pitch", "yaw"), np.deg2rad(rotation), strict=True
                        )
                    },
                    "velocity_range": {},
                },
            },
            force_add=True,
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    return registry.make(
        "G1CricketBatting", num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_scene_preserves_robot_and_adds_free_ball(tmp_path, handedness):
    output = tmp_path / "scene.xml"
    before = ROBOT.read_bytes()
    build_scene(ROBOT, output, handedness)
    root = ET.parse(output).getroot()
    assert len(root.findall(".//freejoint")) == 2
    assert (
        root.find(f".//body[@name='{handedness}_wrist_yaw_link']/body[@name='cricket_bat']")
        is not None
    )
    assert root.find(".//body[@name='cricket_bat']/joint") is None
    assert root.find(".//body[@mocap='true']") is None
    original = ET.fromstring(before)
    assert [e.attrib for e in root.findall("contact/*")] == [
        e.attrib for e in original.findall("contact/*")
    ]
    assert [e.attrib for e in root.findall("actuator/*")] == [
        e.attrib for e in original.findall("actuator/*")
    ]
    for tag in ("joint", "inertial"):
        assert [e.attrib for e in root.findall(f".//{tag}")] == [
            e.attrib for e in original.findall(f".//{tag}")
        ]
    assert ROBOT.read_bytes() == before


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_native_reset_step_and_sensor_snapshots(handedness, monkeypatch):
    env = make_env(handedness)
    directory = Path(env.scene_directory.name)
    try:
        obs, _ = env.reset(seed=4)
        assert env.action_space.shape == (29,)
        assert obs["obs"].shape == (2, 127)
        robot_start = env.scene["robot"].data.root_link_pos_w.copy()
        ball_start = env.scene["ball"].data.root_link_pos_w.copy()
        contacts = env.scene.bind_sensor_data(("ball_bat", "left_support", "bat_fixture_force"))
        assert contacts.read().shape == (2, 615)

        def forbid_pose_write(*args, **kwargs):
            pytest.fail("policy steps must not write root or joint state")

        with monkeypatch.context() as guard:
            for name in (
                "write_root_state_to_sim",
                "write_root_link_pose_to_sim",
                "write_root_link_velocity_to_sim",
                "write_joint_state_to_sim",
            ):
                guard.setattr(Entity, name, forbid_pose_write)
            for _ in range(10):
                state = env.step(np.zeros((2, 29), dtype=np.float32))
                assert np.isfinite(state.obs["obs"]).all()
                assert not state.terminated.any()
        assert not np.array_equal(robot_start, env.scene["robot"].data.root_link_pos_w)
        assert np.all(env.scene["ball"].data.root_link_pos_w[:, 0] < ball_start[:, 0])
        assert np.all(env.scene["ball"].data.root_link_pos_w[:, 2] < ball_start[:, 2])
        env.reset(seed=4)
        np.testing.assert_allclose(env.scene["robot"].data.root_link_pos_w, robot_start)
        np.testing.assert_allclose(env.scene["ball"].data.root_link_pos_w, ball_start)
        model = mujoco.MjModel.from_xml_path(str(directory / "cricket.xml"))
        assert (model.nq, model.nv, model.nu) == (43, 41, 29)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        assert all(contact.dist >= 0 for contact in data.contact)
    finally:
        env.close()
    assert not directory.exists()


def test_invalid_hand_rejected_before_materialization(tmp_path):
    with pytest.raises(ValueError, match="handedness must be right or left"):
        build_scene(ROBOT, tmp_path / "invalid.xml", "both")
    assert not (tmp_path / "invalid.xml").exists()


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_balance_v1_native_reward_and_observation_contract(handedness):
    env = make_env(handedness, "g1_cricket_balance_v1/mujoco")
    try:
        obs, _ = env.reset(seed=4)
        assert obs["obs"].shape == (2, 130)
        assert env.max_episode_length == 300
        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert np.isfinite(state.reward).all()
        assert np.isfinite(state.obs["obs"]).all()
    finally:
        env.close()


@pytest.mark.parametrize("handedness", ["right", "left"])
def test_balance_v2_guard_and_reset_clearance(handedness):
    env = make_env(handedness, "g1_cricket_balance_v2/mujoco")
    try:
        obs, _ = env.reset(seed=4)
        assert obs["obs"].shape == (2, 130)
        guard = IncidentalBatSupport(None, env)
        assert not guard(env).any()
        names = env.cfg.bat_guard_sensor_names
        assert "bat_pitch" in names
        assert f"bat_robot_{handedness}_wrist_yaw_link" not in names
        assert "bat_robot_pelvis" in names
        model = mujoco.MjModel.from_xml_path(str(Path(env.scene_directory.name) / "cricket.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pitch")
        for geom in ("bat_blade", "bat_handle"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            assert mujoco.mj_geomDistance(model, data, geom_id, ground, 10, None) > 0.04
        blade = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bat_blade")
        hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{handedness}_hip_roll_link")
        for geom_id in np.flatnonzero(model.geom_bodyid == hip):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                assert mujoco.mj_geomDistance(model, data, blade, geom_id, 10, None) > 0.04
        for _ in range(env.max_episode_length):
            state = env.step(np.zeros((2, 29), dtype=np.float32))
            if state.terminated.any():
                break
        assert guard(env).all()
        assert state.terminated.all()
    finally:
        env.close()


@pytest.mark.parametrize("handedness", ["right", "left"])
@pytest.mark.parametrize(
    "rotation",
    [(0, 0, 0), (10, 0, 0), (-10, 0, 0), (0, 10, 0), (0, -10, 0), (0, 10, 60), (0, 0, 60)],
)
def test_balance_v3_gravity_is_in_torso_frame(handedness, rotation):
    env = make_env(handedness, "g1_cricket_balance_v3/mujoco", rotation)
    try:
        obs, _ = env.reset(seed=4)
        assert obs["obs"].shape == (2, 130)
        roll, pitch, _ = np.deg2rad(rotation)
        expected = [np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)]
        gravity = obs["obs"][:, 58:61]
        np.testing.assert_allclose(gravity, np.tile(expected, (2, 1)), atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(gravity, axis=1), 1, atol=1e-6)
    finally:
        env.close()


def test_legacy_v2_gravity_and_dynamics_are_preserved():
    old = make_env("right", "g1_cricket_balance_v2/mujoco", (0, 10, 60))
    new = make_env("right", "g1_cricket_balance_v3/mujoco", (0, 10, 60))
    try:
        original, _ = old.reset(seed=4)
        corrected, _ = new.reset(seed=4)
        legacy_gravity = -old.scene.bind_sensor_data(("torso_upvector",)).read()
        np.testing.assert_allclose(original["obs"][:, 58:61], legacy_gravity)
        assert not np.allclose(original["obs"][:, 58:61], corrected["obs"][:, 58:61])
        for _ in range(5):
            action = np.zeros((2, 29), dtype=np.float32)
            a, b = old.step(action), new.step(action)
            np.testing.assert_array_equal(a.reward, b.reward)
            np.testing.assert_array_equal(a.terminated, b.terminated)
            np.testing.assert_array_equal(
                old.scene["robot"].data.root_link_pose_w, new.scene["robot"].data.root_link_pose_w
            )
            np.testing.assert_array_equal(
                old.scene["robot"].data.root_link_vel_w, new.scene["robot"].data.root_link_vel_w
            )
            np.testing.assert_array_equal(
                old.scene["robot"].data.joint_pos, new.scene["robot"].data.joint_pos
            )
    finally:
        old.close()
        new.close()
