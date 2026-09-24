"""Bimanual topology, offline reference export and physics-only tracking."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.bimanual import (
    build_bimanual_scene,
    retarget_batting,
    support_feedforward,
)
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.tracking import export_reference

ROOT = Path(__file__).resolve().parents[2]
ROBOT = ROOT / "src/unilab/assets/robots/g1/g1.xml"


@pytest.fixture(params=["right", "left"])
def scene(request, tmp_path):
    output = tmp_path / "scene.xml"
    build_bimanual_scene(ROBOT, output, request.param)
    return request.param, output, mujoco.MjModel.from_xml_path(str(output))


def test_grips_and_original_robot_constraints(scene):
    hand, path, model = scene
    root, original = ET.parse(path), ET.parse(ROBOT)
    top = "left" if hand == "right" else "right"
    assert root.find(f".//body[@name='{top}_wrist_yaw_link']/body[@name='cricket_bat']") is not None
    eq = root.find("equality/connect")
    assert eq.get("site1") == "bat_lower_grip"
    assert eq.get("site2") == f"{hand}_palm"
    assert model.neq == 1 and model.nu == 29
    for tag in ("joint", "inertial"):
        assert [x.attrib for x in root.findall(f".//{tag}")] == [
            x.attrib for x in original.findall(f".//{tag}")
        ]
    assert [x.get("forcerange") for x in root.findall("actuator/*")] == [
        x.get("forcerange") for x in original.findall("actuator/*")
    ]
    assert model.geom("bat_handle").contype == 0
    assert model.geom("bat_blade").contype != 0
    assert model.geom(f"{hand}_hand_collision").contype != 0


def test_retarget_and_motion_export(scene, tmp_path):
    hand, _, model = scene
    bodies = model.body_pos.copy()
    reference = retarget_batting(model, np.array([0.0, 1.45, 3.0]), hand)
    joints = np.array([model.joint(n).id for n in SDK_JOINTS])
    q = reference["qpos"][:, model.jnt_qposadr[joints]]
    assert np.isfinite(q).all()
    assert (q >= model.jnt_range[joints, 0]).all()
    assert (q <= model.jnt_range[joints, 1]).all()
    np.testing.assert_array_equal(model.body_pos, bodies)
    assert max(row["second_grip_error_m"] for row in reference["errors"]) < 0.01
    path = tmp_path / "motion.npz"
    export_reference(model, reference["qpos"], 50, path)
    motion = np.load(path)
    assert motion["joint_pos"].shape == (3, 29)
    assert motion["body_pos_w"].shape == (3, model.nbody, 3)
    np.testing.assert_allclose(
        motion["body_pos_w"][:, model.body("pelvis").id], reference["qpos"][:, :3]
    )


def test_export_world_angular_velocity_for_rotated_root(scene, tmp_path):
    _, _, model = scene
    poses = np.repeat(model.key_qpos[:1], 3, axis=0)
    yaw = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    for pose, angle in zip(poses, (0, 0.02, 0.04), strict=True):
        pitch = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])
        mujoco.mju_mulQuat(pose[3:7], yaw, pitch)
    output = tmp_path / "rotated.npz"
    export_reference(model, poses, 50, output)
    observed = np.load(output)["body_ang_vel_w"][:, model.body("pelvis").id]
    np.testing.assert_allclose(observed, np.tile([-1, 0, 0], (3, 1)), atol=1e-10)


def test_grounded_reference_and_static_support(scene):
    hand, _, model = scene
    reference = retarget_batting(model, np.array([0.0, 0.02, 0.04, 1.8]), hand, grounded=True)
    assert min(r["minimum_tracked_clearance_m"] for r in reference["errors"]) > 0.005
    for pose in reference["qpos"]:
        torque, residual = support_feedforward(model, pose)
        np.testing.assert_allclose(residual, 0, atol=1e-6)
        assert np.isfinite(torque).all()
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    mujoco.mj_forward(model, data)
    gaps = [
        mujoco.mj_geomDistance(
            model, data, model.geom(f"{s}_foot4_collision").id, model.geom("pitch").id, 1, None
        )
        for s in ("left", "right")
    ]
    assert max(abs(g) for g in gaps) < 0.001


@pytest.mark.parametrize("hand", ["right", "left"])
def test_supported_tracking_targets_use_same_reference_and_original_limits(hand):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=["task=g1_cricket_supported_tracking/mjbatch", f"env.handedness={hand}"],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    env = registry.make(
        "G1CricketBimanualTracking", num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.reset(seed=1)
        action = env.action_manager.get_term("reference")
        command = env.command_manager.get_term("motion")
        expected = (
            command.joint_pos
            + action.velocity_gain * command.joint_vel
            + action.gravity_offset[command.time_steps]
        )
        action.process_actions(np.zeros((2, 29)))
        np.testing.assert_allclose(
            action.target, np.clip(expected, action.limits[:, 0], action.limits[:, 1]), atol=1e-6
        )
        action.process_actions(np.full((2, 29), 10.0))
        assert (action.target >= action.limits[:, 0]).all()
        assert (action.target <= action.limits[:, 1]).all()
        assert not owner.algo.algorithm.disable_finite_checks
    finally:
        env.close()


@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("balanced", [False, True])
def test_supported_tracking_substep_replay(engine, hand, balanced):
    from evaluate_g1_cricket_tracking import TrackingReplay

    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                f"task=g1_cricket_{'balanced' if balanced else 'supported'}_tracking/{engine}",
                f"env.handedness={hand}",
            ],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    env = registry.make(
        "G1CricketBimanualTracking", num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.reset(seed=1)
        replay = TrackingReplay(env)
        for _ in range(150 if balanced else 3):
            initial = env.get_physics_state_snapshot()[0].copy()
            env.step(np.zeros((1, 29), dtype=np.float32))
            result = replay.measure(env, initial)
            assert result["substeps"] == 20
            assert result["unexpected_contacts"] == []
            assert all(np.isfinite(value) for value in result["peaks"].values())
            assert result["peaks"]["motor_force_fraction"] <= 1
            if balanced:
                assert result["peaks"]["hard_joint_limit_excess_rad"] <= 0.0001
                assert result["peaks"]["grip_separation_m"] < 0.006
                assert env.scene["robot"].data.root_link_pos_w[0, 2] > 0.65
        if balanced:
            assert env.state.truncated[0] and not env.state.terminated[0]
        wrong = initial.copy()
        wrong[1] += 0.01
        with pytest.raises(AssertionError):
            replay.measure(env, wrong)
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
@pytest.mark.parametrize("owner_kind", ["bimanual", "supported", "balanced"])
def test_whole_body_owner_has_no_mid_episode_pose_writes(
    hand, engine, owner_kind, monkeypatch, tmp_path
):
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                f"task=g1_cricket_{owner_kind}_tracking/{engine}",
                f"env.handedness={hand}",
            ],
        )
    motion = np.load(ROOT / owner.env.commands.motion.params.motion_file)
    short_clip = tmp_path / "short_clip.npz"
    np.savez_compressed(
        short_clip,
        **{name: value if name == "fps" else value[:5] for name, value in motion.items()},
    )
    owner.env.commands.motion.params.motion_file = str(short_clip)
    if owner_kind != "bimanual":
        with np.load(ROOT / owner.env.actions.reference.reference_file) as full_reference:
            short_reference = tmp_path / "short_reference.npz"
            np.savez_compressed(short_reference, qpos=full_reference["qpos"][:5])
        owner.env.actions.reference.reference_file = str(short_reference)
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    env = registry.make(
        "G1CricketBimanualTracking", num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        obs, _ = env.reset(seed=4)
        np.testing.assert_array_equal(obs["obs"][0], obs["obs"][1])
        assert env.action_space.shape == (29,)
        assert env.command_manager.get_term("motion").cfg.params.truncate_on_clip_end
        reference = env.command_manager.get_term("motion").joint_pos.copy()
        action = np.zeros((2, 29), dtype=np.float32)
        action[:, 0] = 0.5
        expected = reference + 0.25 * action
        if owner_kind != "bimanual":
            term = env.action_manager.get_term("reference")
            command = env.command_manager.get_term("motion")
            expected += term.velocity_gain * command.joint_vel
            expected += term.gravity_offset[command.time_steps]
            if owner_kind == "balanced":
                from unilab.tasks.manipulation.g1_cricket.tracking import ankle_balance

                for i in range(2):
                    correction = ankle_balance(
                        term.reference_root_quaternion[command.time_steps[i]],
                        env.scene["robot"].data.root_link_quat_w[i],
                        env.scene["robot"].data.root_link_ang_vel_b[i],
                        4.0,
                    )
                    expected[i, [4, 10]] += correction[1]
                    expected[i, [5, 11]] += correction[0]
            expected = np.clip(expected, term.control_limits[:, 0], term.control_limits[:, 1])

        def forbid(*args, **kwargs):
            pytest.fail("pose writes are permitted only at episode reset")

        for name in (
            "write_joint_state_to_sim",
            "write_root_state_to_sim",
            "write_root_link_pose_to_sim",
            "write_root_link_velocity_to_sim",
        ):
            monkeypatch.setattr(Entity, name, forbid)
        state = env.step(action)
        target = env.action_manager.get_term("reference").target
        np.testing.assert_allclose(target, expected, atol=1e-6)
        assert np.isfinite(state.obs["obs"]).all()
        while not env.state.truncated[0] and not env.state.terminated[0]:
            env.step(action)
        assert env.termination_manager.get_term("clip_end")[0]
    finally:
        env.close()
