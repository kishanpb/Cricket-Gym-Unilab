"""Native prior owner contract, independent of any learned cricket claim."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.prior import (
    POLICY_TO_SDK,
    SDK_DEFAULT,
    SDK_KD,
    SDK_KP,
    PriorContactGuard,
    build_prior_scene,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("onnxruntime")
from evaluate_g1_cricket_prior import make_env


@pytest.mark.parametrize("hand", ["none", "right", "left"])
@pytest.mark.parametrize("mount", ["legacy", "forward_down"])
def test_prior_scene_preserves_physical_robot_and_clearance(tmp_path, hand, mount):
    source = ROOT / "src/unilab/assets/robots/g1/g1.xml"
    before = source.read_bytes()
    output = tmp_path / "prior.xml"
    names = build_prior_scene(source, output, hand, mount)
    original, modified = ET.fromstring(before), ET.parse(output).getroot()
    for tag in ("joint", "inertial"):
        assert [e.attrib for e in original.findall(f".//{tag}")] == [
            e.attrib for e in modified.findall(f".//{tag}")
        ]
    assert [e.attrib for e in original.findall("contact/*")] == [
        e.attrib for e in modified.findall("contact/*")
    ]
    for old, new in zip(
        original.findall("actuator/*"), modified.findall("actuator/*"), strict=True
    ):
        assert {k: v for k, v in old.attrib.items() if k not in {"kp", "kv"}} == {
            k: v for k, v in new.attrib.items() if k not in {"kp", "kv"}
        }
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.neq == 0 and np.all(model.body_gravcomp == 0)
    assert (model.nq, model.nv, model.nu) == (43, 41, 29)
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], SDK_KP)
    np.testing.assert_array_equal(model.actuator_biasprm[:, 2], -SDK_KD)
    np.testing.assert_array_equal(model.key_ctrl[0], SDK_DEFAULT)
    np.testing.assert_array_equal(model.key_qpos[0, 7:36], SDK_DEFAULT)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    assert all(contact.dist >= 0 for contact in data.contact)
    assert "prior_ground_pelvis" in names
    assert "prior_wicket_1_pelvis" in names
    assert "prior_wicket_1_left_ankle_roll_link" in names
    assert not any("ankle_roll" in n for n in names if n.startswith("prior_ground"))
    assert (modified.find(".//body[@name='cricket_bat']") is None) == (hand == "none")
    assert source.read_bytes() == before


@pytest.mark.parametrize("hand", ["none", "right", "left"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_prior_history_mapping_reset_and_no_policy_pose_writes(hand, version, monkeypatch):
    env = make_env(hand, version)
    try:
        obs, _ = env.reset(seed=4201)
        robot = env.scene["robot"]
        assert obs["obs"].shape == (1, 480)
        assert env.step_dt == 0.02 and env.max_episode_length == 500
        initial = obs["obs"].copy()
        offsets = (robot.data.joint_pos - robot.data.default_joint_pos)[0, POLICY_TO_SDK]
        expected = np.concatenate(
            [
                np.tile(term, 5)
                for term in (
                    np.zeros(3),
                    [0, 0, -1],
                    np.zeros(3),
                    offsets,
                    np.zeros(29),
                    np.zeros(29),
                )
            ]
        )
        np.testing.assert_allclose(initial[0], expected, atol=1e-7)
        policy_action = np.linspace(-0.4, 0.4, 29, dtype=np.float32)[None]
        action = np.empty_like(policy_action)
        action[:, POLICY_TO_SDK] = policy_action

        def forbid(*args, **kwargs):
            pytest.fail("step-time pose overwrite")

        with monkeypatch.context() as guard:
            for name in (
                "write_root_state_to_sim",
                "write_joint_state_to_sim",
                "write_root_link_pose_to_sim",
                "write_root_link_velocity_to_sim",
            ):
                guard.setattr(Entity, name, forbid)
            state = env.step(action)
        history = state.obs["obs"][0, -145:].reshape(5, 29)
        np.testing.assert_array_equal(history[:-1], np.zeros((4, 29)))
        np.testing.assert_array_equal(history[-1], policy_action[0])
        term = env.action_manager.get_term("joint_pos")
        np.testing.assert_allclose(
            term.processed_action[0], SDK_DEFAULT + 0.25 * action[0], atol=1e-7
        )
        term.process_actions(np.full((1, 29), 2, dtype=np.float32))
        assert term.processed_action[0, 18] > 1
        repeated, _ = env.reset(seed=4201)
        np.testing.assert_array_equal(repeated["obs"], initial)
        different, _ = env.reset(seed=4202)
        assert not np.array_equal(different["obs"], initial)
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_forward_down_changes_only_fixed_mount(tmp_path, hand):
    source = ROOT / "src/unilab/assets/robots/g1/g1.xml"
    old, new = tmp_path / "old.xml", tmp_path / "new.xml"
    build_prior_scene(source, old, hand)
    build_prior_scene(source, new, hand, "forward_down")
    old_tree, new_tree = ET.parse(old), ET.parse(new)
    bat = new_tree.find(".//body[@name='cricket_bat']")
    assert bat is not None
    bat.attrib.pop("quat")
    assert ET.tostring(old_tree.getroot()) == ET.tostring(new_tree.getroot())
    model = mujoco.MjModel.from_xml_path(str(new))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    blade_direction = data.body("cricket_bat").xmat.reshape(3, 3) @ [0, 0, -1]
    np.testing.assert_allclose(blade_direction, [2**-0.5, 0, -(2**-0.5)], atol=1e-10)
    pitch = model.geom("pitch").id
    for name in ("bat_blade", "bat_handle"):
        assert mujoco.mj_geomDistance(model, data, model.geom(name).id, pitch, 10, None) > 0.1


@pytest.mark.parametrize("hand,capacity", [("none", 32), ("right", 4), ("left", 4)])
def test_prior_guard_rejects_presence_without_force_and_overflow(hand, capacity):
    env = make_env(hand)
    try:
        env.reset(seed=4201)
        guard = PriorContactGuard(None, env)
        records = np.zeros_like(guard.contacts.read())
        guard.contacts = SimpleNamespace(read=lambda: records)
        assert not guard(env).any()
        records[0, 0] = 1
        assert guard(env).all()
        records[0, 0] = capacity + 1
        with pytest.raises(RuntimeError, match="invalid prior contact"):
            guard(env)
    finally:
        env.close()
