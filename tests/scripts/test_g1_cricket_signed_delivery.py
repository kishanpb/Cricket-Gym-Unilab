import mujoco
import numpy as np
import pytest
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from g1_cricket_signed_delivery import SignedDeliveryReplay, SignedElbowAudit
from probe_g1_cricket_overarm_guard import owner_config
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.arm_geometry import signed_elbow_angle


def test_signed_angle_crosses_straight_without_folding():
    a, origin, axis = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    for angle in (0.5, 2.9, np.pi, 3.3, 3.8):
        b = np.array([np.cos(angle), np.sin(angle), 0])
        assert signed_elbow_angle(a, origin, b, axis) == pytest.approx(angle)
    with pytest.raises(ValueError, match="axis"):
        signed_elbow_angle(a, origin, a, origin)
    with pytest.raises(ValueError, match="projected"):
        signed_elbow_angle(axis, origin, a, axis)


def test_extension_gate_catches_fold_and_preserves_all_other_failures():
    audit = SignedElbowAudit()
    audit.observe(0.0, -0.1, 2.9)
    audit.observe(0.1, 0.1, 3.1)
    audit.release(0.2, 0.5, 3.5)
    audit.observe(0.3, 0.5, 4.0)
    result = audit.result(dict(release={}, failures=["target_corridor_missed"]))
    assert result["maximum_extension_rad"] == pytest.approx(0.4)
    assert result["maximum_angle_rad"] == 3.5
    assert result["failures"] == [
        "signed_elbow_extension_above_15_degrees",
        "target_corridor_missed",
    ]
    assert result["legacy_failures_preserved"] and not result["passed"]
    assert (
        "missing_signed_elbow_release_evidence"
        in SignedElbowAudit().result(dict(release={}, failures=[]))["failures"]
    )


def test_signed_extension_unwraps_branch_crossing():
    audit = SignedElbowAudit()
    audit.observe(0, -0.1, 6.0)
    audit.observe(0.1, 0.1, 6.1)
    audit.release(0.2, 0.3, 0.2)
    assert audit.maximum_extension == pytest.approx(2 * np.pi + 0.2 - 6.1)
    assert audit.release_record["unwrapped_elbow_angle_rad"] == pytest.approx(2 * np.pi + 0.2)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_g1_geometry_matches_signed_hinge_across_poses_and_entire_joint_range(hand):
    env = make_env(owner_config(), hand, dt=0.0000625)
    try:
        m = env.get_playback_model()
        d = mujoco.MjData(m)
        ids = [m.body(f"{hand}_{n}_link").id for n in ("shoulder_yaw", "elbow", "wrist_roll")]
        joint = m.joint(f"{hand}_elbow_joint").id
        rng = np.random.default_rng(4)
        offsets = []
        for q in np.linspace(*m.jnt_range[joint], 15):
            mujoco.mj_resetDataKeyframe(m, d, 0)
            rotation = rng.normal(size=4)
            d.qpos[3:7] = rotation / np.linalg.norm(rotation)
            for name in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"):
                j = m.joint(f"{hand}_{name}_joint").id
                d.qpos[m.jnt_qposadr[j]] = rng.uniform(*m.jnt_range[j])
            d.qpos[m.jnt_qposadr[joint]] = q
            mujoco.mj_kinematics(m, d)
            angle = signed_elbow_angle(*d.xpos[ids], d.xaxis[joint])
            assert 0 < angle < 2 * np.pi
            offsets.append(angle - q)
        np.testing.assert_allclose(offsets, np.full(15, offsets[0]), atol=1e-12)
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["left", "right"])
def test_additional_geometry_observer_changes_no_physics_or_legacy_gate(hand):
    env = make_env(owner_config(), hand, dt=0.0000625)
    results = []
    try:
        for cls in (DeliveryReplay, SignedDeliveryReplay):
            env.reset(seed=6301)
            replay, events = cls(env), DeliveryEvents(hand)
            for tick in range(6):
                action = np.zeros((1, 8), np.float32)
                action[0, 7] = tick >= 4
                replay.step(env, action, events)
            results.append((env.get_physics_state_snapshot().copy(), events.finish(False)))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
        assert replay.audit.release_record["time"] == pytest.approx(0.08)
        assert replay.audit.result(results[1][1])["legacy_failures_preserved"]
    finally:
        env.close()
