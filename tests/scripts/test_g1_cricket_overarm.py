import numpy as np
import pytest
from audit_g1_cricket_pitch_contact import owner_config as parent_config
from omegaconf import OmegaConf
from probe_g1_cricket_overarm import owner_config, reach_sample, target_at
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.overarm import (
    absolute_targets,
    actions_for_targets,
)
from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK


def test_owner_changes_only_selected_arm_action_reference():
    parent = OmegaConf.to_container(parent_config(), resolve=True)
    new = OmegaConf.to_container(owner_config(), resolve=True)
    new["env"]["actions"]["residual"]["_target_"] = parent["env"]["actions"]["residual"]["_target_"]
    assert parent == new
    serial = OmegaConf.to_container(owner_config("mujoco"), resolve=True)
    native = OmegaConf.to_container(owner_config(), resolve=True)
    assert native["env"].pop("mujoco_substep_engine") == "mjbatch"
    assert native == serial


@pytest.mark.parametrize("hand", ["right", "left"])
def test_absolute_target_bounds_inverse_and_prior_independence(monkeypatch, hand):
    env = make_env(owner_config(), hand, dt=0.0000625, count=2)
    try:
        env.reset(seed=6301)
        term = env.action_manager.get_term("residual")
        neutral = env.scene["robot"].data.default_joint_pos[:, term.arm_ids]
        limits = term.joint_limits[term.arm_ids]
        np.testing.assert_array_equal(absolute_targets(np.zeros((2, 7)), neutral, limits), neutral)
        np.testing.assert_allclose(
            absolute_targets(np.full((2, 7), -100), neutral, limits), [limits[:, 0]] * 2
        )
        np.testing.assert_allclose(
            absolute_targets(np.full((2, 7), 100), neutral, limits), [limits[:, 1]] * 2
        )
        action = np.zeros((2, 8), np.float32)
        action[:, :7] = np.linspace(-1, 1, 14).reshape(2, 7)
        expected = absolute_targets(action[:, :7], neutral, limits)
        np.testing.assert_allclose(
            actions_for_targets(expected, neutral, limits), action[:, :7], atol=1e-6
        )
        with pytest.raises(ValueError, match="strictly inside"):
            actions_for_targets(limits[:, 0], neutral[0], limits)
        before = env.get_physics_state_snapshot().copy()
        other = np.setdiff1d(np.arange(29), term.arm_ids)
        for scale in (0.0, 4.0):
            monkeypatch.setattr(
                term.session, "run", lambda *args: [np.full((1, 29), scale, np.float32)]
            )
            term.process_actions(action)
            np.testing.assert_allclose(term.processed_action[:, term.arm_ids], expected)
            sdk = np.empty((2, 29), np.float32)
            sdk[:, POLICY_TO_SDK] = term.baseline_action
            baseline = env.scene["robot"].data.default_joint_pos + 0.25 * sdk
            np.testing.assert_array_equal(term.processed_action[:, other], baseline[:, other])
            np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
        action[0, 7] = 0.5001
        term.process_actions(action)
        np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
        np.testing.assert_array_equal(term.released, [True, False])
        np.testing.assert_array_equal(term.release_physics[0], before[0])
        env.reset(env_ids=np.array([0]))
        assert not term.released.any()
        assert term.action_dim == 8
    finally:
        env.close()


def test_reach_schedule_and_sampled_gate():
    neutral = np.array([0.3, -0.2, 0, 0.97, 0, 0, 0])
    for pitch in (-2.45, -2.8):
        for elbow in (1.1, 1.4):
            np.testing.assert_array_equal(target_at(neutral, "right", pitch, elbow, 0), neutral)
            settled = target_at(neutral, "right", pitch, elbow, 30)
            assert settled[0] == neutral[0]
            assert settled[3] == elbow
            np.testing.assert_allclose(
                target_at(neutral, "right", pitch, elbow, 80), [pitch, -0.35, 0, elbow, 0, 0, 0]
            )
            np.testing.assert_allclose(target_at(neutral, "right", pitch, elbow, 199), neutral)
    assert reach_sample(0.75, 0.25, 0.1)
    assert not reach_sample(0.749, 0.3, 0)
    assert not reach_sample(0.8, 0.249, 0)
    assert not reach_sample(0.8, 0.3, 0.101)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_exact_executor_parity_and_no_step_pose_writes(monkeypatch, hand):
    envs = [
        make_env(owner_config(engine), hand, dt=0.0000625, engine=engine)
        for engine in ("mujoco", "mjbatch")
    ]
    try:
        for env in envs:
            env.reset(seed=6301)
        for tick in range(8):
            action = np.zeros((1, 8), np.float32)
            action[0, :7] = 0.1 * np.sin(tick + np.arange(7))
            action[0, 7] = float(tick == 5)
            with monkeypatch.context() as patch:

                def forbid(*args, **kwargs):
                    raise AssertionError("step cannot write joint/root states")

                for env in envs:
                    for name in ("robot", "ball"):
                        patch.setattr(env.scene[name], "write_root_state_to_sim", forbid)
                        patch.setattr(env.scene[name], "write_joint_state_to_sim", forbid)
                states = [env.step(action) for env in envs]
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            for key in states[0].obs:
                np.testing.assert_array_equal(states[0].obs[key], states[1].obs[key])
            assert states[0].obs["obs"].shape == (1, 122)
            for field in ("reward", "terminated", "truncated"):
                np.testing.assert_array_equal(getattr(states[0], field), getattr(states[1], field))
            for env in envs:
                model = env.get_playback_model()
                sensor_names = tuple(model.sensor(i).name for i in range(model.nsensor))
                assert model.nu == 29 and model.nq == 43
            np.testing.assert_array_equal(
                envs[0].scene.bind_sensor_data(sensor_names).read(),
                envs[1].scene.bind_sensor_data(sensor_names).read(),
            )
    finally:
        for env in envs:
            env.close()
