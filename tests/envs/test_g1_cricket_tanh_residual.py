"""Smooth executed residuals preserve raw PPO actions and zero-action dynamics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK
from unilab.tasks.manipulation.g1_cricket.residual import RESIDUAL_LIMITS
from unilab.tasks.manipulation.g1_cricket.scene import BALL_CONTACT_NAMES
from unilab.utils.sim2sim import CrossBackendIncompatibleError, resolve_sim2sim_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_g1_cricket_residual import CricketReplay


def owner(name):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_{name}/mujoco"])


def make_env(name, hand, nenv=1, dt=0.00025):
    cfg = owner(name)
    override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False, sim_dt=dt)
    env = create_env(cfg, num_envs=nenv, env_cfg_override=override)
    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [0.0]
    env.reset(seed=4301)
    return env


def test_owner_changes_only_action_term_and_rejects_old_checkpoint_contract():
    old, new = [owner(name) for name in ("impact_events_v1", "tanh_v1")]
    with pytest.raises(CrossBackendIncompatibleError, match="env.actions"):
        resolve_sim2sim_config(ROOT / "g1_cricket_results/impact_events_v1/right", new)
    before, after = [OmegaConf.to_container(cfg, resolve=True) for cfg in (old, new)]
    assert before["env"]["actions"]["residual"].pop("_target_").endswith("FrozenPriorResidualCfg")
    assert after["env"]["actions"]["residual"].pop("_target_").endswith("TanhPriorResidualCfg")
    assert before == after


@pytest.mark.parametrize("hand", ["right", "left"])
def test_mapping_limits_raw_alias_history_prior_and_partial_reset(hand):
    levels = np.array([-1e6, -2, -1, -0.1, 0, 0.1, 1, 2, 1e6], dtype=np.float32)
    envs = [make_env(name, hand, len(levels)) for name in ("impact_events_v1", "tanh_v1")]
    try:
        tensor = torch.from_numpy(np.repeat(levels[:, None], 7, axis=1))
        action = tensor.numpy()
        original = action.copy()
        for env in envs:
            env.action_manager.process_action(action)
        old, new = [env.action_manager.get_term("residual") for env in envs]
        expected = np.tanh(original) * RESIDUAL_LIMITS
        np.testing.assert_array_equal(new.residual_radians, expected)
        np.testing.assert_array_equal(expected, -expected[::-1])
        assert (np.diff(expected, axis=0) > 0).all()
        assert np.isfinite(expected).all() and (np.abs(expected) <= RESIDUAL_LIMITS).all()
        np.testing.assert_array_equal(tensor.numpy(), original)
        np.testing.assert_array_equal(new.raw_action, original)
        np.testing.assert_array_equal(new.baseline_action, old.baseline_action)
        sdk = np.empty_like(new.baseline_action)
        sdk[:, POLICY_TO_SDK] = new.baseline_action
        target = envs[1].scene["robot"].data.default_joint_pos + 0.25 * sdk
        target[:, new.arm_ids] += expected
        np.testing.assert_array_equal(new.processed_action, target)
        other = np.setdiff1d(np.arange(29), new.arm_ids)
        np.testing.assert_array_equal(
            new.processed_action[:, other], old.processed_action[:, other]
        )
        for raw in (1.5, 1.6):
            actions = np.full_like(original, raw)
            for env in envs:
                env.action_manager.process_action(actions)
            for name in ("action", "prev_action", "prev_prev_action"):
                np.testing.assert_array_equal(
                    getattr(envs[0].action_manager, name), getattr(envs[1].action_manager, name)
                )
            np.testing.assert_array_equal(new.baseline_action, old.baseline_action)
            if raw == 1.5:
                clipped, smooth = old.residual_radians.copy(), new.residual_radians.copy()
        np.testing.assert_array_equal(old.residual_radians, clipped)
        assert (new.residual_radians > smooth).all()
        untouched = new.processed_action[1:].copy()
        new.reset(np.array([0]))
        assert not new.raw_action[0].any() and not new.residual_radians[0].any()
        np.testing.assert_array_equal(new.processed_action[1:], untouched)
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("dt", [0.00025, 0.000125])
def test_full_zero_action_native_physics_rewards_and_observations_exact(hand, dt):
    envs = [make_env(name, hand, dt=dt) for name in ("impact_events_v1", "tanh_v1")]
    try:
        contacts = [env.scene.bind_sensor_data(BALL_CONTACT_NAMES) for env in envs]
        for _ in range(100):
            a, b = [env.step(np.zeros((1, 7), dtype=np.float32)) for env in envs]
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            np.testing.assert_array_equal(contacts[0].read(), contacts[1].read())
            np.testing.assert_array_equal(a.reward, b.reward)
            np.testing.assert_array_equal(a.terminated, b.terminated)
            for key in a.obs:
                np.testing.assert_array_equal(a.obs[key], b.obs[key])
        assert a.truncated[0] and b.truncated[0]
        assert a.obs["obs"].shape == (1, 115)
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_nonzero_native_replay_uses_transformed_targets(hand):
    env = make_env("tanh_v1", hand)
    try:
        replay = CricketReplay(env)
        for tick in range(20):
            action = np.full((1, 7), 0.15 * np.sin(tick), dtype=np.float32)
            state = replay.step(env, action)[0]
            term = env.action_manager.get_term("residual")
            np.testing.assert_array_equal(term.raw_action, action)
            np.testing.assert_array_equal(term.residual_radians, np.tanh(action) * RESIDUAL_LIMITS)
            if state.terminated[0]:
                break
        assert replay.state_error == replay.sensor_error == 0
    finally:
        env.close()
