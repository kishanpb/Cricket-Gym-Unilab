"""Frozen-prior separation and reset-only delivery for learned arm corrections."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK
from unilab.tasks.manipulation.g1_cricket.residual import (
    RESIDUAL_LIMITS,
    TOSS_OFFSETS,
    BattingReward,
    failure_penalty,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("onnxruntime")
from evaluate_g1_cricket_prior import make_env as make_prior
from evaluate_g1_cricket_residual import CricketReplay

ASSETS = Path.home() / "Library/Caches/unitree_rl_lab/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
pytestmark = pytest.mark.skipif(
    not (ASSETS / "policy.onnx").exists(), reason="external prior not cached"
)


def make_env(hand="right", num_envs=1, toss=True, horizon=2):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=g1_cricket_residual_v1/mujoco",
                f"env.handedness={hand}",
                "env.auto_reset=false",
                f"env.max_episode_seconds={horizon}",
                f"env.events.reset_toss.params.enabled={str(toss).lower()}",
            ],
        )
    return create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override(),
    )


def root_state(entity):
    return np.concatenate((entity.data.root_link_pose_w, entity.data.root_link_vel_w), axis=1)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("seed", range(4201, 4209))
def test_zero_residual_exactly_preserves_prior_history_and_dynamics(hand, seed, monkeypatch):
    prior, env = make_prior(hand, "v2"), make_env(hand, toss=False, horizon=10)
    try:
        prior_obs, _ = prior.reset(seed=seed)
        obs, _ = env.reset(seed=seed)
        assert obs["obs"].shape == (1, 115)
        term = env.action_manager.get_term("residual")
        assert term.action_dim == 7 and env.max_episode_length == 500

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
            for _ in range(env.max_episode_length):
                native_prior = env.observation_manager.compute(update_history=False)["policy"]
                np.testing.assert_array_equal(native_prior, prior_obs["obs"])
                baseline = term.session.run(
                    ["actions"], {"obs": prior_obs["obs"].astype(np.float32)}
                )[0]
                sdk = np.empty_like(baseline)
                sdk[:, POLICY_TO_SDK] = baseline
                prior_state = prior.step(sdk)
                state = env.step(np.zeros((1, 7), dtype=np.float32))
                prior_obs = prior_state.obs
                assert not state.terminated.any()
                np.testing.assert_array_equal(term.baseline_action, baseline)
                np.testing.assert_array_equal(
                    term.processed_action,
                    prior.action_manager.get_term("joint_pos").processed_action,
                )
                for name in ("robot", "ball"):
                    np.testing.assert_array_equal(
                        root_state(env.scene[name]), root_state(prior.scene[name])
                    )
                np.testing.assert_array_equal(
                    env.scene["robot"].data.joint_pos, prior.scene["robot"].data.joint_pos
                )
        assert state.truncated[0] and prior_state.truncated[0]
    finally:
        prior.close()
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_residual_caps_and_partial_reset_leave_other_rows_untouched(hand):
    env = make_env(hand, 2)
    try:
        env.reset(seed=10)
        term = env.action_manager.get_term("residual")
        term.process_actions(np.full((2, 7), 2, dtype=np.float32))
        sdk = np.empty_like(term.baseline_action)
        sdk[:, POLICY_TO_SDK] = term.baseline_action
        base = env.scene["robot"].data.default_joint_pos + 0.25 * sdk
        expected = base.copy()
        expected[:, term.arm_ids] += RESIDUAL_LIMITS
        np.testing.assert_array_equal(term.processed_action, expected)
        previous = term.baseline_action.copy()
        term.reset(np.array([0]))
        assert not term.raw_action[0].any() and not term.baseline_action[0].any()
        np.testing.assert_array_equal(term.baseline_action[1], previous[1])
        np.testing.assert_array_equal(term.processed_action[1], expected[1])
    finally:
        env.close()


@pytest.mark.parametrize(
    "hand,sign,center", [("right", 1, 0.0432396123), ("left", -1, -0.0432296123)]
)
def test_toss_is_repeatable_and_uses_only_predeclared_lanes(hand, sign, center):
    env = make_env(hand, 8)
    try:
        obs, _ = env.reset(seed=11)
        initial = root_state(env.scene["ball"]).copy()
        repeated, _ = env.reset(seed=11)
        np.testing.assert_array_equal(obs["obs"], repeated["obs"])
        np.testing.assert_array_equal(initial, root_state(env.scene["ball"]))
        for y in initial[:, 1]:
            assert any(abs(y - (center + sign * offset)) < 1e-7 for offset in TOSS_OFFSETS)
        np.testing.assert_allclose(initial[:, 7], -2.5)
        np.testing.assert_allclose(initial[:, 2], 0.8717760803)
    finally:
        env.close()


def test_contact_bonus_is_one_shot_and_failure_is_discrete():
    env = make_env()
    try:
        env.reset(seed=1)
        reward = BattingReward(None, env)
        records = np.zeros((1, 68))
        reward.contact = SimpleNamespace(read=lambda: records)
        shaping = reward(env)
        records[0, 0] = 1
        np.testing.assert_allclose((reward(env) - shaping) * env.step_dt, 5)
        np.testing.assert_allclose(reward(env), shaping)
        reward.reset(np.array([0]))
        np.testing.assert_allclose((reward(env) - shaping) * env.step_dt, 5)
        fake = SimpleNamespace(
            step_dt=0.02, termination_manager=SimpleNamespace(terminated=np.array([True, False]))
        )
        np.testing.assert_array_equal(failure_penalty(fake) * fake.step_dt, [1, 0])
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_residual_replay_matches_every_native_endpoint_including_contact(hand):
    env = make_env(hand)
    try:
        env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [0.0]
        env.reset(seed=4302)
        replay = CricketReplay(env)
        contact_seen = False
        for _ in range(30):
            state, trajectory, forces, presence, peaks, fixture, contacts = replay.step(
                env, np.zeros((1, 7), dtype=np.float32)
            )
            assert trajectory.shape == (10, 85) and forces.shape == (10, 29)
            assert fixture.shape == (10, 6)
            assert presence.shape == peaks.shape
            contact_seen |= any(contacts)
            if state.terminated[0]:
                break
        assert contact_seen
        assert replay.state_error == replay.sensor_error == 0
    finally:
        env.close()
