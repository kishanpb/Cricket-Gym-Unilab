from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from evaluate_g1_cricket_approach import ApproachEvents, LoadedFootSlip
from g1_cricket_delivery_trial import DeliveryReplay
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.approach_peak_slip import peak_slip_cost

ROOT = Path(__file__).resolve().parents[2]


def owners():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return [
            compose("config", overrides=[f"task={name}/mjbatch"])
            for name in ("g1_cricket_approach_feedback", "g1_cricket_approach_peak_slip")
        ]


def test_peak_cost_and_single_changed_axis():
    action = SimpleNamespace(slip_peak=np.array([[2.0, 0.5], [0.0, 0.0]]))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: action))
    np.testing.assert_array_equal(peak_slip_cost(env), [4.25, 0])
    original, candidate = owners()
    original.reward.loaded_slip.func = candidate.reward.loaded_slip.func
    assert OmegaConf.to_container(original, resolve=True) == OmegaConf.to_container(
        candidate, resolve=True
    )


@pytest.mark.parametrize("hand", ["right", "left"])
def test_native_reward_change_preserves_trajectory_and_observations(hand):
    envs = []
    try:
        for owner in owners():
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override.update(handedness=hand, auto_reset=False)
            env = create_env(owner, num_envs=1, env_cfg_override=override)
            env.reset(seed=1)
            envs.append(env)
        saw_peak_penalty = False
        for _ in range(160):
            states = [env.step(np.zeros((1, 29), np.float32)) for env in envs]
            np.testing.assert_array_equal(
                envs[0].get_physics_state_snapshot(), envs[1].get_physics_state_snapshot()
            )
            terms = [env.action_manager.get_term("residual") for env in envs]
            np.testing.assert_array_equal(terms[0].processed_action, terms[1].processed_action)
            original_cost = terms[0].slip_squared.sum(axis=1)
            changed_cost = peak_slip_cost(envs[1])
            assert np.all(changed_cost >= original_cost - 1e-12)
            np.testing.assert_allclose(
                states[1].reward - states[0].reward,
                -5 * envs[0].step_dt * (changed_cost - original_cost),
                atol=1e-7,
            )
            for group in ("policy", "bowling"):
                np.testing.assert_array_equal(
                    envs[0].observation_manager.compute(update_history=False)[group],
                    envs[1].observation_manager.compute(update_history=False)[group],
                )
            saw_peak_penalty |= bool(np.any(changed_cost - original_cost > 1))
        assert saw_peak_penalty
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_peak_reward_matches_native_touchdown_contact_speed(hand):
    owner = owners()[1]
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    try:
        env.reset(seed=1)
        action = np.zeros((1, 29), np.float32)
        for _ in range(85):
            env.step(action)
        replay = DeliveryReplay(env)
        slip = LoadedFootSlip(replay.model)
        events = ApproachEvents(hand)
        speeds = []

        def observe(model, data):
            speeds.append(slip.measure(model, data)[:2])

        maximum = 0.0
        for _ in range(15):
            speeds.clear()
            replay.step(env, action, events, observer=observe)
            peak = np.asarray(speeds).max(axis=0)
            maximum = max(maximum, float(peak.max()))
            term = env.action_manager.get_term("residual")
            np.testing.assert_allclose(term.slip_peak[0], peak, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                peak_slip_cost(env)[0], np.square(peak).sum(), atol=1e-12, rtol=0
            )
        assert maximum > 1
    finally:
        env.close()
