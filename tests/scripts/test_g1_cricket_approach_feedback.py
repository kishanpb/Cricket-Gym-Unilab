from pathlib import Path

import numpy as np
import pytest
import torch
from evaluate_g1_cricket_approach_learning import DeliveryReplay, evaluate_case
from hydra import compose, initialize_config_dir
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.training import algo_config_dict

ROOT = Path(__file__).resolve().parents[2]
TEACHER = ROOT / "g1_cricket_results/approach_teacher_v1"


def make_env(hand="right", dt=0.0000625, count=1):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=["task=g1_cricket_approach_feedback/mjbatch"])
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, sim_dt=dt, auto_reset=False)
    return owner, create_env(owner, num_envs=count, env_cfg_override=override)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("seed", [5301, 5302])
@pytest.mark.parametrize("resolution,dt", [("fine", 0.0000625), ("finest", 0.00003125)])
def test_full_zero_residual_preserves_closed_loop_teacher(hand, seed, resolution, dt, monkeypatch):
    _, env = make_env(hand, dt)
    try:
        env.reset(seed=seed)
        with np.load(TEACHER / f"{hand}_{seed}_{resolution}.npz") as source:
            states, controls = source["states"], source["controls"]
        np.testing.assert_array_equal(env.get_physics_state_snapshot()[0], states[0])

        def forbidden(*args, **kwargs):
            raise AssertionError("no live pose or joint-state writes")

        for entity in (env.scene["robot"], env.scene["ball"]):
            monkeypatch.setattr(entity, "write_root_state_to_sim", forbidden)
            monkeypatch.setattr(entity, "write_joint_state_to_sim", forbidden)
        action = env.action_manager.get_term("residual")
        for index in range(400):
            result = env.step(np.zeros((1, 29), np.float32))
            np.testing.assert_array_equal(action.processed_action[0], controls[index])
            np.testing.assert_array_equal(env.get_physics_state_snapshot()[0], states[index + 1])
            assert (
                not action.released.any() and env.equality_constraints.get_equality_active().all()
            )
            assert not result.terminated.any()
        assert result.truncated[0]
    finally:
        env.close()


def test_zero_actor_and_trainable_whole_body_output():
    owner, env = make_env(count=2)
    try:
        env.reset(seed=1)
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        config = normalize_ppo_train_cfg(algo_config_dict(owner))
        config["logger"] = "none"
        runner = OnPolicyRunner(wrapped, config, log_dir=None, device="cpu")
        actor = runner.alg.actor
        obs = wrapped.get_observations()
        assert obs["policy"].shape == (2, 148)
        action = actor(obs)
        torch.testing.assert_close(action, torch.zeros_like(action), rtol=0, atol=0)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
        loss = (action - 0.01).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert actor(obs).abs().max() > 0
        term = env.action_manager.get_term("residual")
        term.process_actions(np.zeros((2, 29), np.float32))
        base = term.processed_action.copy()
        command = np.linspace(-2, 2, 29, dtype=np.float32)[None].repeat(2, axis=0)
        term.process_actions(command)
        np.testing.assert_allclose(
            term.processed_action - base, 0.1 * np.clip(command, -1, 1), atol=1e-7
        )
        env.step(np.zeros((2, 29), np.float32))
        before = env.get_physics_state_snapshot()[1].copy()
        history = env.observation_manager.compute_group("policy")[1].copy()
        env.reset(env_ids=np.array([0]))
        np.testing.assert_array_equal(env.get_physics_state_snapshot()[1], before)
        np.testing.assert_array_equal(env.observation_manager.compute_group("policy")[1], history)
        np.testing.assert_array_equal(term.raw_action[0], 0)
        np.testing.assert_array_equal(term.slip_squared[0], 0)
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_feedback_evaluation_uses_residual_action_and_cotimed_slip(hand, tmp_path, monkeypatch):
    _, env = make_env(hand)
    original = DeliveryReplay.step

    def stop_after_interval(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.terminated[:] = True
        return result

    monkeypatch.setattr(DeliveryReplay, "step", stop_after_interval)
    try:
        result = evaluate_case(
            env,
            lambda: np.zeros((1, 29), np.float32),
            tmp_path,
            "reference",
            "fine",
            False,
            action_name="residual",
        )
        assert result["exact_endpoint_and_sensor_replay"]
        assert result["evaluation_error"] is None
        assert result["maximum_slip_cost_sensor_error"] < 1e-12
        assert not result["passed"]
    finally:
        env.close()
