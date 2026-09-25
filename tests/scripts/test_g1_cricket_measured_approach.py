from pathlib import Path

import numpy as np
import onnxruntime
import pytest
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[2]


def make_env(hand="right", count=1, sampling="start"):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                "task=g1_cricket_measured_approach/mjbatch",
                f"env.handedness={hand}",
                f"env.commands.motion.params.sampling_mode={sampling}",
            ],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    return create_env(owner, num_envs=count, env_cfg_override=override)


def force_frames(command, frames, monkeypatch):
    def sample(ids):
        chosen = np.broadcast_to(frames, (len(ids),)).copy().astype(np.int32)
        command.sampler._set_sampled_frames(ids, chosen)
        return chosen

    monkeypatch.setattr(command.sampler, "sample_frames", sample)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_measured_reset_targets_and_no_online_prior(hand, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no online prior or live pose writes")

    monkeypatch.setattr(onnxruntime, "InferenceSession", forbidden)
    env = make_env(hand)
    try:
        env.reset(seed=1)
        command = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("reference")
        model = env.get_playback_model()
        for frame in (0, 183, 399):
            force_frames(command, frame, monkeypatch)
            env.reset(seed=1)
            state = env.get_physics_state_snapshot()[0]
            np.testing.assert_allclose(
                state[1 : 1 + model.nq], command.poses[frame], atol=1e-7, rtol=0
            )
            np.testing.assert_allclose(
                state[1 + model.nq :], command.velocities[frame], atol=2e-6, rtol=0
            )
            action.process_actions(np.zeros((1, 29), np.float32))
            np.testing.assert_array_equal(action.processed_action[0], command.controls[frame])
            assert env.equality_constraints.get_equality_active().all()
        for entity in (env.scene["robot"], env.scene["ball"]):
            monkeypatch.setattr(entity, "write_root_state_to_sim", forbidden)
            monkeypatch.setattr(entity, "write_joint_state_to_sim", forbidden)
        state = env.step(np.zeros((1, 29), np.float32))
        assert state.truncated[0] and not state.terminated[0]
        assert env.episode_length_buf[0] == 1
        np.testing.assert_array_equal(action.processed_action[0], command.controls[399])
        assert command.time_steps[0] == 399
    finally:
        env.close()


def test_partial_reset_preserves_other_rows_and_rejects_terminal_start(monkeypatch):
    env = make_env(count=2, sampling="uniform")
    try:
        env.reset(seed=1)
        command = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("reference")
        force_frames(command, np.array([100, 200]), monkeypatch)
        env.reset(seed=1)
        env.step(np.full((2, 29), 0.1, np.float32))
        before = env.get_physics_state_snapshot()[1].copy()
        frame, raw = command.time_steps[1], action.raw_action[1].copy()
        sampled = iter((400, 399))

        def sample(ids):
            chosen = np.full(len(ids), next(sampled), np.int32)
            command.sampler._set_sampled_frames(ids, chosen)
            return chosen

        monkeypatch.setattr(command.sampler, "sample_frames", sample)
        env.reset(env_ids=np.array([0]))
        np.testing.assert_array_equal(env.get_physics_state_snapshot()[1], before)
        assert command.time_steps.tolist() == [399, frame]
        np.testing.assert_array_equal(action.raw_action[1], raw)
        np.testing.assert_array_equal(action.raw_action[0], 0)
        result = env.step(np.zeros((2, 29), np.float32))
        np.testing.assert_array_equal(result.truncated, [True, False])
    finally:
        env.close()


def test_residual_is_bounded_without_changing_measured_targets():
    env = make_env()
    try:
        env.reset(seed=1)
        command = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("reference")
        controls = command.controls.copy()
        values = np.linspace(-2, 2, 29, dtype=np.float32)[None, :]
        action.process_actions(values)
        expected = controls[0] + np.float32(0.1) * np.clip(values[0], -1, 1)
        np.testing.assert_array_equal(action.processed_action[0], expected)
        np.testing.assert_array_equal(command.controls, controls)
        np.testing.assert_array_equal(action.raw_action, values)
        assert not env.get_playback_model().actuator_ctrllimited.any()
    finally:
        env.close()
