from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_first_step import FirstStepReplay, evaluation_override
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.entity import Entity

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=[("right", False), ("left", False), ("right", True), ("left", True)])
def env(request):
    hand, grouped = request.param
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                "task=g1_cricket_first_step_foot_reward/mjbatch",
                f"env.handedness={hand}",
            ],
        )
        parent = compose(
            "config",
            overrides=[
                "task=g1_cricket_first_step_uniform/mjbatch",
                f"env.handedness={hand}",
            ],
        )
    candidate = OmegaConf.to_container(owner, resolve=True)
    term = candidate["reward"].pop("motion_world_foot_pos")
    assert term["weight"] == 1 and term["params"] == {"std": 0.02}
    assert candidate == OmegaConf.to_container(parent, resolve=True)
    registry.ensure_registries()
    override = evaluation_override(owner, 0.0000625)
    override["mujoco_group_identical_models"] = grouped
    instance = registry.make(
        owner.training.task_name,
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override=override,
    )
    try:
        instance.reset(seed=1)
        yield instance
    finally:
        instance.close()


def test_foot_reward_uses_world_frame_and_worst_foot_without_state_writes(env, monkeypatch):
    reward = env.reward_manager.get_term_cfg("motion_world_foot_pos").func
    command = reward.command
    frames = np.array([250, 400])
    command.time_steps[:] = frames
    target = reward.targets[frames] + env.scene.env_origins[:, None, :]
    positions = command.robot_body_pos_w.copy()
    positions[:, reward.feet] = target
    monkeypatch.setattr(type(command), "robot_body_pos_w", property(lambda _: positions))
    initial = env.get_physics_state_snapshot().copy()
    np.testing.assert_allclose(reward(env, 0.02), 1, atol=1e-6)
    positions[0, reward.feet[0], 0] += 0.04
    positions[1, reward.feet, 1] += 0.02
    np.testing.assert_allclose(reward(env, 0.02), np.exp([-4, -1]), atol=1e-5, rtol=0)
    offsets = np.array([[1, -2, 0], [3, 4, 0]])
    positions[:, reward.feet] = target + offsets[:, None, :]
    translated = SimpleNamespace(scene=SimpleNamespace(env_origins=env.scene.env_origins + offsets))
    np.testing.assert_allclose(reward(translated, 0.02), 1, atol=1e-6)
    np.testing.assert_array_equal(env.get_physics_state_snapshot(), initial)
    for invalid in (0, -0.02, np.inf, np.nan):
        with pytest.raises(ValueError):
            reward(env, invalid)


def test_foot_reward_matches_native_ankles_at_the_applied_reference_frame(env, monkeypatch):
    def forbid_write(*args, **kwargs):
        raise AssertionError("reward and controller must not write live state")

    reward = env.reward_manager.get_term_cfg("motion_world_foot_pos").func
    model = env.get_playback_model()
    data = mujoco.MjData(model)
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    command = reward.command
    for frame in (0, 225, 275, 325, 425, 500):
        with monkeypatch.context() as patch:

            def sample(ids):
                frames = np.full(len(ids), frame, dtype=np.int32)
                command.sampler._set_sampled_frames(ids, frames)
                return frames

            patch.setattr(command.sampler, "sample_frames", sample)
            env.reset(seed=1)
            patch.setattr(Entity, "write_root_state_to_sim", forbid_write)
            patch.setattr(Entity, "write_joint_state_to_sim", forbid_write)
            driver = FirstStepReplay(env, lambda: np.zeros((2, 29), dtype=np.float32))
            driver.initialize(model, data)
            data.ctrl[:] = driver.begin(model, data)
            for _ in range(env.cfg.sim_substeps):
                mujoco.mj_step(model, data)
            driver.finish(model, data)
            expected = np.exp(
                -np.square(data.xpos[feet].astype(np.float32) - reward.targets[frame])
                .sum(axis=-1)
                .max()
                / 0.02**2
            )
            values = dict(env.reward_manager.get_active_iterable_terms(0))
            np.testing.assert_allclose(values["motion_world_foot_pos"][0], expected, atol=1e-5)
