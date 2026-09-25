from pathlib import Path

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_approach import LoadedFootSlip
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.approach_learning import (
    FOOT_MOTION,
    gait_observation,
    lane_tracking,
    loaded_contact_slip,
    loaded_slip_cost,
)
from unilab.tasks.manipulation.g1_cricket.scene import CONTACT_WIDTH, SUPPORT_NAMES, SUPPORT_SLOTS

ROOT = Path(__file__).resolve().parents[2]


def test_contact_point_speed_allows_roll_and_excludes_unloaded_contacts():
    support = np.zeros((1, 1, 2, 3, CONTACT_WIDTH))
    support[..., 0, 0] = 1
    support[..., 0, 1] = 10
    support[..., 0, 11:14] = [0, 0, 1]
    motion = np.zeros((1, 1, 2, 9))
    motion[..., :3] = [0, 0, 0.1]
    motion[..., 3] = 0.2
    speed, force = loaded_contact_slip(support, motion)
    np.testing.assert_array_equal(speed, 0.2)
    np.testing.assert_array_equal(force, 10)
    motion[..., 7] = 2
    np.testing.assert_allclose(loaded_contact_slip(support, motion)[0], 0, atol=1e-12)
    support[..., 0, 1] = 0.5
    motion[..., 3] = 20
    np.testing.assert_array_equal(loaded_contact_slip(support, motion)[0], 0)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_substep_slip_matches_native_contact_point_velocity(hand, monkeypatch):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=["task=g1_cricket_approach_learning/mjbatch", f"env.handedness={hand}"],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    try:
        env.reset(seed=1)
        command = env.command_manager.get_term("motion")

        def sample(ids):
            frames = np.full(len(ids), 183, dtype=np.int32)
            command.sampler._set_sampled_frames(ids, frames)
            return frames

        monkeypatch.setattr(command.sampler, "sample_frames", sample)
        env.reset(seed=1)
        initial = env.get_physics_state_snapshot()[0].copy()
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        env.step(np.zeros((1, 29), np.float32))
        action = env.action_manager.get_term("reference")
        data.ctrl[:] = action.processed_action[0]
        support_ids = np.concatenate(
            [
                np.arange(model.sensor(n).adr[0], model.sensor(n).adr[0] + model.sensor(n).dim[0])
                for n in SUPPORT_NAMES
            ]
        )
        motion_ids = np.concatenate(
            [
                np.arange(model.sensor(n).adr[0], model.sensor(n).adr[0] + model.sensor(n).dim[0])
                for n in FOOT_MOTION
            ]
        )
        native = LoadedFootSlip(model)
        speeds = []
        for _ in range(env.cfg.sim_substeps):
            mujoco.mj_step(model, data)
            speed, _ = loaded_contact_slip(
                data.sensordata[support_ids].reshape(2, SUPPORT_SLOTS, CONTACT_WIDTH),
                data.sensordata[motion_ids].reshape(2, 9),
            )
            np.testing.assert_allclose(speed, native.measure(model, data)[:2], atol=1e-12, rtol=0)
            speeds.append(speed)
        expected = np.square(speeds).mean(axis=0)
        np.testing.assert_allclose(action.slip_squared[0], expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            action.slip_peak[0], np.max(speeds, axis=0), atol=2e-6, rtol=2e-6
        )
        assert loaded_slip_cost(env)[0] == pytest.approx(expected.sum(), abs=2e-6)
        assert gait_observation(env).shape == (1, 5)
        assert 0 < lane_tracking(env)[0] <= 1
        final = np.empty_like(initial, dtype=np.float64)
        mujoco.mj_getState(model, data, final, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(
            final.astype(initial.dtype), env.get_physics_state_snapshot()[0]
        )
        env.reset(seed=1)
        np.testing.assert_array_equal(action.slip_squared, 0)
        np.testing.assert_array_equal(action.slip_peak, 0)
        np.testing.assert_array_equal(action.normal_load, 0)
    finally:
        env.close()
