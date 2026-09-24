"""Lane translation preserves the complete motion and fixed cricket geometry."""

from pathlib import Path

import evaluate_g1_cricket_running as evaluation
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("hand", ["right", "left"])
def test_lane_shift_preserves_motion_and_native_reset(tmp_path, monkeypatch, hand):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=["task=g1_cricket_running_tracking/mjbatch", f"env.handedness={hand}"],
        )
    with np.load(ROOT / owner.env.actions.reference.reference_file) as data:
        original = {name: data[name].copy() for name in data.files}
    with np.load(ROOT / owner.env.commands.motion.params.motion_file) as data:
        tracking = {name: data[name].copy() for name in data.files}
    (tmp_path / "src").symlink_to(ROOT / "src", target_is_directory=True)
    (tmp_path / "g1_cricket_results").symlink_to(
        ROOT / "g1_cricket_results", target_is_directory=True
    )
    output = tmp_path / "shifted"
    output.mkdir()
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    ref_path, motion_path = evaluation.shift_lane(owner, output, 0.2)
    sign = 1 if hand == "right" else -1
    with np.load(ref_path) as shifted:
        np.testing.assert_array_equal(shifted["qvel"], original["qvel"])
        np.testing.assert_array_equal(shifted["times"], original["times"])
        difference = np.zeros_like(original["qpos"])
        difference[:, [1, 37]] = sign * 0.2
        np.testing.assert_allclose(shifted["qpos"] - original["qpos"], difference, atol=1e-15)
    with np.load(motion_path) as shifted:
        for name in tracking:
            if name != "body_pos_w":
                np.testing.assert_array_equal(shifted[name], tracking[name])
        difference = np.zeros_like(tracking["body_pos_w"])
        difference[:, 1:, 1] = sign * 0.2
        np.testing.assert_allclose(
            shifted["body_pos_w"] - tracking["body_pos_w"], difference, atol=1e-15
        )
    registry.ensure_registries()
    env = registry.make(
        owner.training.task_name,
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=BackendAdapter(owner, root_dir=tmp_path).build_task_env_cfg_override(),
    )
    try:
        env.reset(seed=1)
        model = env.get_playback_model()
        assert env.scene["robot"].data.root_link_pos_w[0, 1] == pytest.approx(sign * 0.7)
        assert model.geom("bowler_wicket_1").pos[0] == -1.22
        assert model.geom("bowler_wicket_1").pos[1] == 0
        assert env.equality_constraints.get_equality_active()[0, 0]
        assert not env.action_manager.get_term("reference").released[0]
        assert env.cfg.sim_substeps == 320
    finally:
        env.close()


def test_lane_variant_cannot_replace_parent(tmp_path):
    with pytest.raises(ValueError, match="separate output"):
        evaluation.evaluate(tmp_path, lane_offset=0.2)
    for offset in (-0.2, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and outward"):
            evaluation.evaluate(tmp_path, output=tmp_path / "child", lane_offset=offset)
    assert not list(tmp_path.iterdir())
