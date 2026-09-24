from pathlib import Path

import mujoco
import numpy as np
import pytest
from g1_cricket_delivery_trial import (
    DeliveryEvents,
    DeliveryReplay,
    arm_geometry,
    capsule_bounds,
    feet_failures,
)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.delivery import BOWLING_X, TARGET_X

ROOT = Path(__file__).resolve().parents[2]


def make_env(hand="right", engine="mjbatch", dt=0.00025):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        cfg = compose("config", overrides=[f"task=g1_cricket_delivery_v1/{engine}"])
    override = BackendAdapter(cfg, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False, sim_dt=dt)
    env = create_env(cfg, num_envs=1, env_cfg_override=override)
    env.reset(seed=6301)
    return env


def test_delivery_owner_only_executor_changes():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        a, b = [
            OmegaConf.to_container(
                compose("config", overrides=[f"task=g1_cricket_delivery_v1/{e}"]), resolve=True
            )
            for e in ("mujoco", "mjbatch")
        ]
    assert b["env"].pop("mujoco_substep_engine") == "mjbatch"
    assert a == b


def test_capsule_bounds_include_rotated_toe_and_heel_not_ankle():
    bounds = capsule_bounds(
        np.array([[1.0, 2, 3]]),
        np.array([[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]]),
        np.array([[0.01, 0.1, 0]]),
        np.array([mujoco.mjtGeom.mjGEOM_CAPSULE]),
    )
    np.testing.assert_allclose(bounds, [[0.89, 1.99, 2.99], [1.11, 2.01, 3.01]])


def good_events():
    events = DeliveryEvents("right")
    bounds = np.array([[-0.1, 0.3, 0], [0.1, 0.5, 0.08]])
    for side in ("right", "left"):
        events.observe_support(0.3, side, False, bounds + [0, 0, 0.01])
        events.observe_support(0.33, side, False, bounds + [0, 0, 0.01])
    events.observe_support(0.5, "right", True, bounds)
    events.observe_support(0.7, "left", True, bounds)
    events.observe_arm(0.5, -0.1, 2.9)
    events.observe_arm(0.6, 0.1, 2.9)
    events.height, events.up = 0.7, 0.9
    return events


def test_planted_force_chatter_is_not_a_landing_and_latest_back_landing_counts():
    events = good_events()
    bounds = np.array(events.landings["right"][0]["bounds"])
    events.observe_support(0.8, "right", False, bounds)
    events.observe_support(0.9, "right", True, bounds)
    assert len(events.landings["right"]) == 1
    events.observe_support(0.91, "right", False, bounds + [0, 0, 0.01])
    events.observe_support(0.94, "right", False, bounds + [0, 0, 0.01])
    events.observe_support(0.96, "right", True, bounds)
    release(events)
    assert events.release_record["back_landing"]["time"] == 0.96
    assert "invalid_delivery_stride_order" in events.failures


def release(events, *, height=1.4, upper=0.6, angle=3.0):
    events.release(1.0, np.array([-0.2, 0.3, height]), np.array([8.0, 0, 0]), 1.1, upper, angle)


def flight(events):
    events.observe_ball(1.0, np.array([0.0, 0.3, 1.4]), [])
    events.observe_ball(2.0, np.array([10.0, 0.3, 0.036]), [("pitch", -0.001, 20.0)])
    events.observe_ball(2.1, np.array([12.0, 0.3, 0.3]), [])
    events.observe_ball(3.0, np.array([18.0, 0.3, 0.5]), [])


def test_synthetic_positive_and_physical_negative_gates():
    events = good_events()
    release(events)
    flight(events)
    assert events.finish(True)["passed"]
    events.failures.add("ball_contact:torso_collision")
    assert not events.finish(True)["passed"]
    events = good_events()
    release(events, height=1.0, upper=-0.1)
    flight(events)
    assert "release_not_overarm" in events.finish(True)["failures"]
    events = good_events()
    events.observe_arm(0.8, 0.5, 3.2)
    release(events, angle=2.9)
    flight(events)
    assert "elbow_extension_above_15_degrees" in events.finish(True)["failures"]
    assert "no_release" in good_events().finish(True)["failures"]
    events = good_events()
    release(events)
    flight(events)
    events.height = 0.3
    assert "pelvis_height" in events.finish(True)["failures"]


def test_foot_fault_even_with_ball_behind_line_and_missing_landing():
    events = good_events()
    events.landings["left"][0]["bounds"][0][0] = 0.01
    release(events)
    assert "front_foot_over_popping_crease" in events.failures
    assert feet_failures(None, None, 1, 1) == {"missing_delivery_stride"}
    events = good_events()
    events.landings["right"][0]["bounds"][1][1] = 1.32
    release(events)
    assert "back_foot_outside_return_crease" in events.failures


def test_arm_angle_detects_straightening_and_degenerate_geometry():
    upper, angle = arm_geometry(np.zeros(3), np.array([0, 0, 1.0]), np.array([0, 0, 2.0]))
    assert upper == 1 and angle == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="degenerate"):
        arm_geometry(np.zeros(3), np.zeros(3), np.ones(3))


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_delivery_scene_and_independent_release_replay(hand, engine):
    env = make_env(hand, engine)
    try:
        m = env.get_playback_model()
        visual = mujoco.MjModel.from_xml_path(env.get_scene_visual_model_file())
        assert m.geom("wicket_1").pos[0] - m.geom("bowler_wicket_1").pos[0] == pytest.approx(20.12)
        assert visual.geom("bowler_popping_crease").pos[0] - visual.geom(
            "bowler_popping_crease"
        ).size[0] == pytest.approx(0)
        assert visual.geom("bowler_return_1").pos[1] - visual.geom("bowler_return_1").size[
            1
        ] == pytest.approx(1.32)
        assert m.geom("bowler_wicket_1").pos[0] == BOWLING_X
        assert m.geom("wicket_1").pos[0] == pytest.approx(TARGET_X)
        assert env.scene["robot"].data.root_link_pos_w[0, 1] == pytest.approx(
            0.5 if hand == "right" else -0.5
        )
        replay = DeliveryReplay(env)
        events = DeliveryEvents(hand)
        for tick in range(8):
            action = np.zeros((1, 8), np.float32)
            if tick == 4:
                action[0, 7] = 1
            replay.step(env, action, events)
        assert events.release_record is not None
        assert "missing_delivery_stride" in events.failures
        assert not events.finish(False)["passed"]
    finally:
        env.close()
