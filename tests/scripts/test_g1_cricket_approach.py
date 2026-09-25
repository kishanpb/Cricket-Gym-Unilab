import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from evaluate_g1_cricket_approach import (
    COLUMNS,
    ApproachEvents,
    LoadedFootSlip,
    make_env,
    resolution_comparison,
    summarize,
)

from unilab.tasks.manipulation.g1_cricket.approach import approach_command, approach_speed
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.tracking import export_reference


def test_profile_boundaries_and_reset_clock():
    times = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 4.5, 5, 8])
    np.testing.assert_array_equal(approach_speed(times), [0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    env = SimpleNamespace(num_envs=2, episode_length_buf=np.array([75, 225]), step_dt=0.02)
    first = approach_command(env)
    np.testing.assert_array_equal(first, [[0.5, 0, 0], [0.5, 0, 0]])
    np.testing.assert_array_equal(approach_command(env), first)
    env.episode_length_buf[0] = 0
    np.testing.assert_array_equal(approach_command(env), [[0, 0, 0], [0.5, 0, 0]])


@pytest.mark.parametrize("hand", ["right", "left"])
def test_native_start_and_exact_velocity_export(hand, tmp_path, monkeypatch):
    env = make_env(hand, 0.0000625)
    try:
        env.reset(seed=5301)
        model = env.get_playback_model()
        initial = env.get_physics_state_snapshot()[0].copy()
        np.testing.assert_allclose(initial[1:4], [-4.5, 0.7 if hand == "right" else -0.7, 0.8])
        np.testing.assert_array_equal(initial[1 + model.nq :], 0)
        assert model.geom("bowler_wicket_0").pos[0] == pytest.approx(-1.22)
        assert env.max_episode_length == 400

        def forbid(*args, **kwargs):
            raise AssertionError("no live state writes after reset")

        for name in ("robot", "ball"):
            for method in ("write_root_state_to_sim", "write_joint_state_to_sim"):
                monkeypatch.setattr(env.scene[name], method, forbid)
        env.step(np.zeros((1, 8), np.float32))
        term = env.action_manager.get_term("residual")
        assert not term.released.any()
        assert env.equality_constraints.get_equality_active().all()
        native = env.get_physics_state_snapshot()[0].copy()
        qpos = np.array([initial[1 : 1 + model.nq], native[1 : 1 + model.nq]])
        qvel = np.array([initial[1 + model.nq :], native[1 + model.nq :]])
        before = qvel.copy()
        destination = tmp_path / "tracking.npz"
        export_reference(model, qpos, 50, destination, qvel=qvel)
        np.testing.assert_array_equal(qvel, before)
        np.testing.assert_array_equal(env.get_physics_state_snapshot()[0], native)
        with np.load(destination) as motion:
            joints = [model.joint(name).id for name in SDK_JOINTS]
            np.testing.assert_array_equal(motion["joint_vel"], qvel[:, model.jnt_dofadr[joints]])
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = qpos[-1], qvel[-1]
            mujoco.mj_forward(model, data)
            velocity = np.empty(6)
            for body in range(1, model.nbody):
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
                np.testing.assert_array_equal(motion["body_ang_vel_w"][-1, body], velocity[:3])
                np.testing.assert_array_equal(motion["body_lin_vel_w"][-1, body], velocity[3:])
    finally:
        env.close()


def passing_fixture():
    events = ApproachEvents("right")
    events.height, events.up = 0.79, 0.999
    events.minimum_foot_y, events.maximum_foot_y, events.minimum_side_clearance = 0.4, 0.9, 0.4
    events.landings = {side: [{"time": 2}, {"time": 3}] for side in ("left", "right")}
    steps = np.zeros((800, len(COLUMNS)))
    steps[:, 0] = np.arange(1, 801) * 0.01
    steps[:, 2:4] = [0.7, 0.8]
    steps[(steps[:, 0] >= 2) & (steps[:, 0] < 4), 4] = 1
    steps[:, 9:11] = 150
    states = np.array([[0, -4.5, 0.7, 0.8], [8, -1.5, 0.7, 0.8]])
    return events, steps, states


def test_complete_gates_are_json_serializable():
    events, steps, states = passing_fixture()
    result = summarize(events, steps, states, 300, True)
    assert result["passed"]
    result.update(hand="right", seed=5301)
    assert resolution_comparison(result, result)["passed"]
    json.dumps(result, allow_nan=False)
    json.dumps(resolution_comparison(result, result), allow_nan=False)


@pytest.mark.parametrize(
    "failure",
    [
        "incomplete",
        "unsettled",
        "no_steps",
        "side",
        "joint",
        "holder",
        "slip_speed",
        "slip_distance",
    ],
)
def test_failures_cannot_qualify_teacher(failure):
    events, steps, states = passing_fixture()
    if failure == "unsettled":
        steps[-30:, 4] = 0.2
    elif failure == "no_steps":
        events.landings["left"] = []
    elif failure == "side":
        events.minimum_side_clearance = -0.01
    elif failure == "joint":
        events.limit_excess = 0.001
    elif failure == "holder":
        steps[-1, 11] = 0.002
    elif failure == "slip_speed":
        steps[-1, 13] = 0.21
    elif failure == "slip_distance":
        steps[-1, 16] = 0.031
    result = summarize(events, steps, states, 300, failure != "incomplete")
    assert not result["passed"]
    assert result["failures"]


def test_each_airborne_interval_is_independent():
    events = ApproachEvents("right")
    air = np.array([[0, 0.5, 0.01], [0.2, 0.7, 0.02]])
    ground = air.copy()
    ground[:, 2] -= 0.01
    for time in (0.3, 0.33, 0.36):
        events.observe_support(time, "left", False, air)
    events.observe_support(0.4, "left", True, ground)
    events.observe_support(0.5, "left", False, air)
    events.observe_support(0.51, "left", True, ground)
    assert len(events.landings["left"]) == 1
    events.observe_support(0.6, "left", False, air)
    events.observe_support(0.63, "left", False, air)
    events.observe_support(0.7, "left", True, ground)
    assert len(events.landings["left"]) == 2


def test_loaded_contact_slip_distinguishes_rolling_from_translation():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001"/><worldbody>
      <geom name="pitch" type="plane" size="0 0 .1"/>
      <body name="left_ankle_roll_link" pos="0 0 .099"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body name="right_ankle_roll_link" pos="1 0 1"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[0] = 0.1
    mujoco.mj_forward(model, data)
    slip = LoadedFootSlip(model)
    np.testing.assert_allclose(slip.measure(model, data), [0.1, 0, 0.0001, 0], atol=1e-12)
    np.testing.assert_allclose(slip.measure(model, data), [0.1, 0, 0.0002, 0], atol=1e-12)
    data.qvel[0], data.qvel[4] = data.xpos[1, 2] - data.contact[0].pos[2], 1
    mujoco.mj_forward(model, data)
    assert slip.measure(model, data)[0] < 1e-12
    data.qpos[2] = 1
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(slip.measure(model, data), 0)
