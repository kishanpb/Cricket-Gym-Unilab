"""Native running tracking preserves reset, motor, release and sensor boundaries."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.entity import Entity
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=[("right", False), ("left", False), ("right", True), ("left", True)])
def env(request):
    hand, grouped = request.param
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                "task=g1_cricket_running_tracking/mjbatch",
                f"env.handedness={hand}",
            ],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["mujoco_group_identical_models"] = grouped
    instance = registry.make(
        owner.training.task_name, num_envs=2, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        instance.reset(seed=1)
        yield instance
    finally:
        instance.close()


def test_held_ball_reset_matches_reference_hand_velocity(env):
    model = env.get_playback_model()
    action = env.action_manager.get_term("reference")
    command = env.command_manager.get_term("motion")
    state = env.get_physics_state_snapshot()
    data = mujoco.MjData(model)
    wrist = model.body(f"{env.cfg.handedness}_wrist_yaw_link").id
    ball = model.joint("ball_free")
    q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
    for row in state:
        mujoco.mj_setState(model, data, row, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        rotation = data.xmat[wrist].reshape(3, 3)
        point = data.xpos[wrist] + rotation @ command.offset
        np.testing.assert_allclose(data.qpos[q : q + 3], point, atol=2e-6)
        body_velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, wrist, body_velocity, 0)
        linear = body_velocity[3:] + np.cross(body_velocity[:3], rotation @ command.offset)
        np.testing.assert_allclose(data.qvel[v : v + 3], linear, atol=2e-6)
    np.testing.assert_array_equal(action.released, [False, False])
    assert action.action_dim == 29
    np.testing.assert_array_equal(env.equality_constraints.get_equality_active(), [[True], [True]])


def test_motor_only_step_matches_independent_native_replay(env, monkeypatch):
    model = env.get_playback_model()
    initial = env.get_physics_state_snapshot().copy()
    action = env.action_manager.get_term("reference")
    sensor_view = env.scene.bind_sensor_data(
        tuple(model.sensor(i).name for i in range(model.nsensor))
    )

    def forbid_state_write(*args, **kwargs):
        raise AssertionError("running policy must not overwrite root or joint state")

    monkeypatch.setattr(Entity, "write_root_state_to_sim", forbid_state_write)
    monkeypatch.setattr(Entity, "write_joint_state_to_sim", forbid_state_write)
    env.step(np.zeros((2, 29), dtype=np.float32))
    observed = env.get_physics_state_snapshot()
    sensors = sensor_view.read()
    data = mujoco.MjData(model)
    expected = np.empty(initial.shape[1])
    for index, state in enumerate(initial):
        mujoco.mj_resetData(model, data)
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        data.ctrl[:] = action.target[index]
        for _ in range(env.cfg.sim_substeps):
            mujoco.mj_step(model, data)
        mujoco.mj_getState(model, data, expected, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(expected.astype(observed.dtype), observed[index])
        np.testing.assert_array_equal(data.sensordata.astype(sensors.dtype), sensors[index])
    assert np.isfinite(action.impulse_world).all()
    assert np.all(action.peak_load > 0)
    assert np.all((action.touch_fraction >= 0) & (action.touch_fraction <= 1))
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    assert (action.target >= model.jnt_range[joints, 0]).all()
    assert (action.target <= model.jnt_range[joints, 1]).all()


def test_release_changes_only_holder_and_partial_reset_restores_it(env):
    action = env.action_manager.get_term("reference")
    command = env.command_manager.get_term("motion")
    command.time_steps[:] = [91, 90]
    command._refresh_motion()
    before = env.get_physics_state_snapshot().copy()
    action.process_actions(np.zeros((2, 29), dtype=np.float32))
    np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
    np.testing.assert_array_equal(action.release_physics[0], before[0])
    np.testing.assert_array_equal(action.released, [True, False])
    np.testing.assert_array_equal(env.equality_constraints.get_equality_active(), [[False], [True]])
    command.time_steps[:] = [91, 91]
    action.process_actions(np.zeros((2, 29), dtype=np.float32))
    env.reset(env_ids=np.array([0]))
    np.testing.assert_array_equal(action.released, [False, True])
    np.testing.assert_array_equal(env.equality_constraints.get_equality_active(), [[True], [False]])
    assert not action.release_physics[0].any()


def test_existing_delivery_gate_accepts_running_control_interface(env):
    replay = DeliveryReplay(env, action_name="reference")
    events = DeliveryEvents(env.cfg.handedness)
    replay.step(env, np.zeros((2, 29), dtype=np.float32), events)
    result = events.finish(False)
    assert not result["passed"]
    assert "no_release" in result["failures"]
    assert "episode_incomplete" in result["failures"]
    assert result["peak_holder_force_n"] > 0
