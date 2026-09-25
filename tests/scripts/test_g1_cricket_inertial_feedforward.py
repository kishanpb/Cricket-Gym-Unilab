import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.tasks.manipulation.g1_cricket.inertial_feedforward import (
    bounded_reference_forces,
    motion_increment,
)


def slider():
    return mujoco.MjModel.from_xml_string("""
    <mujoco><option gravity="0 0 0"/><worldbody><body>
      <joint name="slide" type="slide" axis="1 0 0" damping="3"/>
      <geom size=".1" mass="2"/>
    </body></worldbody><actuator>
      <position joint="slide" kp="10" kv="2" forcerange="-5 5"/>
    </actuator></mujoco>""")


@pytest.mark.parametrize(
    "velocity,acceleration,expected", [(0, 0, 0), (2, 0, 6), (0, 4, 8), (2, 4, 14)]
)
def test_motion_increment_inertia_passive_sign_and_zero_motion(velocity, acceleration, expected):
    model = slider()
    data = mujoco.MjData(model)
    np.testing.assert_allclose(
        motion_increment(model, data, model.qpos0, [velocity], [acceleration]), [expected]
    )
    np.testing.assert_array_equal(data.qfrc_applied, 0)
    np.testing.assert_array_equal(data.xfrc_applied, 0)


def test_total_reference_force_bounds_and_exactly_once_servo_velocity():
    model = slider()
    times = np.arange(11) * 0.02
    poses = (0.1 + 2 * times)[:, None]
    gravity = np.full((11, 1), 4.0)
    limits = np.array([[-0.1, 0.3]])
    forces, rows = bounded_reference_forces(model, poses, times, gravity, limits)
    for pose, force, row in zip(poses, forces, rows, strict=True):
        bounds = np.array(row["reachable_motor_force_nm"])
        assert (force >= bounds[:, 0]).all() and (force <= bounds[:, 1]).all()
        assert (np.abs(force) <= 5).all()
        control = pose + 2 / 10 * 2 + force / 10
        control = np.clip(control, *limits.T)
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.ctrl[:] = pose, 2, control
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, force, atol=1e-12)
    assert any(max(row["clipping_nm"]) > 0 for row in rows)


def test_floating_root_coupling_retained_and_independent_ball_omitted():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody><body><freejoint/><geom size=".1" mass="3"/>
      <body pos="0 0 .3"><joint name="slide" type="slide" axis="1 0 0"/>
        <geom size=".1" mass="2"/></body></body>
      <body pos="5 0 5"><freejoint name="ball_free"/><geom size=".03" mass=".16"/></body>
    </worldbody><actuator><position joint="slide" kp="10" forcerange="-20 20"/></actuator></mujoco>""")
    data = mujoco.MjData(model)
    velocity, acceleration = np.zeros(model.nv), np.zeros(model.nv)
    acceleration[0] = 3
    result = motion_increment(model, data, model.qpos0, velocity, acceleration)
    assert result[6] == pytest.approx(6)
    acceleration[-6:-3] = [100, 200, -300]
    other = motion_increment(model, data, model.qpos0, velocity, acceleration)
    np.testing.assert_allclose(other[:7], result[:7])
    np.testing.assert_array_equal(data.qfrc_applied, 0)
    times = np.arange(11) * 0.02
    poses = np.tile(model.qpos0, (11, 1))
    poses[:, 0] += 1.5 * times**2
    gravity, limits = np.zeros((11, 1)), np.array([[-2, 2]])
    forces, _ = bounded_reference_forces(model, poses, times, gravity, limits)
    moving_ball = poses.copy()
    moving_ball[:, -7:-4] += times[:, None] ** 2 * [100, 200, -300]
    alternate, _ = bounded_reference_forces(model, moving_ball, times, gravity, limits)
    np.testing.assert_allclose(alternate, forces, atol=1e-12)


def test_stationary_reference_keeps_existing_gravity_force():
    model = slider()
    poses = np.full((11, 1), 0.1)
    gravity = np.full((11, 1), 2.0)
    forces, rows = bounded_reference_forces(
        model, poses, np.arange(11) * 0.02, gravity, np.array([[-2, 2]])
    )
    np.testing.assert_array_equal(forces, gravity)
    assert not any(any(row["clipping_nm"]) for row in rows)


@pytest.mark.parametrize("hand", ["right", "left"])
def test_g1_compensation_is_motor_only_and_preserves_physics(hand):
    root = Path(__file__).resolve().parents[2]
    saved = json.loads(
        (
            root / f"g1_cricket_results/bimanual_batting_learning_v1/ppo_{hand}/run_config.json"
        ).read_text()
    )
    owner = OmegaConf.create(saved["config"])
    owner.env.actions.reference.inertial_compensation = True
    owner.env.actions.reference.reference_file = (
        f"g1_cricket_results/bimanual_projected_v1/{hand}_reference.npz"
    )
    owner.env.commands.motion.params.motion_file = (
        f"g1_cricket_results/bimanual_projected_v1/{hand}_tracking.npz"
    )
    registry.ensure_registries()
    override = BackendAdapter(owner, root_dir=root).build_task_env_cfg_override()
    env = registry.make(
        "G1CricketBimanualLearning", num_envs=1, sim_backend="mujoco", env_cfg_override=override
    )
    try:
        env.reset(seed=1)
        action = env.action_manager.get_term("reference")
        assert len(action.inertial_audit) == 151
        model = env.get_playback_model()
        for row, offset in zip(action.inertial_audit, action.gravity_offset, strict=True):
            force = np.array(row["bounded_motor_force_nm"])
            np.testing.assert_allclose(offset * model.actuator_gainprm[:, 0], force)
            bounds = np.array(row["reachable_motor_force_nm"])
            assert (force >= bounds[:, 0]).all() and (force <= bounds[:, 1]).all()
            assert len(row["unapplied_root_increment"]) == 6
        before = env.get_physics_state_snapshot().copy()
        action.process_actions(np.zeros((1, 29)))
        np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
        assert (action.target >= action.control_limits[:, 0]).all()
        assert (action.target <= action.control_limits[:, 1]).all()
    finally:
        env.close()
