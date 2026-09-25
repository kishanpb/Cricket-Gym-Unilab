import json
from pathlib import Path

import evaluate_g1_cricket_moving_delivery as evaluation
import numpy as np
import pytest
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from hydra import compose, initialize_config_dir

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.moving_delivery import (
    REFERENCE_START,
    arm_weight,
    moving_command,
)
from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK
from unilab.tasks.manipulation.g1_cricket.running import END_TIME, GATHER_TIME, RELEASE_TIME

ROOT = Path(__file__).resolve().parents[2]


def make_env(hand, velocity_feedforward=False):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=["task=g1_cricket_moving_delivery/mjbatch"])
    owner.env.actions.residual.arm_velocity_feedforward = velocity_feedforward
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override.update(handedness=hand, auto_reset=False)
    return create_env(owner, num_envs=1, env_cfg_override=override)


def test_continuous_arm_envelope():
    times = REFERENCE_START + np.array([-1, 0, GATHER_TIME, RELEASE_TIME, END_TIME, END_TIME + 0.6])
    np.testing.assert_allclose(arm_weight(times), [0, 0, 1, 1, 1, 0], atol=1e-14)
    for boundary in times[1:]:
        assert abs(float(arm_weight(boundary + 1e-6) - arm_weight(boundary - 1e-6))) < 1e-9


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("velocity_feedforward", [False, True])
def test_live_prior_targets_and_physical_release(hand, velocity_feedforward, monkeypatch):
    env = make_env(hand, velocity_feedforward)
    try:
        env.reset(seed=1)

        def forbidden(*args, **kwargs):
            raise AssertionError("motor targets only; no live state writes")

        for entity in (env.scene["robot"], env.scene["ball"]):
            monkeypatch.setattr(entity, "write_root_state_to_sim", forbidden)
            monkeypatch.setattr(entity, "write_joint_state_to_sim", forbidden)
        term = env.action_manager.get_term("residual")
        for time in (0, REFERENCE_START + GATHER_TIME, REFERENCE_START + RELEASE_TIME, 7):
            env.episode_length_buf[:] = round(time / env.step_dt)
            before = env.get_physics_state_snapshot().copy()
            term.process_actions(np.zeros((1, 29), np.float32))
            np.testing.assert_array_equal(env.get_physics_state_snapshot(), before)
            sdk = np.empty_like(term.baseline_action)
            sdk[:, POLICY_TO_SDK] = term.baseline_action
            expected = env.scene["robot"].data.default_joint_pos + 0.25 * sdk
            np.testing.assert_array_equal(term.processed_action[:, :15], expected[:, :15])
            if time == REFERENCE_START + GATHER_TIME:
                local = time - REFERENCE_START
                target = term.arm_reference(local)
                if velocity_feedforward:
                    target += term.arm_reference(local, nu=1) * term.velocity_lead
                np.testing.assert_allclose(
                    term.processed_action[0, 15:],
                    np.clip(target, term.arm_limits[:, 0], term.arm_limits[:, 1]),
                    atol=1e-6,
                )
            assert term.released[0] == (time >= REFERENCE_START + RELEASE_TIME)
            assert env.equality_constraints.get_equality_active()[0, 0] != term.released[0]
            if time == REFERENCE_START + RELEASE_TIME:
                assert moving_command(env)[0, 0] == 1
        monkeypatch.undo()
        env.reset(seed=1)
        assert not term.released[0]
    finally:
        env.close()


@pytest.mark.parametrize("hand", ["right", "left"])
def test_release_interval_matches_native_integration(hand):
    env = make_env(hand)
    try:
        env.reset(seed=1)
        replay, events = DeliveryReplay(env), DeliveryEvents(hand)
        env.episode_length_buf[:] = round((REFERENCE_START + RELEASE_TIME) / env.step_dt)
        initial = env.get_physics_state_snapshot()[0].copy()
        replay.step(env, np.zeros((1, 29), np.float32), events)
        assert events.release_record is not None
        velocity_start = 1 + replay.model.nq + replay.ball_vadr
        np.testing.assert_array_equal(
            events.release_record["velocity"], initial[velocity_start : velocity_start + 3]
        )
        assert env.action_manager.get_term("residual").just_released[0]
        assert not env.equality_constraints.get_equality_active()[0, 0]
    finally:
        env.close()


@pytest.mark.parametrize("velocity_feedforward", [False, True])
def test_evaluator_keeps_complete_bilateral_resolution_pool(
    tmp_path, monkeypatch, velocity_feedforward
):
    calls = []

    def case(owner, hand, dt, output, render):
        assert owner.env.actions.residual.arm_velocity_feedforward == velocity_feedforward
        calls.append((hand, dt))
        return dict(hand=hand, simulation_dt=dt, passed=False, failures=["no_release"])

    monkeypatch.setattr(evaluation, "evaluate_case", case)
    output = tmp_path / "evaluation"
    evaluation.main(output, False, velocity_feedforward)
    summary = json.loads((output / "summary.json").read_text())
    assert calls == [(hand, dt) for hand in ("right", "left") for dt in (0.0000625, 0.00003125)]
    assert len(summary["rows"]) == 4
    assert all(not row["passed"] for row in summary["rows"])
    assert not summary["qualified_showcase"]
    with pytest.raises(FileExistsError):
        evaluation.main(output, False, velocity_feedforward)
