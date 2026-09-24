import numpy as np
import pytest
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from omegaconf import OmegaConf
from probe_g1_cricket_overarm import owner_config as parent_config
from probe_g1_cricket_overarm_guard import GuardAudit, owner_config
from train_g1_cricket_delivery import make_env

from unilab.tasks.manipulation.g1_cricket.prior import POLICY_TO_SDK


def test_guard_config_changes_only_prior_command_bounds():
    guarded = OmegaConf.to_container(owner_config(), resolve=True)
    assert guarded["env"]["actions"]["residual"].pop("prior_target_limit_fraction") == 0.9
    assert guarded == OmegaConf.to_container(parent_config(), resolve=True)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_guard_preserves_arm_targets_and_hardware(monkeypatch, hand):
    envs = [make_env(cfg(), hand, dt=0.0000625) for cfg in (parent_config, owner_config)]
    try:
        models = [e.get_playback_model() for e in envs]
        for name in (
            "jnt_range",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
            "body_mass",
            "body_inertia",
            "pair_solref",
            "eq_solref",
        ):
            np.testing.assert_array_equal(getattr(models[0], name), getattr(models[1], name))
        terms = [e.action_manager.get_term("residual") for e in envs]
        for scale in (-10.0, 10.0):
            for env, term in zip(envs, terms, strict=True):
                env.reset(seed=6301)
                monkeypatch.setattr(
                    term.session, "run", lambda *args: [np.full((1, 29), scale, np.float32)]
                )
                term.process_actions(np.full((1, 8), 0.2, np.float32))
            old, new = terms
            np.testing.assert_array_equal(
                old.processed_action[:, old.arm_ids], new.processed_action[:, new.arm_ids]
            )
            bounds = new.joint_limits[new.prior_ids]
            lo = bounds.mean(axis=1) - 0.45 * (bounds[:, 1] - bounds[:, 0])
            hi = bounds.mean(axis=1) + 0.45 * (bounds[:, 1] - bounds[:, 0])
            np.testing.assert_array_equal(
                new.processed_action[:, new.prior_ids],
                np.clip(old.processed_action[:, old.prior_ids], lo, hi),
            )
            assert np.any(old.processed_action != new.processed_action)
            assert not new.released.any()
    finally:
        for env in envs:
            env.close()


def test_guard_observer_matches_processed_prior_targets():
    env = make_env(owner_config(), "right", dt=0.0000625)
    try:
        env.reset(seed=6301)
        audit = GuardAudit(env)
        replay = DeliveryReplay(env)
        events = DeliveryEvents("right")
        replay.step(env, np.zeros((1, 8), np.float32), events, observer=lambda m, d: audit(0, m, d))
        assert audit.samples == env.cfg.sim_substeps
        assert len(audit.context) == 1
        term = env.action_manager.get_term("residual")
        sdk = np.empty(29, np.float32)
        sdk[POLICY_TO_SDK] = term.baseline_action[0]
        raw = env.scene["robot"].data.default_joint_pos[0] + 0.25 * sdk
        ids, limits = term.prior_ids, term.prior_target_limits
        assert audit.clip_ticks == int(
            ((raw[ids] < limits[:, 0]) | (raw[ids] > limits[:, 1])).any()
        )
    finally:
        env.close()
