import pytest

from g1_cricket_cpu.dynamics.bowling_support_clock import SupportSwingClock
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import DeliveryEvents
from g1_cricket_cpu.dynamics.unilab_tasks.bowling import (
    G1CricketBowlingCfg,
    G1CricketBowlingPolicyCfg,
)


PATHS = dict(runtime_bundle="runtime", groot_checkout="groot",
             amp_checkout="amp", scene_bundle="scenes")


def test_checkpoint_profile_is_separate_from_retained_task():
    old = G1CricketBowlingCfg(**PATHS)
    new = G1CricketBowlingPolicyCfg(**PATHS, policy_bundle="policy")
    old.validate()
    new.validate()
    assert old.reward_config == {"version": "retained_bowling_v1"}
    assert new.reward_config == {"version": "checkpoint_bowling_v1"}


@pytest.mark.parametrize("changes", [
    {"policy_bundle": ""}, {"delivery_style": "underarm"},
    {"reward_config": {"version": "retained_bowling_v1"}},
])
def test_checkpoint_profile_rejects_incompatible_configuration(changes):
    config = dict(PATHS, policy_bundle="policy")
    config.update(changes)
    with pytest.raises(ValueError):
        G1CricketBowlingPolicyCfg(**config).validate()


def test_legacy_replay_does_not_disable_learned_release_footwork_veto():
    clock = SupportSwingClock()
    events = DeliveryEvents("right")
    assert not clock.can_release(1.1, events, enforce_footwork=False)
    clock.started_at = 1.0
    assert clock.can_release(1.1, events, enforce_footwork=False)
    assert not clock.can_release(1.1, events)
    assert not clock.can_release(1.1, events, enforce_footwork=True)
