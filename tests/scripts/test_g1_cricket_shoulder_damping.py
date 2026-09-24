import json

import mujoco
import numpy as np
import pytest
from g1_cricket_delivery_trial import DeliveryEvents
from g1_cricket_signed_delivery import SignedDeliveryReplay
from probe_g1_cricket_overarm_guard import owner_config as parent_config
from probe_g1_cricket_shoulder_damping import owner_config
from train_g1_cricket_delivery import make_env

from unilab.utils.sim2sim import (
    CrossBackendIncompatibleError,
    extract_contract_snapshot,
    resolve_sim2sim_config,
)


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize("engine", ["mujoco", "mjbatch"])
def test_only_selected_actuator_damping_changes_and_replays(hand, engine):
    original = make_env(parent_config(engine), hand, dt=0.0000625, engine=engine)
    modified = make_env(owner_config(engine), hand, dt=0.0000625, engine=engine)
    try:
        old, new = original.get_playback_model(), modified.get_playback_model()
        selected = new.actuator(f"{hand}_shoulder_pitch_joint").id
        expected = old.actuator_biasprm.copy()
        expected[selected, 2] = -2
        np.testing.assert_array_equal(new.actuator_biasprm, expected)
        np.testing.assert_array_equal(old.actuator_biasprm[:, 2][15:], np.full(14, -10))
        for name in dir(old):
            value = getattr(old, name)
            if isinstance(value, np.ndarray) and name != "actuator_biasprm":
                np.testing.assert_array_equal(value, getattr(new, name), err_msg=name)
        d = mujoco.MjData(new)
        mujoco.mj_resetDataKeyframe(new, d, 0)
        joint = new.actuator_trnid[selected, 0]
        d.ctrl[:] = d.qpos[new.jnt_qposadr[new.actuator_trnid[:, 0]]]
        d.ctrl[selected] += 0.1
        d.qvel[new.jnt_dofadr[joint]] = 1
        mujoco.mj_forward(new, d)
        assert d.actuator_force[selected] == pytest.approx(2)
        modified.reset(seed=6301)
        replay, events = SignedDeliveryReplay(modified), DeliveryEvents(hand)
        for tick in range(4):
            action = np.zeros((1, 8), np.float32)
            action[0, 0] = 0.4
            action[0, 7] = tick >= 2
            replay.step(modified, action, events)
        assert events.force_fraction <= 1
    finally:
        original.close()
        modified.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_strict_checkpoint_contract_rejects_original_controller(tmp_path, reverse):
    old, new = parent_config(), owner_config()
    source, target = (new, old) if reverse else (old, new)
    (tmp_path / "run_config.json").write_text(
        json.dumps(dict(contract_snapshot=extract_contract_snapshot(source)))
    )
    with pytest.raises(CrossBackendIncompatibleError, match="env.actions"):
        resolve_sim2sim_config(tmp_path, target)


def test_same_controller_cross_executor_contract_is_valid(tmp_path):
    (tmp_path / "run_config.json").write_text(
        json.dumps(dict(contract_snapshot=extract_contract_snapshot(owner_config("mujoco"))))
    )
    target = owner_config("mjbatch")
    assert resolve_sim2sim_config(tmp_path, target) is target
