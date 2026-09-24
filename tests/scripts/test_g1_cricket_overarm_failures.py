import mujoco
import numpy as np
from audit_g1_cricket_overarm_failures import FailureAudit, phase
from g1_cricket_delivery_trial import DeliveryEvents, DeliveryReplay
from probe_g1_cricket_overarm import owner_config
from train_g1_cricket_delivery import make_env


def test_joint_attribution_retains_first_and_worst_without_state_writes():
    env = make_env(owner_config(), "right", dt=0.0000625)
    try:
        audit = FailureAudit(env)
        model = env.get_playback_model()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        joint = model.joint("left_knee_joint")
        actuator = model.actuator("left_knee_joint").id
        for tick, excess in ((80, 0.01), (81, 0.02), (82, 0.015)):
            data.qpos[joint.qposadr[0]] = joint.range[1] + excess
            data.qvel[joint.dofadr[0]] = 0.3
            data.ctrl[actuator] = joint.range[1] + 0.1
            data.time = tick * 0.02
            mujoco.mj_forward(model, data)
            before = np.r_[data.qpos, data.qvel, data.ctrl, data.sensordata].copy()
            audit(tick, model, data)
            np.testing.assert_array_equal(
                np.r_[data.qpos, data.qvel, data.ctrl, data.sensordata], before
            )
        result = audit.result()
        assert result["first_joint"]["tick"] == 80
        assert result["worst_joint"]["tick"] == 81
        assert result["first_joint"]["joint"] == "left_knee_joint"
        assert not result["first_joint"]["selected_arm"]
        assert result["first_joint"]["phase"] == "hold"
        assert result["first_joint"]["qvel"] == 0.3
        assert len(result["context"]) == 3
        assert result["substeps"] == 3
        assert [phase(t) for t in (0, 10, 30, 80, 120, 170)] == [
            "neutral",
            "settle",
            "raise",
            "hold",
            "recover",
            "final_settle",
        ]
    finally:
        env.close()


def test_replay_observer_preserves_exact_endpoint_and_counts_every_substep():
    env = make_env(owner_config(), "left", dt=0.0000625)
    try:
        states = []
        for observe in (False, True):
            env.reset(seed=6301)
            replay = DeliveryReplay(env)
            events = DeliveryEvents("left")
            audit = FailureAudit(env)
            replay.step(
                env,
                np.zeros((1, 8), np.float32),
                events,
                observer=(lambda m, d: audit(0, m, d)) if observe else None,
            )
            states.append(env.get_physics_state_snapshot().copy())
            assert audit.samples == (env.cfg.sim_substeps if observe else 0)
        np.testing.assert_array_equal(*states)
    finally:
        env.close()
