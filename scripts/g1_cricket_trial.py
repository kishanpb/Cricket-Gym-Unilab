"""Single-trial diagnostics using the retained full-shot gate and native replay."""

import mujoco
import numpy as np
from audit_g1_cricket_impact_reward import collect_episode, solved_impact
from evaluate_g1_cricket_residual import CricketReplay

from unilab.tasks.manipulation.g1_cricket.residual import RESIDUAL_LIMITS


class ImpactReplay(CricketReplay):
    def step(self, env, action, tick):
        initial = env.get_physics_state_snapshot()
        result = super().step(env, action)
        term = env.action_manager.get_term("residual")
        np.testing.assert_array_equal(term.raw_action, action)
        np.testing.assert_array_equal(term.residual_radians, np.tanh(action) * RESIDUAL_LIMITS)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = term.processed_action[0]
        address = self.model.jnt_dofadr[self.model.joint("ball_free").id]
        blade = self.model.geom("bat_blade").id
        impacts = []
        for index in range(self.steps):
            pre = self.data.qvel[address : address + 3].copy()
            mujoco.mj_step(self.model, self.data)
            impacts.append(
                solved_impact(
                    self.model, self.data, self.ball_geom, blade, pre, tick * self.steps + index
                )
            )
        endpoint = np.empty(initial.shape[1])
        mujoco.mj_getState(self.model, self.data, endpoint, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(endpoint, result[1][-1])
        if self.data.warning.number.any():
            raise RuntimeError("MuJoCo warning during impact replay")
        return result, impacts


class BallEvents:
    """All-geometry occupancy gates, including contacts after the first exit."""

    def __init__(self):
        self.first = self.separation = None
        self.seen = self.previous = False
        self.penetration = 0.0
        self.counts, self.peaks = {}, {}
        self.failures, self.events = set(), []

    def observe(self, contacts, seconds, velocity):
        names = {c["geom"] for c in contacts}
        blade = "bat_blade" in names
        if names and self.first is None:
            self.first = {"seconds": seconds, "geoms": sorted(names)}
            if names != {"bat_blade"}:
                self.failures.add("first_contact_not_blade_only")
        if blade and not self.previous:
            self.events.append({"seconds": seconds, "event": "blade_contact_start"})
        if self.previous and not blade and self.separation is None:
            self.separation = float(velocity)
            self.events.append(
                {
                    "seconds": seconds,
                    "event": "first_blade_separation",
                    "ball_vx_m_s": self.separation,
                }
            )
        for contact in contacts:
            name = contact["geom"]
            if name == "bat_blade":
                self.penetration = max(self.penetration, -contact["distance_m"])
            self.peaks[name] = max(self.peaks.get(name, 0.0), contact["force_norm_n"])
            if name not in {"bat_blade", "pitch"} or (name == "pitch" and not self.seen):
                self.failures.add(f"ball_contact:{name}")
        for name in names:
            self.counts[name] = self.counts.get(name, 0) + 1
        self.seen |= blade
        self.previous = blade


def finish_gate(ball, *, complete, guarded, height, up, excess, fraction):
    failures = set(ball.failures)
    for name, bad in (
        ("native_episode_incomplete", not complete),
        ("no_blade_contact", not ball.seen),
        ("outgoing_velocity_not_above_1_m_s", ball.separation is None or ball.separation <= 1),
        ("guarded_contact", guarded),
        ("pelvis_height", height < 0.48),
        ("pelvis_orientation", up < 0.65),
        ("joint_limit", excess > 1e-6),
        ("actuator_limit", fraction > 1 + 1e-6),
    ):
        if bad:
            failures.add(name)
    original_pass = not failures
    failures = sorted(failures)
    if ball.penetration > 0.006:
        failures.append("blade_penetration_above_6_mm")
    return failures, original_pass


def trial(env, replay, identity, action_at, ticks):
    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [identity["offset_m"]]
    env.reset(seed=identity["seed"])
    assert 0 < ticks <= env.max_episode_length == 100
    model = replay.model
    joints = model.actuator_trnid[:, 0]
    joint_indices = 1 + model.jnt_qposadr[joints]
    limits = model.jnt_range[joints]
    velocity_index = 1 + model.nq + model.jnt_dofadr[model.joint("ball_free").id]
    counts = np.zeros(len(replay.names), dtype=int)
    peaks = np.zeros(len(replay.names))
    fixture_peak = np.zeros(2)
    height = up = 1.0
    excess = fraction = total = 0.0
    ball, episodes, first_loaded = BallEvents(), [], None
    replay.state_error = replay.sensor_error = 0.0
    for tick in range(ticks):
        result, impacts = replay.step(env, action_at(tick), tick)
        state, trajectory, forces, presence, contact_peaks, fixture, contacts = result
        total += float(state.reward[0])
        counts += presence.sum(axis=0)
        peaks = np.maximum(peaks, contact_peaks.max(axis=0))
        fixture_peak = np.maximum(
            fixture_peak, np.linalg.norm(fixture.reshape(-1, 2, 3), axis=2).max(axis=0)
        )
        height = min(height, float(trajectory[:, 3].min()))
        up = min(up, float((1 - 2 * (trajectory[:, 5:7] ** 2).sum(axis=1)).min()))
        q = trajectory[:, joint_indices]
        excess = max(excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max()))
        fraction = max(fraction, float((np.abs(forces) / model.actuator_forcerange[:, 1]).max()))
        for substep, (records, impact) in enumerate(zip(contacts, impacts, strict=True)):
            collect_episode(episodes, impact, env.cfg.sim_dt)
            if first_loaded is None and any(c["normal_force_n"] > 0 for c in impact["contacts"]):
                first_loaded = impact
            ball.observe(
                records,
                (tick * replay.steps + substep + 1) * env.cfg.sim_dt,
                trajectory[substep, velocity_index],
            )
        if state.terminated[0] or state.truncated[0]:
            break
    terminated, truncated = bool(state.terminated[0]), bool(state.truncated[0])
    failures, original_pass = finish_gate(
        ball,
        complete=truncated and not terminated,
        guarded=counts.any(),
        height=height,
        up=up,
        excess=excess,
        fraction=fraction,
    )
    outcome = {
        **identity,
        "seconds": (tick + 1) * env.step_dt,
        "passed": not failures,
        "failures": failures,
        "return": total,
        "first_ball_contact": ball.first,
        "blade_contact_seen": ball.seen,
        "maximum_blade_penetration_m": ball.penetration,
        "first_separation_ball_vx_m_s": ball.separation,
        "blade_events": ball.events,
        "ball_contact_physics_step_counts": ball.counts,
        "ball_contact_peak_force_norm_n": ball.peaks,
        "guard_contact_physics_step_counts": dict(zip(replay.names, counts.tolist(), strict=True)),
        "guard_contact_peak_force_norm_n": dict(zip(replay.names, peaks.tolist(), strict=True)),
        "fixture_peak_force_norm_n": float(fixture_peak[0]),
        "fixture_peak_torque_norm_nm": float(fixture_peak[1]),
        "minimum_pelvis_height_m": height,
        "minimum_pelvis_up_z": up,
        "maximum_joint_limit_excess_rad": excess,
        "maximum_actuator_limit_fraction": fraction,
        "maximum_endpoint_state_error": replay.state_error,
        "maximum_endpoint_sensor_error": replay.sensor_error,
        "original_shot_gate_passed": original_pass,
    }
    return {
        "outcome": outcome,
        "terminated": terminated,
        "truncated": truncated,
        "blade_episode_count": len(episodes),
        "first_impact": episodes[0] if episodes else None,
        "first_loaded_impact": first_loaded,
    }
