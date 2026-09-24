"""Reconcile complete native impact trials with their actual control-rate reward."""

import json
from importlib.metadata import version

import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_residual import ROOT, CricketReplay, load_policy, sha256
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env

IDENTITY = ("hand", "controller", "offset_m", "seed")


def point_velocity(model, data, geom, point):
    velocity = np.empty(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, velocity, 0)
    return velocity[3:] + np.cross(velocity[:3], point - data.geom_xpos[geom])


def solved_impact(model, data, ball, blade, pre_ball_velocity, step):
    contacts = []
    for index, contact in enumerate(data.contact):
        if set(contact.geom) != {ball, blade}:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        sign = 1 if contact.geom[1] == ball else -1
        normal = sign * contact.frame[:3]
        force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
        bat_velocity = point_velocity(model, data, blade, contact.pos)
        ball_velocity = point_velocity(model, data, ball, contact.pos)
        contacts.append(
            {
                "position_world_m": contact.pos.tolist(),
                "normal_on_ball_world": normal.tolist(),
                "force_on_ball_world_n": force.tolist(),
                "force_norm_n": float(np.linalg.norm(wrench[:3])),
                "normal_force_n": float(wrench[0]),
                "distance_m": float(contact.dist),
                "bat_point_velocity_world_m_s": bat_velocity.tolist(),
                "ball_point_velocity_world_m_s": ball_velocity.tolist(),
                "relative_normal_velocity_m_s": float((ball_velocity - bat_velocity) @ normal),
            }
        )
    address = model.jnt_dofadr[model.joint("ball_free").id]
    return {
        "solve_substep": step,
        "solve_seconds": step * model.opt.timestep,
        "integrated_seconds": (step + 1) * model.opt.timestep,
        "pre_ball_velocity_world_m_s": pre_ball_velocity.tolist(),
        "post_ball_velocity_world_m_s": data.qvel[address : address + 3].tolist(),
        "contacts": contacts,
    }


def reward_readout(env, before):
    manager = env.reward_manager
    cfg = manager.get_term_cfg("batting")
    term = cfg.func
    velocity = float(term.ball.data.root_link_lin_vel_w[0, 0])
    distance = float(np.linalg.norm(term.ball.data.root_link_pos_w - term.bat.read()))
    approach = 5 * np.exp(-((distance / 0.18) ** 2)) * (velocity < 0)
    rates = {name: values[0] for name, values in manager.get_active_iterable_terms(0)}
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    bonus = (rates["batting"] - approach * cfg.weight) * scale
    paid = not before[1] and bool(term.scored[0])
    expected = paid * 5 * (1 + np.tanh(velocity - 1)) / env.step_dt * cfg.weight * scale
    np.testing.assert_allclose(bonus, expected, rtol=1e-5, atol=1e-6)
    return {
        "touching": bool((term.contact.read().reshape(1, 4, 17)[..., 0] > 0).any()),
        "hit_seen_before": before[0],
        "hit_seen_after": bool(term.hit_seen[0]),
        "scored_before": before[1],
        "scored_after": bool(term.scored[0]),
        "separation_reward_event": paid,
        "sampled_ball_vx_m_s": velocity,
        "approach_step_reward": float(approach * cfg.weight * scale),
        "separation_step_reward": float(bonus),
        **{f"rate:{name}": value for name, value in rates.items()},
    }


class RewardReplay(CricketReplay):
    def __init__(self, env):
        super().__init__(env)
        assert self.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    def step(self, env, action, tick):
        initial = env.get_physics_state_snapshot()
        term = env.reward_manager.get_term_cfg("batting").func
        before = bool(term.hit_seen[0]), bool(term.scored[0])
        result = super().step(env, action)
        sample = reward_readout(env, before)
        scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
        reconstructed = sum(v for k, v in sample.items() if k.startswith("rate:")) * scale
        np.testing.assert_allclose(reconstructed, result[0].reward[0], rtol=1e-6, atol=1e-6)
        sample = {
            "control_tick": tick,
            "sensor_solve_substep": (tick + 1) * self.steps - 1,
            **sample,
            "native_step_reward": float(result[0].reward[0]),
            "cached_sum_step_reward": reconstructed,
        }
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = env.action_manager.get_term("residual").processed_action[0]
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
        assert sample["touching"] == bool(impacts[-1]["contacts"])
        return result, impacts, sample


def collect_episode(episodes, sample, dt):
    active = bool(episodes) and episodes[-1]["end"] is None
    if sample["contacts"]:
        if not active:
            episodes.append(
                dict(
                    start=sample,
                    end=None,
                    last_contact=sample,
                    contact_steps=0,
                    loaded_steps=0,
                    impulse_on_ball_world_ns=[0.0, 0.0, 0.0],
                    force_norm_integral_ns=0.0,
                    maximum_penetration_m=0.0,
                    peak_force_norm_n=0.0,
                )
            )
        episode = episodes[-1]
        episode["last_contact"] = sample
        episode["contact_steps"] += 1
        episode["loaded_steps"] += any(c["normal_force_n"] > 0 for c in sample["contacts"])
        for contact in sample["contacts"]:
            episode["impulse_on_ball_world_ns"] = (
                np.array(episode["impulse_on_ball_world_ns"])
                + np.array(contact["force_on_ball_world_n"]) * dt
            ).tolist()
            episode["force_norm_integral_ns"] += contact["force_norm_n"] * dt
            episode["maximum_penetration_m"] = max(
                episode["maximum_penetration_m"], -contact["distance_m"]
            )
            episode["peak_force_norm_n"] = max(
                episode["peak_force_norm_n"], contact["force_norm_n"]
            )
    elif active:
        episodes[-1]["end"] = sample


def audit_trial(env, wrapped, policy, replay, reference):
    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [reference["offset_m"]]
    env.reset(seed=reference["seed"])
    term = env.reward_manager.get_term_cfg("batting").func
    assert not term.hit_seen[0] and not term.scored[0]
    model = replay.model
    joints = model.actuator_trnid[:, 0]
    joint_indices = 1 + model.jnt_qposadr[joints]
    limits = model.jnt_range[joints]
    counts = np.zeros(len(replay.names), dtype=int)
    peaks = np.zeros(len(replay.names))
    fixture_peak = np.zeros(2)
    min_height = min_up = 1.0
    excess = fraction = total = penetration = 0.0
    first = separation = None
    seen = previous = False
    ball_counts, ball_peaks = {}, {}
    failures, events, episodes, samples = set(), [], [], []
    replay.state_error = replay.sensor_error = 0.0
    for tick in range(env.max_episode_length):
        with torch.inference_mode():
            action = (
                policy(wrapped.get_observations()).numpy()
                if reference["controller"] == "ppo"
                else np.zeros((1, 7), dtype=np.float32)
            )
        result, impacts, sample = replay.step(env, action, tick)
        state, trajectory, forces, presence, contact_peaks, fixture, contacts = result
        samples.append(sample)
        total += float(state.reward[0])
        counts += presence.sum(axis=0)
        peaks = np.maximum(peaks, contact_peaks.max(axis=0))
        fixture_peak = np.maximum(
            fixture_peak, np.linalg.norm(fixture.reshape(-1, 2, 3), axis=2).max(axis=0)
        )
        min_height = min(min_height, float(trajectory[:, 3].min()))
        min_up = min(min_up, float((1 - 2 * (trajectory[:, 5:7] ** 2).sum(axis=1)).min()))
        q = trajectory[:, joint_indices]
        excess = max(excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max()))
        fraction = max(fraction, float((np.abs(forces) / model.actuator_forcerange[:, 1]).max()))
        for records, impact in zip(contacts, impacts, strict=True):
            collect_episode(episodes, impact, env.cfg.sim_dt)
            names = {c["geom"] for c in records}
            blade = "bat_blade" in names
            seconds = impact["integrated_seconds"]
            if names and first is None:
                first = {"seconds": seconds, "geoms": sorted(names)}
                if names != {"bat_blade"}:
                    failures.add("first_contact_not_blade_only")
            if blade and not previous:
                events.append({"seconds": seconds, "event": "blade_contact_start"})
            if previous and not blade and separation is None:
                separation = impact["post_ball_velocity_world_m_s"][0]
                events.append(
                    {
                        "seconds": seconds,
                        "event": "first_blade_separation",
                        "ball_vx_m_s": separation,
                    }
                )
            for c in records:
                name = c["geom"]
                if name == "bat_blade":
                    penetration = max(penetration, -c["distance_m"])
                ball_peaks[name] = max(ball_peaks.get(name, 0.0), c["force_norm_n"])
                if name not in {"bat_blade", "pitch"} or (name == "pitch" and not seen):
                    failures.add(f"ball_contact:{name}")
            for name in names:
                ball_counts[name] = ball_counts.get(name, 0) + 1
            seen |= blade
            previous = blade
        if state.terminated[0] or state.truncated[0]:
            break
    for name, bad in (
        ("native_episode_incomplete", not (state.truncated[0] and not state.terminated[0])),
        ("no_blade_contact", not seen),
        ("outgoing_velocity_not_above_1_m_s", separation is None or separation <= 1),
        ("guarded_contact", counts.any()),
        ("pelvis_height", min_height < 0.48),
        ("pelvis_orientation", min_up < 0.65),
        ("joint_limit", excess > 1e-6),
        ("actuator_limit", fraction > 1 + 1e-6),
    ):
        if bad:
            failures.add(name)
    original_pass = not failures
    if penetration > 0.006:
        failures.add("blade_penetration_above_6_mm")
    reproduced = {
        **{k: reference[k] for k in IDENTITY},
        "seconds": (tick + 1) * env.step_dt,
        "passed": not failures,
        "failures": sorted(failures),
        "return": total,
        "first_ball_contact": first,
        "blade_contact_seen": seen,
        "maximum_blade_penetration_m": penetration,
        "first_separation_ball_vx_m_s": separation,
        "blade_events": events,
        "ball_contact_physics_step_counts": ball_counts,
        "ball_contact_peak_force_norm_n": ball_peaks,
        "guard_contact_physics_step_counts": dict(zip(replay.names, counts.tolist(), strict=True)),
        "guard_contact_peak_force_norm_n": dict(zip(replay.names, peaks.tolist(), strict=True)),
        "fixture_peak_force_norm_n": float(fixture_peak[0]),
        "fixture_peak_torque_norm_nm": float(fixture_peak[1]),
        "minimum_pelvis_height_m": min_height,
        "minimum_pelvis_up_z": min_up,
        "maximum_joint_limit_excess_rad": excess,
        "maximum_actuator_limit_fraction": fraction,
        "maximum_endpoint_state_error": replay.state_error,
        "maximum_endpoint_sensor_error": replay.sensor_error,
        "original_shot_gate_passed": original_pass,
    }
    assert reproduced == reference, {
        k: (v, reference.get(k)) for k, v in reproduced.items() if v != reference.get(k)
    }
    for episode in episodes:
        start = episode["start"]["solve_substep"]
        end = episode["end"]["solve_substep"] if episode["end"] else (tick + 1) * replay.steps
        episode["duration_seconds"] = (end - start) * env.cfg.sim_dt
        episode["observed_control_ticks"] = [
            s["control_tick"]
            for s in samples
            if start <= s["sensor_solve_substep"] < end and s["touching"]
        ]
    paid = [s for s in samples if s["separation_reward_event"]]
    loaded = [e for e in episodes if e["loaded_steps"]]
    return {
        **{k: reference[k] for k in IDENTITY},
        "parent_row_exactly_reproduced": True,
        "episodes": episodes,
        "control_sample_fields": list(samples[0]),
        "control_samples": [list(s.values()) for s in samples],
        "loaded_episodes": len(loaded),
        "unobserved_loaded_episodes": sum(not e["observed_control_ticks"] for e in loaded),
        "paid_separation_events": len(paid),
        "first_paid_ball_vx_m_s": paid[0]["sampled_ball_vx_m_s"] if paid else None,
        "physical_first_separation_vx_m_s": separation,
    }


def audit():
    directory = ROOT / "g1_cricket_results/impact_v1"
    parent_path = directory / "trained_evaluation.json"
    parent_hash = sha256(parent_path)
    parent = json.loads(parent_path.read_text())["reports"][1]
    for name, expected in parent["source_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    checkpoint = ROOT / parent["checkpoint"]["path"]
    assert sha256(checkpoint) == parent["checkpoint"]["sha256"]
    for name in ("run_config", "run_summary"):
        assert sha256(directory / "right" / f"{name}.json") == parent[f"{name}_sha256"]
    versions = {name: version(name) for name in parent["versions"]}
    assert versions == parent["versions"]
    sources = dict(parent["source_sha256"])
    for name in (
        "scripts/audit_g1_cricket_impact_reward.py",
        "docs/g1_cricket_impact_reward_audit.md",
        "src/unilab/managers/reward_manager.py",
        "src/unilab/envs/manager_based_rl_env.py",
    ):
        sources[name] = sha256(ROOT / name)
    owner = OmegaConf.create(
        json.loads((directory / "right/run_config.json").read_text())["config"]
    )
    rows, reward_contracts = [], []
    for hand in ("right", "left"):
        override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
        override.update(handedness=hand, auto_reset=False, sim_dt=0.000125)
        env = create_env(owner, num_envs=1, env_cfg_override=override)
        try:
            assert env.step_dt == 0.02 and env.cfg.sim_substeps == 160
            wrapped, policy = load_policy(owner, env, checkpoint)
            replay = RewardReplay(env)
            reward_contracts.append(
                {
                    "hand": hand,
                    "scale_by_dt": env.cfg.scale_rewards_by_dt,
                    "weights": {
                        n: env.reward_manager.get_term_cfg(n).weight
                        for n in env.reward_manager.active_terms
                    },
                }
            )
            for reference in parent["rows"]:
                if reference["hand"] == hand:
                    rows.append(audit_trial(env, wrapped, policy, replay, reference))
            print(json.dumps({"hand": hand, "completed_rows": len(rows)}), flush=True)
        finally:
            env.close()
    assert len(rows) == 96
    assert sha256(parent_path) == parent_hash
    for name, expected in sources.items():
        assert sha256(ROOT / name) == expected, name
    report = {
        "scope": "full_pool_frozen_policy_impact_to_reward_diagnostic",
        "parent_report_sha256": parent_hash,
        "checkpoint": parent["checkpoint"],
        "versions": versions,
        "run_config_sha256": parent["run_config_sha256"],
        "run_summary_sha256": parent["run_summary_sha256"],
        "external_asset_sha256": parent["external_asset_sha256"],
        "source_sha256": sources,
        "reward_contracts": reward_contracts,
        "physics_dt_seconds": 0.000125,
        "control_dt_seconds": 0.02,
        "kinematic_phase": "contact geometry, point velocities and force are pre-integration solved quantities; post_ball_velocity is after that step",
        "rows": rows,
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (directory / "impact_reward_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    audit()
