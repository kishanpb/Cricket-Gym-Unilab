"""First-impact motion and pre-contact control headroom, with full pool accounting."""

import json
from importlib.metadata import version

import mujoco
import numpy as np
import torch
from audit_g1_cricket_impact_reward import collect_episode, solved_impact
from evaluate_g1_cricket_impact_events_learning import validate_pool
from evaluate_g1_cricket_impact_resolution import IDENTITY
from evaluate_g1_cricket_residual import ROOT, CricketReplay, load_policy, sha256
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES, SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.residual import RESIDUAL_LIMITS

DIRECTORY = ROOT / "g1_cricket_results/impact_events_v1"


def window_slice(tick, steps, contact_step, window_steps):
    begin = max(tick * steps, contact_step - window_steps, 0)
    end = min((tick + 1) * steps, contact_step)
    return slice(max(0, begin - tick * steps), max(0, end - tick * steps))


def summarize_window(raw, errors, velocities, force_fractions):
    return {
        "physics_samples": len(raw),
        "raw_action_clipped_fraction": (np.abs(raw) > 1).mean(axis=0).tolist(),
        "residual_at_bound_fraction": (np.abs(raw) >= 1).mean(axis=0).tolist(),
        "target_error_rms_rad": np.sqrt((errors**2).mean(axis=0)).tolist(),
        "target_error_peak_abs_rad": np.abs(errors).max(axis=0).tolist(),
        "joint_velocity_peak_abs_rad_s": np.abs(velocities).max(axis=0).tolist(),
        "actuator_force_peak_fraction": force_fractions.max(axis=0).tolist(),
        "actuator_at_limit_fraction": (force_fractions >= 1 - 1e-6).mean(axis=0).tolist(),
    }


class MotionReplay(CricketReplay):
    def step(self, env, action, tick):
        initial = env.get_physics_state_snapshot()
        result = super().step(env, action)
        term = env.action_manager.get_term("residual")
        np.testing.assert_array_equal(term.raw_action, action)
        np.testing.assert_array_equal(
            term.residual_radians, np.clip(action, -1, 1) * RESIDUAL_LIMITS
        )
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = term.processed_action[0]
        address = self.model.jnt_dofadr[self.model.joint("ball_free").id]
        blade = self.model.geom("bat_blade").id
        impacts = []
        for substep in range(self.steps):
            pre = self.data.qvel[address : address + 3].copy()
            mujoco.mj_step(self.model, self.data)
            impacts.append(
                solved_impact(
                    self.model, self.data, self.ball_geom, blade, pre, tick * self.steps + substep
                )
            )
        endpoint = np.empty(initial.shape[1])
        mujoco.mj_getState(self.model, self.data, endpoint, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(endpoint, result[1][-1])
        return result, impacts


def audit_trial(env, wrapped, policy, replay, reference):
    identity = {k: reference[k] for k in IDENTITY}
    if not reference["blade_contact_seen"]:
        return {
            **identity,
            "status": "unavailable_no_blade_contact",
            "parent_seconds": reference["seconds"],
            "parent_failures": reference["failures"],
            "diagnostic": None,
        }
    onset = next(e for e in reference["blade_events"] if e["event"] == "blade_contact_start")
    separation = next(
        e for e in reference["blade_events"] if e["event"] == "first_blade_separation"
    )
    contact_step = round(onset["seconds"] / env.cfg.sim_dt) - 1
    final_step = round(separation["seconds"] / env.cfg.sim_dt) - 1
    window_steps = round(0.1 / env.cfg.sim_dt)
    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [reference["offset_m"]]
    env.reset(seed=reference["seed"])
    term = env.action_manager.get_term("residual")
    model = replay.model
    assert not model.actuator_ctrllimited.any(), "headroom requires unclamped position targets"
    joints = model.actuator_trnid[:, 0]
    assert [model.joint(j).name for j in joints] == SDK_JOINTS
    arm = term.arm_ids
    qcols = 1 + model.jnt_qposadr[joints[arm]]
    vcols = 1 + model.nq + model.jnt_dofadr[joints[arm]]
    np.testing.assert_array_equal(-model.actuator_forcerange[:, 0], model.actuator_forcerange[:, 1])
    limits = model.actuator_forcerange[arm, 1]
    traces, raw, errors, velocities, fractions, episodes = [], [], [], [], [], []
    first_loaded = None
    replay.state_error = replay.sensor_error = 0.0
    for tick in range(final_step // replay.steps + 1):
        with torch.inference_mode():
            action = (
                policy(wrapped.get_observations()).numpy()
                if reference["controller"] == "ppo"
                else np.zeros((1, 7), dtype=np.float32)
            )
        result, impacts = replay.step(env, action, tick)
        state, trajectory, forces = result[:3]
        selected = window_slice(tick, replay.steps, contact_step, window_steps)
        q = trajectory[selected][:, qcols]
        if len(q):
            target = term.processed_action[0, arm].copy()
            traces.append(
                {
                    "control_tick": tick,
                    "first_solve_substep": tick * replay.steps + selected.start,
                    "last_solve_substep": tick * replay.steps + selected.stop - 1,
                    "physics_samples": len(q),
                    "raw_action": action[0].tolist(),
                    "residual_radians": term.residual_radians[0].tolist(),
                    "target_radians": target.tolist(),
                    "first_integrated_q_rad": q[0].tolist(),
                    "last_integrated_q_rad": q[-1].tolist(),
                }
            )
            raw.append(np.broadcast_to(action[0], q.shape).copy())
            errors.append(target - q)
            velocities.append(trajectory[selected][:, vcols])
            fractions.append(np.abs(forces[selected][:, arm]) / limits)
        for impact in impacts:
            if impact["solve_substep"] > final_step:
                break
            collect_episode(episodes, impact, env.cfg.sim_dt)
            if first_loaded is None and any(c["normal_force_n"] > 0 for c in impact["contacts"]):
                first_loaded = impact
        assert not state.terminated[0] and not state.truncated[0], "contact prefix ended early"
    assert len(episodes) == 1 and episodes[0]["end"] is not None
    episode = episodes[0]
    assert episode["start"]["integrated_seconds"] == onset["seconds"]
    assert episode["end"]["integrated_seconds"] == separation["seconds"]
    assert (
        episode["end"]["post_ball_velocity_world_m_s"][0]
        == reference["first_separation_ball_vx_m_s"]
    )
    episode["duration_seconds"] = (final_step - contact_step) * env.cfg.sim_dt
    statistics = summarize_window(
        *(np.concatenate(v) for v in (raw, errors, velocities, fractions))
    )
    assert statistics["physics_samples"] == min(contact_step, window_steps)
    return {
        **identity,
        "status": "first_impact_prefix_reproduced",
        "parent_seconds": reference["seconds"],
        "parent_failures": reference["failures"],
        "diagnostic": {
            "prefix_control_seconds": (tick + 1) * env.step_dt,
            "maximum_endpoint_state_error": replay.state_error,
            "maximum_endpoint_sensor_error": replay.sensor_error,
            "arm_joints": [SDK_JOINTS[i] for i in arm],
            "residual_limits_rad": RESIDUAL_LIMITS.tolist(),
            "first_episode": episode,
            "first_loaded_impact": first_loaded,
            "window_seconds": statistics["physics_samples"] * env.cfg.sim_dt,
            "precontact_control_samples": traces,
            "precontact_statistics": statistics,
        },
    }


def audit():
    parent_path = DIRECTORY / "trained_evaluation.json"
    parent_hash = sha256(parent_path)
    parent = json.loads(parent_path.read_text())["reports"][1]
    validate_pool(parent["rows"])
    assert parent["evaluation"]["physics_dt_seconds"] == 0.000125
    assert {name: version(name) for name in parent["versions"]} == parent["versions"]
    assert parent["external_asset_sha256"] == ASSET_HASHES
    sources = dict(parent["source_sha256"])
    for name in (
        "scripts/audit_g1_cricket_impact_headroom.py",
        "docs/g1_cricket_impact_headroom_v1.md",
    ):
        sources[name] = sha256(ROOT / name)
    pinned = {
        **sources,
        str(parent_path.relative_to(ROOT)): parent_hash,
        parent["checkpoint"]["path"]: parent["checkpoint"]["sha256"],
        **{
            f"g1_cricket_results/impact_events_v1/right/{name}.json": parent[f"{name}_sha256"]
            for name in ("run_config", "run_summary")
        },
    }
    for name, expected in pinned.items():
        assert sha256(ROOT / name) == expected, name
    owner = OmegaConf.create(
        json.loads((DIRECTORY / "right/run_config.json").read_text())["config"]
    )
    owner.env.sim_dt = 0.000125
    rows = []
    for hand in ("right", "left"):
        override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
        override.update(handedness=hand, auto_reset=False)
        env = create_env(owner, num_envs=1, env_cfg_override=override)
        try:
            wrapped, policy = load_policy(owner, env, ROOT / parent["checkpoint"]["path"])
            replay = MotionReplay(env)
            for reference in parent["rows"]:
                if reference["hand"] == hand:
                    rows.append(audit_trial(env, wrapped, policy, replay, reference))
        finally:
            env.close()
    validate_pool(rows)
    for name, expected in pinned.items():
        assert sha256(ROOT / name) == expected, name
    result = {
        "scope": "first_impact_prefix_headroom_diagnostic_not_training_or_promotion",
        "parent_report_sha256": parent_hash,
        "input_sha256": pinned,
        "checkpoint": parent["checkpoint"],
        "versions": parent["versions"],
        "external_asset_sha256": parent["external_asset_sha256"],
        "state_phase": "post-integration q/qvel; solved forces and contact-point velocities; not an instantaneous PD-law reconstruction",
        "window_selection": "100 ms before first blade-contact solve, excluding that solve; physics-step-weighted clipping fractions",
        "rows": rows,
        "physical_calibration_validated": False,
        "policy_promoted": False,
    }
    (DIRECTORY / "impact_headroom.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"rows": len(rows), "prefixes": sum(r["diagnostic"] is not None for r in rows)})
    )


if __name__ == "__main__":
    audit()
