"""Frozen-policy timestep audit of all native G1 residual development trials."""

import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from evaluate_g1_cricket_residual import ROOT, SEEDS, CricketReplay, load_policy, sha256
from omegaconf import OmegaConf

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.residual import TOSS_OFFSETS

TIMESTEPS = (0.002, 0.001, 0.0005)
TOLERANCES = {
    "blade_peak_force_norm_n": 1.0,
    "blade_force_norm_integral_ns": 0.005,
    "blade_contact_seconds": 0.002,
    "blade_max_penetration_m": 0.0001,
    "first_separation_ball_vx_m_s": 0.05,
}


def consistency(coarse, fine):
    failures = []
    for key in ("blade_contact_seen", "guard_contacts", "seconds", "terminated"):
        if coarse[key] != fine[key]:
            failures.append(key)
    for key, absolute in TOLERANCES.items():
        a, b = coarse[key], fine[key]
        if a is None or b is None:
            if a != b:
                failures.append(key)
        elif abs(a - b) > max(absolute, 0.05 * abs(b)):
            failures.append(key)
    return failures


class ContactReplay(CricketReplay):
    def step(self, env, action):
        initial = env.get_physics_state_snapshot()
        result = super().step(env, action)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = env.action_manager.get_term("residual").processed_action[0]
        impulse = np.zeros(3)
        for _ in range(self.steps):
            mujoco.mj_step(self.model, self.data)
            for index, contact in enumerate(self.data.contact):
                if set(contact.geom) != {self.ball_geom, self.model.geom("bat_blade").id}:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                # MuJoCo reports the contact-frame wrench on geom2.
                sign = 1 if contact.geom[1] == self.ball_geom else -1
                impulse += sign * (contact.frame.reshape(3, 3).T @ wrench[:3]) * env.cfg.sim_dt
        endpoint = np.empty(initial.shape[1])
        mujoco.mj_getState(self.model, self.data, endpoint, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        np.testing.assert_array_equal(endpoint, result[1][-1])
        return result, impulse


def audit():
    directory = ROOT / "g1_cricket_results/residual_v3"
    parent_path = directory / "evaluation.json"
    parent = json.loads(parent_path.read_text())
    for name, expected in parent["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"evaluated source changed: {name}")
    checkpoint = ROOT / parent["checkpoint"]["path"]
    if sha256(checkpoint) != parent["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed")
    config_path = directory / "right/run_config.json"
    if sha256(config_path) != parent["run_config_sha256"]:
        raise ValueError("configuration changed")
    owner = OmegaConf.create(json.loads(config_path.read_text())["config"])
    rows, model_parameters = [], []
    for dt in TIMESTEPS:
        for hand in ("right", "left"):
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override.update(handedness=hand, auto_reset=False, sim_dt=dt)
            env = create_env(owner, num_envs=1, env_cfg_override=override)
            try:
                assert env.step_dt == 0.02 and env.cfg.sim_substeps == round(0.02 / dt)
                wrapped, policy = load_policy(owner, env, checkpoint)
                replay = ContactReplay(env)
                model = replay.model
                vx_index = 1 + model.nq + model.jnt_dofadr[model.joint("ball_free").id]
                model_parameters.append(
                    {
                        "hand": hand,
                        "sim_dt": dt,
                        "ctrl_dt": env.step_dt,
                        "integrator": int(model.opt.integrator),
                        "explicit_pair_count": model.npair,
                        "geoms": {
                            name: {
                                "solref": model.geom_solref[model.geom(name).id].tolist(),
                                "solimp": model.geom_solimp[model.geom(name).id].tolist(),
                                "margin": float(model.geom_margin[model.geom(name).id]),
                                "gap": float(model.geom_gap[model.geom(name).id]),
                            }
                            for name in ("ball_geom", "bat_blade")
                        },
                    }
                )
                for controller in ("zero_residual", "ppo"):
                    for offset in TOSS_OFFSETS:
                        env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [offset]
                        for seed in SEEDS:
                            env.reset(seed=seed)
                            count = active = 0
                            peak = integral = penetration = reward = 0.0
                            impulse = np.zeros(3)
                            guards = set()
                            separation = None
                            previous_blade = False
                            for tick in range(env.max_episode_length):
                                with torch.inference_mode():
                                    action = (
                                        policy(wrapped.get_observations()).numpy()
                                        if controller == "ppo"
                                        else np.zeros((1, 7), dtype=np.float32)
                                    )
                                result, step_impulse = replay.step(env, action)
                                state, trajectory, _, presence, _, _, contacts = result
                                impulse += step_impulse
                                reward += float(state.reward[0])
                                guards.update(
                                    np.asarray(replay.names)[presence.any(axis=0)].tolist()
                                )
                                for step, records in enumerate(contacts):
                                    blade = [c for c in records if c["geom"] == "bat_blade"]
                                    if previous_blade and not blade and separation is None:
                                        separation = float(trajectory[step, vx_index])
                                    previous_blade = bool(blade)
                                    count += bool(blade)
                                    active += any(c["force_norm_n"] > 0 for c in blade)
                                    for c in blade:
                                        peak = max(peak, c["force_norm_n"])
                                        integral += c["force_norm_n"] * dt
                                        penetration = max(penetration, -c["distance_m"])
                                if state.terminated[0] or state.truncated[0]:
                                    break
                            row = {
                                "sim_dt": dt,
                                "hand": hand,
                                "controller": controller,
                                "offset_m": offset,
                                "seed": seed,
                                "seconds": (tick + 1) * env.step_dt,
                                "terminated": bool(state.terminated[0]),
                                "return": reward,
                                "blade_contact_seen": count > 0,
                                "blade_contact_seconds": count * dt,
                                "blade_active_load_seconds": active * dt,
                                "blade_max_penetration_m": penetration,
                                "blade_peak_force_norm_n": peak,
                                "blade_force_norm_integral_ns": integral,
                                "blade_world_impulse_on_ball_ns": impulse.tolist(),
                                "first_separation_ball_vx_m_s": separation,
                                "guard_contacts": sorted(guards),
                                "maximum_endpoint_state_error": replay.state_error,
                                "maximum_endpoint_sensor_error": replay.sensor_error,
                            }
                            if dt == TIMESTEPS[0]:
                                old = next(
                                    r
                                    for r in parent["rows"]
                                    if all(
                                        r[k] == row[k]
                                        for k in ("hand", "controller", "offset_m", "seed")
                                    )
                                )
                                assert reward == old["return"] and row["seconds"] == old["seconds"]
                                assert count == old["ball_contact_physics_step_counts"].get(
                                    "bat_blade", 0
                                )
                                assert peak == old["ball_contact_peak_force_norm_n"].get(
                                    "bat_blade", 0
                                )
                                assert separation == old["first_separation_ball_vx_m_s"]
                                assert guards == {
                                    k
                                    for k, v in old["guard_contact_physics_step_counts"].items()
                                    if v
                                }
                            rows.append(row)
                print(
                    json.dumps({"sim_dt": dt, "hand": hand, "completed_rows": len(rows)}),
                    flush=True,
                )
            finally:
                env.close()
    comparisons = []
    for coarse, fine in zip(rows[96:192], rows[192:], strict=True):
        identity = {k: coarse[k] for k in ("hand", "controller", "offset_m", "seed")}
        assert all(fine[k] == v for k, v in identity.items())
        comparisons.append({**identity, "failed_checks": consistency(coarse, fine)})
    report = {
        "scope": "frozen_v3_policy_contact_resolution_not_training_or_promotion",
        "parent_report_sha256": sha256(parent_path),
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "scripts/audit_g1_cricket_contact_resolution.py",
                "docs/g1_cricket_contact_resolution.md",
            )
        },
        "timesteps": list(TIMESTEPS),
        "expected_rows": 288,
        "relative_tolerance": 0.05,
        "absolute_tolerances": TOLERANCES,
        "model_parameters": model_parameters,
        "rows": rows,
        "one_vs_half_ms_comparisons": comparisons,
        "numerically_consistent_rows": sum(not r["failed_checks"] for r in comparisons),
        "physical_calibration_validated": False,
    }
    (directory / "contact_resolution.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"numerically_consistent_rows": report["numerically_consistent_rows"], "total": 96}
        )
    )


if __name__ == "__main__":
    audit()
