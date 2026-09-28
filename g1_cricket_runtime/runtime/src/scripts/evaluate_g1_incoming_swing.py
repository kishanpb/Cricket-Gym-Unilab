"""Test both retained swing actors against every declared incoming delivery."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from audit_g1_guard_contacts import METRICS, GuardAuditCfg, GuardContactAudit, checks
from initialize_g1_guard_actor import ROOT, digest, rollout
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.tasks.manipulation.g1_cricket.articulated_learning import ARTICULATED_DELIVERY_POOL
from unilab.tasks.manipulation.g1_cricket.scene import (
    BALL_CONTACT_NAMES,
    CONTACT_SLOTS,
    CONTACT_WIDTH,
)
from unilab.tasks.manipulation.g1_cricket.task import make_g1_cricket_env
from unilab.training import algo_config_dict

BALL_COLUMNS = (
    "ball_x_m",
    "ball_y_m",
    "ball_z_m",
    "ball_vx_m_s",
    "ball_vy_m_s",
    "ball_vz_m_s",
    *(f"{name}_normal_force_n" for name in BALL_CONTACT_NAMES),
    *(f"{name}_penetration_m" for name in BALL_CONTACT_NAMES),
    "blade_x_m",
    "blade_y_m",
    "blade_z_m",
)


class IncomingContactAudit(GuardContactAudit):
    def sensor_names(self, env):
        return (*super().sensor_names(env), "bat_center_world")

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ball_blocks = []
        self.ball_robot_blocks = []

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.ball_blocks.clear()
        self.ball_robot_blocks.clear()

    def observe(self, sensors, integrated_velocity):
        super().observe(sensors, integrated_velocity)
        if self.check_ball_robot_contact:
            self.ball_robot_blocks.append(self.ball_robot_metrics[0].copy())
        width = len(BALL_CONTACT_NAMES) * CONTACT_SLOTS * CONTACT_WIDTH
        contacts = sensors[0, :, :width].reshape(
            -1, len(BALL_CONTACT_NAMES), CONTACT_SLOTS, CONTACT_WIDTH
        )
        active = contacts[..., 0] > 0
        normal = np.where(active, contacts[..., 1], 0).sum(axis=-1)
        depth = np.maximum(0, np.where(active, -contacts[..., 7], 0).max(axis=-1))
        self.ball_blocks.append(
            np.column_stack(
                (
                    sensors[0, :, width : width + 3] - self.origins[0],
                    integrated_velocity[0],
                    normal,
                    depth,
                    sensors[0, :, -3:] - self.origins[0],
                )
            )
        )

    def arrays(self):
        arrays = {**super().arrays(), "ball_metrics": np.concatenate(self.ball_blocks)}
        if self.check_ball_robot_contact:
            arrays["ball_robot_metrics"] = np.concatenate(self.ball_robot_blocks)
        return arrays


def make_env(owner):
    owner.reward.boundary.func = "scripts.evaluate_g1_incoming_swing.IncomingContactAudit"
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    cfg = GuardAuditCfg()
    apply_cfg_overrides(cfg, override)
    return make_g1_cricket_env(cfg, num_envs=1, backend_type="mujoco")


def delivery_diagnostic(values, dt, *, sample_times=None):
    columns = dict(zip(BALL_COLUMNS, values.T, strict=True))
    pitch = columns["ball_pitch_normal_force_n"] > 0
    completed = np.r_[False, pitch[:-1]] & ~pitch
    delta = columns["ball_x_m"] - columns["blade_x_m"]
    crossings = np.flatnonzero((delta[1:] <= 0) & (delta[:-1] > 0)) + 1
    gate = int(crossings[0]) if len(crossings) else None
    bat = columns["ball_bat_normal_force_n"] > 0
    contact = np.flatnonzero(bat)
    times = np.arange(len(values)) * dt if sample_times is None else np.asarray(sample_times)
    if times.shape != (len(values),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Delivery sample times must be finite, increasing and match the rows")
    return {
        "bat_plane_reached": gate is not None,
        "bat_plane_time_s": float(times[gate]) if gate is not None else None,
        "ball_height_at_bat_plane_m": float(values[gate, 2]) if gate is not None else None,
        "completed_bounces_at_bat_plane": int(completed[:gate].sum()) if gate is not None else None,
        "first_loaded_bat_contact_s": float(times[contact[0]]) if len(contact) else None,
        "completed_pitch_contacts": int(completed.sum()),
        "minimum_ball_blade_center_distance_m": float(
            np.linalg.norm(values[:, :3] - values[:, -3:], axis=1).min()
        ),
        "peak_ball_bat_normal_force_n": float(columns["ball_bat_normal_force_n"].max()),
        "maximum_ball_bat_penetration_m": float(columns["ball_bat_penetration_m"].max()),
        "scope": "plane/center diagnostics are not contact or scoring evidence; unreached plane is censored at episode stop",
    }


def evaluate(output, checkpoints):
    parent = checkpoints / "evaluation/summary.json"
    fingerprints = dict(json.loads(parent.read_text())["source_input_sha256"])
    fingerprints.update({str(p.relative_to(ROOT)): digest(p) for p in (parent, Path(__file__))})
    for name, expected in fingerprints.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"source/input changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for hand in ("right", "left"):
        run = checkpoints / f"ppo_{hand}"
        saved = json.loads((run / "run_config.json").read_text())["config"]
        checkpoint = Path(json.loads((run / "run_summary.json").read_text())["last_checkpoint"])
        weights = torch.load(checkpoint, weights_only=False, map_location="cpu")["actor_state_dict"]
        for feed_index, feed in enumerate(ARTICULATED_DELIVERY_POOL):
            owner = OmegaConf.create(saved)
            owner.env.events.delivery = {
                "func": "unilab.tasks.manipulation.g1_cricket.articulated_learning.ResetArticulatedDelivery",
                "mode": "reset",
                "params": {"evaluation_pool": False, "delivery_index": feed_index},
            }
            env = make_env(owner)
            try:
                torch.manual_seed(1)
                wrapper = RslRlVecEnvWrapper(env, device="cpu")
                config = normalize_ppo_train_cfg(algo_config_dict(owner))
                config["logger"] = "none"
                runner = OnPolicyRunner(wrapper, config, log_dir=None, device="cpu")
                runner.alg.actor.load_state_dict(weights)
                result, trace = rollout(env, wrapper, runner.alg.actor)
                observer = env.reward_manager.get_term_cfg("boundary").func
                arrays = observer.arrays()
                gates = checks(arrays["metrics"], arrays["contact_time_s"])
                del gates["retained_bat"], gates["wrist_stability"]
                gates["native_episode_completed"] = result["completed_without_termination"]
                case = f"{hand}_feed{feed_index}"
                np.savez_compressed(output / f"{case}.npz", **trace, **arrays)
                row = {
                    "case": case,
                    "hand": hand,
                    "feed_index": feed_index,
                    "canonical_feed": feed,
                    "result": result,
                    "checks": gates,
                    "physical_checks_passed": all(gates.values()),
                    "valid_hit": bool(observer.valid_hit[0]),
                    "disqualified": bool(observer.disqualified[0]),
                    "loaded_hit_count": int(observer.loaded_hit_count[0]),
                    "boundary_runs": int(observer.boundary_runs[0]),
                    "qualified_boundary_runs": int(observer.boundary_runs[0])
                    if all(gates.values()) and not observer.disqualified[0]
                    else 0,
                    "separation_velocity_m_s": observer.separation_velocity_m_s[0].tolist(),
                    "delivery": delivery_diagnostic(arrays["ball_metrics"], env.cfg.sim_dt),
                    "maximum": dict(
                        zip(METRICS, arrays["metrics"].max(axis=0).tolist(), strict=True)
                    ),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    for name, expected in fingerprints.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"source/input changed during evaluation: {name}")
    report = {
        "scope": "all-feed diagnostic transfer of no-ball trained PPO; not learned interception or a showcase",
        "protocol": "both final actors, all seven existing feeds, seed17 resets, fresh recurrent state, eight seconds or first failure",
        "changed_axis": "restore declared incoming delivery instead of parking the ball",
        "actor_observes_ball": False,
        "learned_swing_timing": False,
        "assistance": "nominal coordinated body reference and finger PD; learned 12-leg commands; no runtime bat weld or root fixture",
        "ball_time_scope": "positions/contacts are native pre-integration substeps; velocities are integrated per-substep values",
        "metric_columns": METRICS,
        "ball_columns": BALL_COLUMNS,
        "case_count": len(rows),
        "valid_hit_cases": sum(row["valid_hit"] for row in rows),
        "qualified_boundary_runs": sum(row["qualified_boundary_runs"] for row in rows),
        "rows": rows,
        "source_input_sha256": fingerprints,
        "artifact_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--checkpoints", type=Path, default=ROOT / "g1_cricket_results/phase_leg_swing_ppo_v1"
    )
    args = parser.parse_args()
    evaluate(args.output.resolve(), args.checkpoints.resolve())
