"""All-row residual interception evaluation with independent physics-step replay."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch
from audit_g1_cricket_prior_substeps import IntervalReplay
from evaluate_g1_cricket_smoke import sha256
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from uni_rl.algos.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime

from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES
from unilab.tasks.manipulation.g1_cricket.residual import TOSS_OFFSETS
from unilab.tasks.manipulation.g1_cricket.scene import BALL_CONTACT_NAMES
from unilab.training import algo_config_dict
from unilab.utils.sim2sim import policy_load_dim_guard

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(4301, 4309))


class CricketReplay(IntervalReplay):
    def __init__(self, env):
        super().__init__(env)
        self.sensor_names += BALL_CONTACT_NAMES + ("bat_fixture_force", "bat_fixture_torque")
        self.native_sensors = env.scene.bind_sensor_data(self.sensor_names)
        self.indices = np.concatenate(
            [
                np.arange(
                    self.model.sensor(name).adr[0],
                    self.model.sensor(name).adr[0] + self.model.sensor(name).dim[0],
                )
                for name in self.sensor_names
            ]
        )
        self.ball_geom = self.model.geom("ball_geom").id

    def step(self, env, action):
        initial = env.get_physics_state_snapshot()
        state = env.step(action)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, initial[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.data.ctrl[:] = env.action_manager.get_term("residual").processed_action[0]
        trajectory, sensors, ball_contacts = [], [], []
        for _ in range(self.steps):
            mujoco.mj_step(self.model, self.data)
            snapshot = np.empty(initial.shape[1])
            mujoco.mj_getState(self.model, self.data, snapshot, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            trajectory.append(snapshot)
            sensors.append(self.data.sensordata[self.indices].copy())
            contacts = []
            for index, contact in enumerate(self.data.contact):
                if self.ball_geom not in contact.geom:
                    continue
                other = int(
                    contact.geom[1] if contact.geom[0] == self.ball_geom else contact.geom[0]
                )
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                contacts.append(
                    {
                        "geom": self.model.geom(other).name,
                        "force_norm_n": float(np.linalg.norm(wrench[:3])),
                        "distance_m": float(contact.dist),
                    }
                )
            ball_contacts.append(contacts)
        trajectory, sensors = np.asarray(trajectory), np.asarray(sensors)
        actual = env.get_physics_state_snapshot()
        expected = trajectory[-1:].astype(actual.dtype)
        native = self.native_sensors.read()[0]
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(sensors[-1].astype(native.dtype), native)
        self.state_error = max(self.state_error, float(np.abs(expected - actual).max()))
        self.sensor_error = max(
            self.sensor_error, float(np.abs(sensors[-1].astype(native.dtype) - native).max())
        )
        if not np.isfinite(trajectory).all() or not np.isfinite(sensors).all():
            raise RuntimeError("non-finite cricket replay")
        guard_width = int(self.slot_limits.size * 17)
        guard = sensors[:, :guard_width].reshape(self.steps, -1, 17)
        ball = sensors[:, guard_width + 29 : -6].reshape(self.steps, 5, 4, 17)
        if (guard[..., 0] > self.slot_limits).any() or (ball[..., 0] > 4).any():
            raise RuntimeError("cricket contact sensor overflow")
        presence = np.maximum.reduceat(guard[..., 0] > 0, self.starts, axis=1)
        peaks = np.maximum.reduceat(np.linalg.norm(guard[..., 1:4], axis=-1), self.starts, axis=1)
        return (
            state,
            trajectory,
            sensors[:, guard_width : guard_width + 29],
            presence,
            peaks,
            sensors[:, -6:],
            ball_contacts,
        )


def load_policy(owner, env, checkpoint):
    rl_cfg = algo_config_dict(owner)
    runtime = resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)
    wrapped = runtime.wrapper_cls(env, device="cpu")
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    train_cfg["logger"] = "none"
    runner = (runtime.runner_cls or OnPolicyRunner)(wrapped, train_cfg, log_dir=None, device="cpu")
    with policy_load_dim_guard(env_obs_dim=wrapped.num_obs, env_action_dim=7, algo_name="ppo"):
        runner.load(
            str(checkpoint),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
        )
    return wrapped, runner.get_inference_policy(device="cpu")


def evaluate(run_dir, evaluation_overrides=None):
    saved = json.loads((run_dir / "run_config.json").read_text())
    summary = json.loads((run_dir / "run_summary.json").read_text())
    owner = OmegaConf.create(saved["config"])
    if evaluation_overrides:
        owner = OmegaConf.merge(owner, evaluation_overrides)
    checkpoint = ROOT / summary["last_checkpoint"]
    rows, contact_models = [], []
    for hand in ("right", "left"):
        override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
        override.update(handedness=hand, auto_reset=False)
        env = create_env(owner, num_envs=1, env_cfg_override=override)
        try:
            wrapped, policy = load_policy(owner, env, checkpoint)
            replay = CricketReplay(env)
            model = replay.model
            contact_models.append(
                {
                    "hand": hand,
                    "pairs": [
                        {
                            "name": model.pair(i).name,
                            "geoms": [
                                model.geom(int(model.pair_geom1[i])).name,
                                model.geom(int(model.pair_geom2[i])).name,
                            ],
                            "solref": model.pair_solref[i].tolist(),
                            "solimp": model.pair_solimp[i].tolist(),
                            "friction": model.pair_friction[i].tolist(),
                            "dim": int(model.pair_dim[i]),
                            "margin": float(model.pair_margin[i]),
                            "gap": float(model.pair_gap[i]),
                        }
                        for i in range(model.npair)
                    ],
                }
            )
            joints = model.actuator_trnid[:, 0]
            joint_indices = 1 + model.jnt_qposadr[joints]
            limits = model.jnt_range[joints]
            force_limits = model.actuator_forcerange[:, 1]
            ball_velocity_index = 1 + model.nq + model.jnt_dofadr[model.joint("ball_free").id]
            for controller in ("zero_residual", "ppo"):
                for offset in TOSS_OFFSETS:
                    env.event_manager.get_term_cfg("reset_toss").params["offsets"] = [offset]
                    for seed in SEEDS:
                        env.reset(seed=seed)
                        counts = np.zeros(len(replay.names), dtype=int)
                        peaks = np.zeros(len(replay.names))
                        min_height, min_up = 1.0, 1.0
                        excess = fraction = episode_return = penetration = 0.0
                        fixture_peak = np.zeros(2)
                        first_contact = separation_velocity = None
                        blade_seen = previous_blade = False
                        ball_counts, ball_peaks = {}, {}
                        failures = set()
                        events = []
                        replay.state_error = replay.sensor_error = 0.0
                        for tick in range(env.max_episode_length):
                            with torch.inference_mode():
                                action = (
                                    policy(wrapped.get_observations()).numpy()
                                    if controller == "ppo"
                                    else np.zeros((1, 7), dtype=np.float32)
                                )
                            (
                                state,
                                trajectory,
                                forces,
                                presence,
                                contact_peaks,
                                fixture,
                                contacts,
                            ) = replay.step(env, action)
                            episode_return += float(state.reward[0])
                            counts += presence.sum(axis=0)
                            peaks = np.maximum(peaks, contact_peaks.max(axis=0))
                            fixture_peak = np.maximum(
                                fixture_peak,
                                np.linalg.norm(fixture.reshape(-1, 2, 3), axis=2).max(axis=0),
                            )
                            min_height = min(min_height, float(trajectory[:, 3].min()))
                            min_up = min(
                                min_up, float((1 - 2 * (trajectory[:, 5:7] ** 2).sum(axis=1)).min())
                            )
                            q = trajectory[:, joint_indices]
                            excess = max(
                                excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max())
                            )
                            fraction = max(fraction, float((np.abs(forces) / force_limits).max()))
                            for substep, step_contacts in enumerate(contacts):
                                seconds = (tick * replay.steps + substep + 1) * env.cfg.sim_dt
                                names = {c["geom"] for c in step_contacts}
                                blade = "bat_blade" in names
                                if names and first_contact is None:
                                    first_contact = {"seconds": seconds, "geoms": sorted(names)}
                                    if names != {"bat_blade"}:
                                        failures.add("first_contact_not_blade_only")
                                if blade and not previous_blade:
                                    events.append(
                                        {"seconds": seconds, "event": "blade_contact_start"}
                                    )
                                if previous_blade and not blade and separation_velocity is None:
                                    separation_velocity = float(
                                        trajectory[substep, ball_velocity_index]
                                    )
                                    events.append(
                                        {
                                            "seconds": seconds,
                                            "event": "first_blade_separation",
                                            "ball_vx_m_s": separation_velocity,
                                        }
                                    )
                                for contact in step_contacts:
                                    name = contact["geom"]
                                    if name == "bat_blade":
                                        penetration = max(penetration, -contact["distance_m"])
                                    ball_peaks[name] = max(
                                        ball_peaks.get(name, 0.0), contact["force_norm_n"]
                                    )
                                    if name not in {"bat_blade", "pitch"} or (
                                        name == "pitch" and not blade_seen
                                    ):
                                        failures.add(f"ball_contact:{name}")
                                for name in names:
                                    ball_counts[name] = ball_counts.get(name, 0) + 1
                                blade_seen |= blade
                                previous_blade = blade
                            if state.terminated[0] or state.truncated[0]:
                                break
                        if not (state.truncated[0] and not state.terminated[0]):
                            failures.add("native_episode_incomplete")
                        if not blade_seen:
                            failures.add("no_blade_contact")
                        if separation_velocity is None or separation_velocity <= 1:
                            failures.add("outgoing_velocity_not_above_1_m_s")
                        for name, bad in (
                            ("guarded_contact", counts.any()),
                            ("pelvis_height", min_height < 0.48),
                            ("pelvis_orientation", min_up < 0.65),
                            ("joint_limit", excess > 1e-6),
                            ("actuator_limit", fraction > 1 + 1e-6),
                        ):
                            if bad:
                                failures.add(name)
                        rows.append(
                            {
                                "hand": hand,
                                "controller": controller,
                                "offset_m": offset,
                                "seed": seed,
                                "seconds": (tick + 1) * env.step_dt,
                                "passed": not failures,
                                "failures": sorted(failures),
                                "return": episode_return,
                                "first_ball_contact": first_contact,
                                "blade_contact_seen": blade_seen,
                                **(
                                    {"maximum_blade_penetration_m": penetration}
                                    if evaluation_overrides
                                    else {}
                                ),
                                "first_separation_ball_vx_m_s": separation_velocity,
                                "blade_events": events,
                                "ball_contact_physics_step_counts": ball_counts,
                                "ball_contact_peak_force_norm_n": ball_peaks,
                                "guard_contact_physics_step_counts": dict(
                                    zip(replay.names, counts.tolist(), strict=True)
                                ),
                                "guard_contact_peak_force_norm_n": dict(
                                    zip(replay.names, peaks.tolist(), strict=True)
                                ),
                                "fixture_peak_force_norm_n": float(fixture_peak[0]),
                                "fixture_peak_torque_norm_nm": float(fixture_peak[1]),
                                "minimum_pelvis_height_m": min_height,
                                "minimum_pelvis_up_z": min_up,
                                "maximum_joint_limit_excess_rad": excess,
                                "maximum_actuator_limit_fraction": fraction,
                                "maximum_endpoint_state_error": replay.state_error,
                                "maximum_endpoint_sensor_error": replay.sensor_error,
                            }
                        )
        finally:
            env.close()
    sources = [
        "scripts/evaluate_g1_cricket_residual.py",
        "scripts/audit_g1_cricket_prior_substeps.py",
        "src/unilab/tasks/manipulation/g1_cricket/residual.py",
        "src/unilab/tasks/manipulation/g1_cricket/prior.py",
        "src/unilab/tasks/manipulation/g1_cricket/task.py",
        "src/unilab/tasks/manipulation/g1_cricket/scene.py",
        "src/unilab/conf/ppo/task/g1_cricket_residual_v1/mujoco.yaml",
        "src/unilab/conf/ppo/task/g1_cricket_prior_v1/mujoco.yaml",
        "src/unilab/conf/ppo/task/g1_cricket_prior_v2/mujoco.yaml",
        "src/unilab/assets/robots/g1/g1.xml",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "docs/g1_cricket_residual_v1.md",
    ]
    return {
        "scope": "staged_soft_toss_residual_interception_not_full_cricket_or_bowling",
        "rows": rows,
        "contact_models": contact_models,
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256(checkpoint)},
        "training": {
            "transitions": summary["run_env_steps"],
            "seed": summary["effective_seed"],
            "hand": "right",
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "run_config_sha256": sha256(run_dir / "run_config.json"),
        "run_summary_sha256": sha256(run_dir / "run_summary.json"),
        "evaluation_overrides": evaluation_overrides or {},
        "external_asset_sha256": ASSET_HASHES,
        "evaluation": {
            "seeds": list(SEEDS),
            "offsets_m": list(TOSS_OFFSETS),
            "left_hand": "untrained_transfer",
            "expected_rows": 96,
            "physics_dt_seconds": float(owner.env.sim_dt),
            "control_dt_seconds": 0.02,
            "horizon_seconds": 2.0,
            "strict_outgoing_vx_threshold_m_s": 1.0,
            "contact_scope": "Every physics solve at the declared timestep, independently replayed from public native state with executed targets; exact float32 endpoint state and named-sensor equality required",
            "tactile_scope": "Simulated per-geometry occupancy counts and peak contact-force norms, not hardware tactile pressure; fixture wrench includes inertial and gravity loads",
            "force_units": "contact/fixture force N, fixture torque N m; actuator force fraction is dimensionless",
            "gate": "Blade-first, no pre-hit ground or any ball/body/handle/wicket contact, vx > 1 immediately after first separation, full 2 s stability/guard/joint/actuator checks",
        },
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "versions": {
            name: version(name)
            for name in ("unilab-rl", "unisim-core", "mujoco", "onnxruntime", "torch", "rsl-rl-lib")
        },
        "aggregates": [
            {
                "hand": hand,
                "controller": controller,
                "passed": sum(
                    r["passed"] for r in rows if r["hand"] == hand and r["controller"] == controller
                ),
                "total": 24,
            }
            for hand in ("right", "left")
            for controller in ("zero_residual", "ppo")
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "g1_cricket_results/residual_v1/right"
    )
    args = parser.parse_args()
    report = evaluate(args.run_dir)
    output = args.run_dir.parent / "evaluation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["aggregates"], indent=2))
