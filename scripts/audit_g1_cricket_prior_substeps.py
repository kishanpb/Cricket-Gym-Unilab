"""Replay each native G1 interval to inspect all physics-step contacts and loads."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import mujoco
import mujoco.rollout
import numpy as np
from evaluate_g1_cricket_prior import ROOT, SEEDS, load_prior, make_env, sha256

from unilab.tasks.manipulation.g1_cricket.prior import (
    ASSET_HASHES,
    GUARD_SLOTS,
    POLICY_TO_SDK,
    SDK_JOINTS,
)
from unilab.tasks.manipulation.g1_cricket.task import fallen


class IntervalReplay:
    """Independent serial solver, seeded only from the public native snapshot."""

    def __init__(self, env):
        spec = mujoco.MjSpec.from_file(str(Path(env.scene_directory.name) / "cricket.xml"))
        spec.compiler.discardvisual = True
        # Native tracking-sensor injection serializes MjSpec before compilation.
        self.model = mujoco.MjModel.from_xml_string(spec.to_xml())
        self.model.opt.timestep = env.cfg.sim_dt
        self.data = mujoco.MjData(self.model)
        self.steps = env.cfg.sim_substeps
        self.names = env.cfg.bat_guard_sensor_names
        force_names = tuple(f"prior_force_{name}" for name in SDK_JOINTS)
        self.sensor_names = self.names + force_names
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
        capacities = np.array(
            [4 if name.startswith("bat_") else GUARD_SLOTS for name in self.names]
        )
        self.starts = np.r_[0, np.cumsum(capacities)[:-1]]
        self.slot_limits = np.repeat(capacities, capacities)
        self.state_error = self.sensor_error = 0.0

    def step(self, env, action):
        initial = env.get_physics_state_snapshot()
        state = env.step(action)
        targets = env.action_manager.get_term("joint_pos").processed_action
        trajectory, sensors = mujoco.rollout.rollout(
            self.model,
            self.data,
            initial,
            control=np.broadcast_to(targets[:, None, :], (1, self.steps, 29)),
        )
        selected = sensors[0][:, self.indices]
        actual = env.get_physics_state_snapshot()
        expected = trajectory[:, -1].astype(actual.dtype)
        native = self.native_sensors.read()[0]
        self.state_error = max(self.state_error, float(np.abs(expected - actual).max()))
        self.sensor_error = max(self.sensor_error, float(np.abs(selected[-1] - native).max()))
        # A mismatched replay cannot supply evidence about the native interval.
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(selected[-1].astype(native.dtype), native)
        records = selected[:, :-29].reshape(self.steps, -1, 17)
        if not all(np.isfinite(x).all() for x in (trajectory, selected)):
            raise RuntimeError("non-finite interval replay")
        if np.any(records[..., 0] > self.slot_limits):
            raise RuntimeError("replayed contact overflow")
        np.testing.assert_array_equal(records[-1, :, 0], native[:-29].reshape(-1, 17)[:, 0])
        presence = np.maximum.reduceat(records[..., 0] > 0, self.starts, axis=1)
        peaks = np.maximum.reduceat(np.linalg.norm(records[..., 1:4], axis=-1), self.starts, axis=1)
        return state, trajectory[0], selected[:, -29:], presence, peaks


def audit(assets: Path) -> dict:
    policy = load_prior(assets)
    parent_path = ROOT / "g1_cricket_results/unitree_prior_v2/evaluation.json"
    parent = json.loads(parent_path.read_text())
    for name, expected in parent["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"parent source changed: {name}")
    robot_hashes = {
        key: sha256(ROOT / "src/unilab/assets/robots/g1" / name)
        for key, name in (
            ("robot_xml_sha256", "g1.xml"),
            ("robot_scene_flat_sha256", "scene_flat.xml"),
        )
    }
    if any(parent[key] != digest for key, digest in robot_hashes.items()):
        raise ValueError("parent robot asset changed")
    if parent["external_asset_sha256"] != ASSET_HASHES:
        raise ValueError("parent uses different external policy assets")
    rows = []
    for hand in ("none", "right", "left"):
        env = make_env(hand, "v2")
        try:
            replay = IntervalReplay(env)
            model = replay.model
            joints = model.actuator_trnid[:, 0]
            qpos_indices = 1 + model.jnt_qposadr[joints]
            limits = model.jnt_range[joints]
            force_limits = model.actuator_forcerange[:, 1]
            for controller in ("constant_target", "unitree_onnx"):
                for seed in SEEDS:
                    obs, _ = env.reset(seed=seed)
                    start = env.scene["robot"].data.root_link_pos_w[0].copy()
                    min_height = float(start[2])
                    min_up_z = 1.0
                    drift = excess = fraction = 0.0
                    counts = np.zeros(len(replay.names), dtype=int)
                    peaks = np.zeros(len(replay.names))
                    first_contact = None
                    replay.state_error = replay.sensor_error = 0.0
                    for tick in range(env.max_episode_length):
                        policy_action = (
                            policy.run(["actions"], {"obs": obs["obs"].astype(np.float32)})[0]
                            if controller == "unitree_onnx"
                            else np.zeros((1, 29), dtype=np.float32)
                        )
                        action = np.empty_like(policy_action)
                        action[:, POLICY_TO_SDK] = policy_action
                        state, trajectory, forces, presence, contact_peaks = replay.step(
                            env, action
                        )
                        obs = state.obs
                        min_height = min(min_height, float(trajectory[:, 3].min()))
                        min_up_z = min(
                            min_up_z, float((1 - 2 * (trajectory[:, 5:7] ** 2).sum(axis=1)).min())
                        )
                        drift = max(
                            drift,
                            float(np.linalg.norm(trajectory[:, 1:3] - start[:2], axis=1).max()),
                        )
                        q = trajectory[:, qpos_indices]
                        excess = max(
                            excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max())
                        )
                        fraction = max(fraction, float((np.abs(forces) / force_limits).max()))
                        counts += presence.sum(axis=0)
                        peaks = np.maximum(peaks, contact_peaks.max(axis=0))
                        if first_contact is None and presence.any():
                            substep, channel = np.argwhere(presence)[0]
                            first_contact = {
                                "seconds": (tick * replay.steps + int(substep) + 1)
                                * env.cfg.sim_dt,
                                "channel": replay.names[channel],
                            }
                        if state.terminated[0] or state.truncated[0]:
                            break
                    else:
                        raise RuntimeError("native episode exceeded declared horizon")
                    native_completed = bool(state.truncated[0] and not state.terminated[0])
                    rows.append(
                        {
                            "hand": hand,
                            "controller": controller,
                            "seed": seed,
                            "physics_steps": (tick + 1) * replay.steps,
                            "seconds": (tick + 1) * env.step_dt,
                            "native_completed": native_completed,
                            "substep_gate_passed": native_completed
                            and not counts.any()
                            and min_height >= 0.48
                            and min_up_z >= 0.65
                            and excess <= 1e-6
                            and fraction <= 1 + 1e-6,
                            "fell": bool(fallen(env)[0]),
                            "first_guarded_contact": first_contact,
                            "minimum_pelvis_height_m": min_height,
                            "minimum_pelvis_up_z": min_up_z,
                            "maximum_xy_drift_m": drift,
                            "maximum_joint_limit_excess_rad": excess,
                            "maximum_actuator_force_limit_fraction": fraction,
                            "contact_presence_physics_step_counts": dict(
                                zip(replay.names, counts.tolist(), strict=True)
                            ),
                            "contact_force_norm_physics_step_peaks_n": dict(
                                zip(replay.names, peaks.tolist(), strict=True)
                            ),
                            "compared_policy_ticks": tick + 1,
                            "maximum_endpoint_state_error": replay.state_error,
                            "maximum_endpoint_sensor_error": replay.sensor_error,
                            "final_root_pose": env.scene["robot"].data.root_link_pose_w[0].tolist(),
                        }
                    )
        finally:
            env.close()
    for row, old in zip(rows, parent["rows"], strict=True):
        for name in ("hand", "controller", "seed", "seconds", "final_root_pose"):
            if row[name] != old[name]:
                raise ValueError(f"native parent replay differs: {name}")
        if row["native_completed"] != old["completed"]:
            raise ValueError("native parent completion differs")
    sources = list(parent["source_sha256"]) + ["scripts/audit_g1_cricket_prior_substeps.py"]
    return {
        "scope": "Native interval replay audit of external locomotion; no learned cricket or hardware force claim",
        "parent_report": str(parent_path.relative_to(ROOT)),
        "parent_report_sha256": sha256(parent_path),
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "external_asset_sha256": {name: sha256(assets / name) for name in ASSET_HASHES},
        "versions": {
            name: version(name)
            for name in ("unilab-rl", "unisim-core", "mujoco", "onnxruntime", "numpy")
        },
        **robot_hashes,
        "contract": {
            "policy_dt_s": 0.02,
            "physics_dt_s": 0.002,
            "replay_steps_per_interval": 10,
            "horizon_seconds": 10,
            "seeds": list(SEEDS),
            "comparison": "Every native control interval independently replayed from public float32 FULLPHYSICS snapshot with executed position targets; endpoints and named contact/force sensors checked before accepting intermediate evidence",
            "endpoint_comparison": "exact after the native float32 output cast",
            "sensor_error_scope": "Recorded raw float64 replay minus float32 native sensor rounding difference; float32 cast comparison must be exact",
            "contact_presence_endpoint_comparison": "exact",
            "contact_scope": "Every serially replayed 2 ms solver step; contact loads belong to the solve preceding the returned integrated state, not continuous collision detection or timestep-converged hardware loads",
            "native_pose_write_after_reset": False,
            "root_support": False,
            "rejected_approach": "Ten native 2 ms calls introduce extra float32 state roundtrips and are not equivalent to one native 20 ms call",
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "hand",
                        "controller",
                        "seed",
                        "seconds",
                        "native_completed",
                        "substep_gate_passed",
                    )
                }
                for r in report["rows"]
            ]
        )
    )


if __name__ == "__main__":
    main()
