"""Bounded motor-reference feasibility checks, not learned batting evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from unilab.tasks.manipulation.g1_cricket.bimanual import build_bimanual_scene, support_feedforward
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS

ROOT = Path(__file__).resolve().parents[1]


def position_velocity_control(model, position, velocity):
    """Encode Kp(q_ref-q) + Kd(v_ref-v) through the existing position actuator."""
    return position - model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0] * velocity


def ankle_balance(reference_quaternion, quaternion, angular_velocity, gain):
    tilt = np.empty(3)
    mujoco.mju_subQuat(tilt, quaternion, reference_quaternion)
    reference_rotation, rotation = np.empty(9), np.empty(9)
    mujoco.mju_quat2Mat(reference_rotation, reference_quaternion)
    mujoco.mju_quat2Mat(rotation, quaternion)
    velocity = reference_rotation.reshape(3, 3).T @ rotation.reshape(3, 3) @ angular_velocity
    return np.clip(gain * (0.7 * tilt[:2] + 0.1 * velocity[:2]), -0.3, 0.3)


def feasibility_checks(row):
    trace = row["trace"]
    return {
        "complete": row["completed_three_seconds"],
        "pelvis_height": min(t["minimum_substep_pelvis_height_m"] for t in trace) > 0.65,
        "no_unexpected_contact": not any(t["unexpected_contacts"] for t in trace),
        "hard_joint_limits": max(t["joint_limit_excess_rad"] for t in trace) <= 0.0001,
        "grip": max(t["maximum_substep_grip_gap_m"] for t in trace) < 0.006,
        "motor_limits": max(t["motor_fraction_peak"] for t in trace) <= 1,
        "root_tracking": max(t["root_translation_error_m"] for t in trace) < 0.15,
        "joint_tracking": max(t["joint_rmse_rad"] for t in trace) < 0.2,
        "bat_tracking": max(t["bat_tracking_error_m"] for t in trace) < 0.08,
    }


def audit(
    hand,
    reference_path,
    motion,
    controller,
    balance_gain=0.0,
    clip_motor_target=True,
    states=None,
    waist_compensation=0.0,
):
    with TemporaryDirectory(prefix="g1-tracking-control-") as directory:
        scene = Path(directory) / "scene.xml"
        build_bimanual_scene(ROOT / "src/unilab/assets/robots/g1/g1.xml", scene, hand)
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.001
        data = mujoco.MjData(model)
        with np.load(reference_path) as saved:
            poses = saved["qpos"].copy()
        if not motion:
            poses[:] = poses[0]
        velocity = np.empty((len(poses), model.nv))
        for i in range(len(poses)):
            before, after = max(0, i - 1), min(len(poses) - 1, i + 1)
            mujoco.mj_differentiatePos(
                model, velocity[i], (after - before) * 0.02, poses[before], poses[after]
            )
        data.qpos[:], data.qvel[:] = poses[0], velocity[0]
        ball = model.joint("ball_free")
        data.qpos[ball.qposadr[0] : ball.qposadr[0] + 3] = [8, 0, 0.036]
        data.qvel[ball.dofadr[0] : ball.dofadr[0] + 6] = 0
        mujoco.mj_forward(model, data)
        if states is not None:
            states.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        addresses, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        limits = model.jnt_range[joints]
        feedforward = (
            [support_feedforward(model, pose) for pose in poses[:-1]]
            if controller == "supported_pd"
            else None
        )
        reference_data = mujoco.MjData(model)
        contact_force = np.empty(6)
        trace = []
        for i, pose in enumerate(poses[:-1]):
            target = pose[addresses]
            if controller != "position":
                target = position_velocity_control(model, target, velocity[i, dofs])
            if feedforward is not None:
                target = target + feedforward[i][0] / model.actuator_gainprm[:, 0]
                correction = ankle_balance(pose[3:7], data.qpos[3:7], data.qvel[3:6], balance_gain)
                target[[4, 10]] += correction[1]
                target[[5, 11]] += correction[0]
                tilt = np.empty(3)
                mujoco.mju_subQuat(tilt, data.qpos[3:7], pose[3:7])
                target[[13, 14, 12]] -= waist_compensation * tilt
                if clip_motor_target:
                    target = np.clip(target, limits[:, 0], limits[:, 1])
            data.ctrl[:] = target
            peak_motor_fraction = peak_contact = excess = 0.0
            minimum_height, grip_gap = float(data.qpos[2]), 0.0
            contacts = set()
            for _ in range(20):
                mujoco.mj_step(model, data)
                if (
                    data.warning.number.any()
                    or not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                ):
                    raise RuntimeError("invalid physics state in control audit")
                peak_motor_fraction = max(
                    peak_motor_fraction,
                    float(np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1])),
                )
                q = data.qpos[addresses]
                minimum_height = min(minimum_height, float(data.qpos[2]))
                grip_gap = max(
                    grip_gap,
                    float(
                        np.linalg.norm(
                            data.site("bat_lower_grip").xpos - data.site(f"{hand}_palm").xpos
                        )
                    ),
                )
                excess = max(excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max()))
                for j, contact in enumerate(data.contact):
                    names = {model.geom(int(g)).name for g in contact.geom}
                    if "pitch" in names and (
                        "ball_geom" in names or any("foot" in n for n in names)
                    ):
                        continue
                    mujoco.mj_contactForce(model, data, j, contact_force)
                    if contact_force[0] > 0.1:
                        peak_contact = max(peak_contact, float(contact_force[0]))
                        contacts.add("/".join(sorted(names)))
            mujoco.mj_forward(model, data)
            if states is not None:
                states.append((data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))
            reference_data.qpos[:] = poses[i + 1]
            mujoco.mj_kinematics(model, reference_data)
            trace.append(
                {
                    "time_s": float(data.time),
                    "pelvis_height_m": float(data.qpos[2]),
                    "minimum_substep_pelvis_height_m": minimum_height,
                    "maximum_substep_grip_gap_m": grip_gap,
                    "bat_tracking_error_m": float(
                        np.linalg.norm(
                            data.site("bat_center").xpos - reference_data.site("bat_center").xpos
                        )
                    ),
                    "root_translation_error_m": float(np.linalg.norm(data.qpos[:3] - pose[:3])),
                    "joint_rmse_rad": float(
                        np.sqrt(np.mean((data.qpos[addresses] - pose[addresses]) ** 2))
                    ),
                    "motor_fraction_peak": peak_motor_fraction,
                    "joint_limit_excess_rad": excess,
                    "unexpected_contact_force_peak_n": peak_contact,
                    "unexpected_contacts": sorted(contacts),
                }
            )
            if data.qpos[2] < 0.5:
                break
        return {
            "hand": hand,
            "reference": "swing" if motion else "static_guard",
            "controller": controller,
            "balance_gain": balance_gain,
            "clip_motor_target": clip_motor_target,
            "waist_compensation": waist_compensation,
            "maximum_static_base_force_residual_n": max(
                np.linalg.norm(row[1][:3]) for row in feedforward
            )
            if feedforward
            else None,
            "maximum_static_base_torque_residual_nm": max(
                np.linalg.norm(row[1][3:]) for row in feedforward
            )
            if feedforward
            else None,
            "completed_three_seconds": bool(len(trace) == 150 and data.qpos[2] >= 0.5),
            "peak_initial_joint_velocity_rad_s": float(np.max(np.abs(velocity[0, dofs]))),
            "trace": trace,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--balance-sweep", action="store_true")
    mode.add_argument("--feedforward-sweep", action="store_true")
    mode.add_argument("--waist-sweep", action="store_true")
    parser.add_argument(
        "--reference-dir", type=Path, default=ROOT / "g1_cricket_results/bimanual_v1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT / "src/unilab/assets/robots/g1/g1.xml"]
    sources += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    references = {
        hand: args.reference_dir.resolve() / f"{hand}_reference.npz" for hand in ("right", "left")
    }
    sources += list(references.values())
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if args.balance_sweep:
        rows = [
            audit(hand, reference, True, "supported_pd", gain)
            for hand, reference in references.items()
            for gain in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0)
        ]
    elif args.feedforward_sweep or args.waist_sweep:
        rows = []
        for hand, reference in references.items():
            cases = (
                [(True, gain) for gain in (0.0, 0.5, 1.0, 1.5)]
                if args.waist_sweep
                else [(True, 0.0), (False, 0.0)]
            )
            for clipped, waist in cases:
                states = []
                row = audit(hand, reference, True, "supported_pd", 4.0, clipped, states, waist)
                suffix = (
                    f"waist_{waist:g}"
                    if args.waist_sweep
                    else ("clipped" if clipped else "unclipped")
                )
                row["trajectory_file"] = f"{hand}_{suffix}.npz"
                np.savez_compressed(
                    args.output / row["trajectory_file"],
                    qpos=np.array([s[0] for s in states]),
                    qvel=np.array([s[1] for s in states]),
                    control=np.array([s[2] for s in states]),
                )
                rows.append(row)
    else:
        rows = [
            audit(hand, reference, motion, tracking)
            for hand, reference in references.items()
            for motion in (False, True)
            for tracking in ("position", "position_velocity", "supported_pd")
        ]
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))]
        for p in sources
    ):
        raise RuntimeError("control-audit sources changed")
    if args.balance_sweep or args.feedforward_sweep or args.waist_sweep:
        for row in rows:
            row["feasibility_checks"] = feasibility_checks(row)
    report = {
        "scope": "serial_physics_feasibility_not_RL_or_full_cricket_qualification",
        "balance_sweep": args.balance_sweep,
        "feedforward_sweep": args.feedforward_sweep,
        "waist_sweep": args.waist_sweep,
        "physics_dt_s": 0.001,
        "control_dt_s": 0.02,
        "stop_rule": "pelvis below 0.5 m or 3 seconds; any solver warning/nonfinite aborts",
        "mujoco_version": mujoco.__version__,
        "input_sha256": hashes,
        "rows": rows,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for r in rows:
        print(
            r["hand"],
            r["reference"],
            r["controller"],
            r["balance_gain"],
            r["clip_motor_target"],
            r["waist_compensation"],
            r["trace"][-1]["time_s"],
            r["trace"][-1]["pelvis_height_m"],
            sorted({c for t in r["trace"] for c in t["unexpected_contacts"]}),
            [name for name, passed in r.get("feasibility_checks", {}).items() if not passed],
        )


if __name__ == "__main__":
    main()
