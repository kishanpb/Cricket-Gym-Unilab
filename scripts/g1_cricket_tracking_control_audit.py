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


def audit(hand, reference_path, motion, controller):
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
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        addresses, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        limits = model.jnt_range[joints]
        feedforward = (
            [support_feedforward(model, pose) for pose in poses[:-1]]
            if controller == "supported_pd"
            else None
        )
        contact_force = np.empty(6)
        trace = []
        for i, pose in enumerate(poses[:-1]):
            target = pose[addresses]
            if controller != "position":
                target = position_velocity_control(model, target, velocity[i, dofs])
            if feedforward is not None:
                target = target + feedforward[i][0] / model.actuator_gainprm[:, 0]
                target = np.clip(target, limits[:, 0], limits[:, 1])
            data.ctrl[:] = target
            peak_motor_fraction = peak_contact = excess = 0.0
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
            trace.append(
                {
                    "time_s": float(data.time),
                    "pelvis_height_m": float(data.qpos[2]),
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
    report = {
        "scope": "serial_physics_feasibility_not_RL_or_full_cricket_qualification",
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
            r["trace"][-1]["time_s"],
            r["trace"][-1]["pelvis_height_m"],
            sorted({c for t in r["trace"] for c in t["unexpected_contacts"]}),
        )


if __name__ == "__main__":
    main()
