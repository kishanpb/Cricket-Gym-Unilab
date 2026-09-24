"""Retarget both running deliveries and test a fixed physics-only PD baseline."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running import RELEASE_TIME, retarget_running_delivery
from unilab.tasks.manipulation.g1_cricket.tracking import (
    ankle_balance,
    export_reference,
    root_position_balance,
)

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "src/unilab/assets/robots/g1/g1.xml"


def velocity_reference(model, poses, dt):
    velocity = np.zeros((len(poses), model.nv))
    for i in range(len(poses) - 1):
        mujoco.mj_differentiatePos(model, velocity[i], dt, poses[i], poses[i + 1])
    velocity[-1] = velocity[-2]
    return velocity


def render_poses(model, poses, hand, path, label):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 4.2, -90 if hand == "right" else 90, -8
    with mujoco.Renderer(model, height=540, width=960) as renderer:
        with imageio.get_writer(path, fps=25, macro_block_size=1) as writer:
            for index, pose in enumerate(poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0], 0, 0.8]
                renderer.update_scene(data, camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 62), fill="#17201d")
                draw.text(
                    (12, 10),
                    f"{hand} | {label} | frame={index}",
                    font=ImageFont.load_default(size=18),
                )
                draw.text(
                    (12, 35),
                    "Development only | mechanical ball holder | 0.5x | not a learned rollout",
                    font=ImageFont.load_default(size=16),
                )
                writer.append_data(np.asarray(frame))


def run(hand, output, render):
    with TemporaryDirectory(prefix="g1-running-") as temporary:
        scene = Path(temporary) / "scene.xml"
        G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.0000625
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
        times = np.arange(136) * 0.02
        reference = retarget_running_delivery(model, times, hand)
        poses = reference["qpos"]
        velocity = velocity_reference(model, poses, 0.02)
        np.savez_compressed(
            output / f"{hand}_reference.npz", times=times, qpos=poses, qvel=velocity
        )
        export_reference(model, poses, 50, output / f"{hand}_tracking.npz")
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = poses[0], velocity[0]
        mujoco.mj_forward(model, data)
        joints = np.array([model.joint(name).id for name in SDK_JOINTS])
        qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        limits = model.jnt_range[joints]
        kv_over_kp = -model.actuator_biasprm[:, 2] / model.actuator_gainprm[:, 0]
        holder = model.equality("ball_holder").id
        ball = model.joint("ball_free")
        bq, bv = int(ball.qposadr[0]), int(ball.dofadr[0])
        release, trace, physical_poses = None, [], [data.qpos.copy()]
        wrench = np.zeros(6)
        for tick in range(135):
            target = poses[tick]
            control = (
                target[qa]
                + kv_over_kp * velocity[tick, va]
                + data.qfrc_bias[va] / model.actuator_gainprm[:, 0]
            )
            correction = ankle_balance(target[3:7], data.qpos[3:7], data.qvel[3:6], 4)
            correction += root_position_balance(
                target[3:7], data.qpos[:3] - target[:3], data.qvel[:3] - velocity[tick, :3], 4
            )
            correction = np.clip(correction, -0.3, 0.3)
            control[[4, 10]] += correction[1]
            control[[5, 11]] += correction[0]
            control[14] += target[qa[14]] - data.qpos[qa[14]]
            data.ctrl[:] = np.clip(control, limits[:, 0], limits[:, 1])
            if tick * 0.02 >= RELEASE_TIME and release is None:
                release = {
                    "time_s": float(data.time),
                    "position_m": data.qpos[bq : bq + 3].tolist(),
                    "velocity_m_s": data.qvel[bv : bv + 3].tolist(),
                }
                data.eq_active[holder] = False
            peak_force, peak_limit, peak_motor, min_height = 0.0, 0.0, 0.0, float(data.qpos[2])
            unexpected = set()
            for _ in range(320):
                mujoco.mj_step(model, data)
                q = data.qpos[qa]
                peak_limit = max(
                    peak_limit, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max())
                )
                peak_motor = max(
                    peak_motor,
                    float(np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1])),
                )
                min_height = min(min_height, float(data.qpos[2]))
                peak_force = max(
                    peak_force, float(np.linalg.norm(data.sensor("holder_force").data))
                )
                for i, contact in enumerate(data.contact):
                    names = {model.geom(int(g)).name for g in contact.geom}
                    if "pitch" in names and (
                        "ball_geom" in names or any("foot" in n for n in names)
                    ):
                        continue
                    mujoco.mj_contactForce(model, data, i, wrench)
                    if wrench[0] > 0.1:
                        unexpected.add("/".join(sorted(names)))
                if (
                    data.warning.number.any()
                    or not np.isfinite(data.qpos).all()
                    or not np.isfinite(data.qvel).all()
                ):
                    raise RuntimeError("invalid running-delivery physics")
            trace.append(
                {
                    "time_s": float(data.time),
                    "pelvis_position_m": data.qpos[:3].tolist(),
                    "minimum_substep_height_m": min_height,
                    "holder_peak_force_n": peak_force,
                    "joint_limit_excess_rad": peak_limit,
                    "motor_force_fraction": peak_motor,
                    "unexpected_contacts": sorted(unexpected),
                    "root_tracking_error_m": float(
                        np.linalg.norm(data.qpos[:3] - poses[tick + 1, :3])
                    ),
                }
            )
            physical_poses.append(data.qpos.copy())
            if min_height < 0.48:
                break
        np.savez_compressed(output / f"{hand}_physical.npz", qpos=physical_poses)
        if render:
            render_poses(
                model,
                poses,
                hand,
                output / f"{hand}_offline_targets.mp4",
                "OFFLINE IK TARGET, NOT PHYSICS",
            )
            render_poses(
                model,
                physical_poses,
                hand,
                output / f"{hand}_pd_diagnostic.mp4",
                "PHYSICS PD BASELINE, NOT PPO",
            )
        return {
            "hand": hand,
            "kinematic_errors": reference["errors"],
            "reference_forward_travel_m": float(poses[-1, 0] - poses[0, 0]),
            "completed_physical_motion": len(trace) == 135
            and trace[-1]["minimum_substep_height_m"] >= 0.48,
            "release": release,
            "trace": trace,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(__file__), ROBOT, ROBOT.parent / "scene_flat.xml"]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows = [run(hand, args.output, args.render) for hand in ("right", "left")]
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("reference inputs changed during evaluation")
    result = {
        "scope": "offline_running_reference_and_PD_feasibility_not_learned_bowling",
        "mujoco_version": mujoco.__version__,
        "input_sha256": hashes,
        "rows": rows,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        [{k: v for k, v in row.items() if k not in {"trace", "kinematic_errors"}} for row in rows]
    )
