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
from unilab.tasks.manipulation.g1_cricket.running import (
    RELEASE_TIME,
    BallisticRunupCOM,
    retarget_running_delivery,
)
from unilab.tasks.manipulation.g1_cricket.running_ground_momentum import RunningGroundMomentum
from unilab.tasks.manipulation.g1_cricket.running_momentum import HeldBallMomentum
from unilab.tasks.manipulation.g1_cricket.running_support import LateralSupportCOM
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


def running_control(model, data, target, velocity, qa, va, *, balance_gain=4.0):
    kp = model.actuator_gainprm[:, 0]
    control = target[qa] - model.actuator_biasprm[:, 2] / kp * velocity[va]
    control += data.qfrc_bias[va] / kp
    correction = ankle_balance(target[3:7], data.qpos[3:7], data.qvel[3:6], balance_gain)
    correction += root_position_balance(
        target[3:7], data.qpos[:3] - target[:3], data.qvel[:3] - velocity[:3], balance_gain
    )
    correction = np.clip(correction, -0.3, 0.3)
    control[[4, 10]] += correction[1]
    control[[5, 11]] += correction[0]
    control[14] += target[qa[14]] - data.qpos[qa[14]]
    return control, correction


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


def render_review(output):
    sheet = Image.new("RGB", (1440, 4 * 161), "#eef1ef")
    draw = ImageDraw.Draw(sheet)
    for row, (kind, hand) in enumerate(
        (kind, hand) for kind in ("offline_targets", "pd_diagnostic") for hand in ("right", "left")
    ):
        with imageio.get_reader(output / f"{hand}_{kind}.mp4") as reader:
            count = reader.count_frames()
            frames = (
                [0, 30, 60, 83, 91, 135]
                if kind == "offline_targets"
                else np.linspace(0, count - 1, 6, dtype=int)
            )
            for column, frame in enumerate(frames):
                tile = Image.fromarray(reader.get_data(int(frame))).resize((240, 135))
                sheet.paste(tile, (column * 240, row * 161 + 26))
                draw.text(
                    (column * 240 + 4, row * 161 + 6),
                    f"{hand} {kind} | {frame * 0.02:.2f}s",
                    fill="#18251d",
                    font=ImageFont.load_default(size=12),
                )
    sheet.save(output / "running_motion_review.png")


def run(
    hand,
    output,
    render,
    *,
    ballistic_parent=None,
    lane_offset=0.0,
    conserve_momentum=False,
    lateral_support=False,
    ground_momentum_parent=None,
    retarget_substeps=1,
):
    with TemporaryDirectory(prefix="g1-running-") as temporary:
        scene = Path(temporary) / "scene.xml"
        G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.0000625
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
        times = np.arange(136) * 0.02
        com_target = None
        if ballistic_parent is not None:
            with np.load(ballistic_parent / f"{hand}_reference.npz") as parent:
                np.testing.assert_array_equal(parent["times"], times)
                parent_poses = parent["qpos"]
            data = mujoco.MjData(model)
            centers = []
            for pose in parent_poses:
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                centers.append(data.subtree_com[0].copy())
            centers = np.asarray(centers)
            centers[:, 1] += lane_offset * (1 if hand == "right" else -1)
            com_target = BallisticRunupCOM(times, centers, -model.opt.gravity[2])
            if lateral_support:
                lane = (1 if hand == "right" else -1) * (0.5 + lane_offset)
                com_target = LateralSupportCOM(com_target, hand, lane)
        momentum_target = None
        if ground_momentum_parent is not None:
            with np.load(ground_momentum_parent / f"{hand}_reference.npz") as parent:
                np.testing.assert_array_equal(parent["times"], times)
                parent_poses = parent["qpos"]
            helper = HeldBallMomentum(model, hand)
            mean_momentum = np.mean(
                [helper.measure(parent_poses[i], parent_poses[i + 1], 0.02) for i in range(30)],
                axis=0,
            )
            momentum_target = RunningGroundMomentum(
                com_target, model.body_mass.sum(), mean_momentum
            )
        reference = retarget_running_delivery(
            model,
            np.arange(135 * retarget_substeps + 1) * (0.02 / retarget_substeps),
            hand,
            com_target=com_target,
            lane_offset=lane_offset,
            conserve_momentum=conserve_momentum,
            momentum_target=momentum_target,
        )
        dense_velocity = velocity_reference(model, reference["qpos"], 0.02 / retarget_substeps)
        poses = reference["qpos"][::retarget_substeps]
        velocity = dense_velocity[::retarget_substeps]
        if retarget_substeps > 1:
            np.savez_compressed(
                output / f"{hand}_dense_reference.npz",
                times=reference["times"],
                qpos=reference["qpos"],
                qvel=dense_velocity,
            )
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
        holder = model.equality("ball_holder").id
        ball = model.joint("ball_free")
        bq, bv = int(ball.qposadr[0]), int(ball.dofadr[0])
        release, trace, physical_poses = None, [], [data.qpos.copy()]
        wrench = np.zeros(6)
        for tick in range(135):
            target = poses[tick]
            control, _ = running_control(model, data, target, velocity[tick], qa, va)
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
            "runup_momentum_target": {
                "cycle_mean_nms": mean_momentum.tolist(),
                "initial_nms": momentum_target.initial.tolist(),
                "vertical_force_n": float(momentum_target.force_z),
            }
            if momentum_target is not None
            else None,
            "kinematic_errors": reference["errors"],
            "reference_forward_travel_m": float(poses[-1, 0] - poses[0, 0]),
            "reference_peak_joint_speed_rad_s": {
                name: float(np.abs(dense_velocity[:, dof]).max())
                for name, dof in zip(SDK_JOINTS, va, strict=True)
            },
            "completed_physical_motion": len(trace) == 135
            and trace[-1]["minimum_substep_height_m"] >= 0.48,
            "release": release,
            "trace": trace,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ballistic-parent", type=Path)
    parser.add_argument("--lane-offset", type=float, default=0.0)
    parser.add_argument("--conserve-momentum", action="store_true")
    parser.add_argument("--lateral-support", action="store_true")
    parser.add_argument("--ground-momentum-parent", type=Path)
    parser.add_argument("--retarget-substeps", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    if args.lateral_support and args.ballistic_parent is None:
        parser.error("lateral support requires a ballistic parent")
    if args.ground_momentum_parent is not None and not (
        args.lateral_support and args.conserve_momentum
    ):
        parser.error("ground momentum requires lateral support and momentum-conserving rotation")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(__file__), ROBOT, ROBOT.parent / "scene_flat.xml"]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    if args.ballistic_parent is not None:
        inputs += [args.ballistic_parent / f"{hand}_reference.npz" for hand in ("right", "left")]
    if args.ground_momentum_parent is not None:
        inputs += [
            args.ground_momentum_parent / f"{hand}_reference.npz" for hand in ("right", "left")
        ]
    inputs = [path.resolve() for path in inputs]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows = [
        run(
            hand,
            args.output,
            args.render,
            ballistic_parent=args.ballistic_parent,
            lane_offset=args.lane_offset,
            conserve_momentum=args.conserve_momentum,
            lateral_support=args.lateral_support,
            ground_momentum_parent=args.ground_momentum_parent,
            retarget_substeps=args.retarget_substeps,
        )
        for hand in ("right", "left")
    ]
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("reference inputs changed during evaluation")
    result = {
        "scope": "offline_running_reference_and_PD_feasibility_not_learned_bowling",
        "mujoco_version": mujoco.__version__,
        "ik_direction": "forward",
        "ballistic_runup_com": args.ballistic_parent is not None,
        "momentum_conserving_runup": args.conserve_momentum,
        "lateral_support_com": args.lateral_support,
        "stance_ground_momentum": args.ground_momentum_parent is not None,
        "retarget_substeps": args.retarget_substeps,
        "retarget_period_s": 0.02 / args.retarget_substeps,
        "physical_control_period_s": 0.02,
        "outward_lane_offset_m": args.lane_offset,
        "input_sha256": hashes,
        "rows": rows,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if args.render:
        render_review(args.output)
    print(
        [{k: v for k, v in row.items() if k not in {"trace", "kinematic_errors"}} for row in rows]
    )
