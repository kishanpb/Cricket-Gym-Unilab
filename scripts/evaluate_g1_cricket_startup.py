"""Native settling and first-step diagnostics, not a running delivery."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from audit_g1_cricket_running_stance import foot_loads
from g1_cricket_delivery_trial import capsule_bounds
from retarget_g1_cricket_running import ROBOT, ROOT, render_poses, running_control

from unilab.tasks.manipulation.g1_cricket.contact_control import ContactAccelerationControl
from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running_startup import startup_reference

COLUMNS = [
    "time_s",
    "left_load_n",
    "right_load_n",
    "pelvis_z_m",
    "pelvis_up",
    "joint_limit_excess_rad",
    "motor_fraction",
    "root_speed_m_s",
    "root_angular_speed_rad_s",
    "maximum_joint_speed_rad_s",
    "holder_force_n",
    "holder_error_m",
    "maximum_foot_displacement_m",
    "com_x_m",
    "com_y_m",
    "com_z_m",
]
STEP_COLUMNS = COLUMNS + [
    "stance_foot_displacement_m",
    "swing_foot_clearance_m",
    "swing_foot_forward_m",
    "swing_foot_lateral_m",
    "foot_target_error_m",
]


def replay_startup(
    model,
    reference,
    hand,
    *,
    first_step=False,
    support_only_balance=False,
    controller=None,
):
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = reference["qvel"][0]
    mujoco.mj_forward(model, data)
    joints = [model.joint(name).id for name in SDK_JOINTS]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    feet_start = data.xpos[feet].copy()
    front, swing = (0, 1) if hand == "right" else (1, 0)
    swing_geoms = np.flatnonzero(
        (model.geom_bodyid == feet[swing])
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    airborne_since, longest_airborne, landing_time = None, 0.0, None
    pelvis = model.body("pelvis").id
    wrist = model.body(f"{hand}_wrist_yaw_link").id
    ball = model.body("cricket_ball").id
    holder = model.equality("ball_holder").id
    offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
    sensor = model.sensor("holder_force")
    force_address = int(sensor.adr[0])
    steps, poses, velocities = [], [data.qpos.copy()], [data.qvel.copy()]
    unexpected = set()
    first_limit = None
    peak_penetration = 0.0
    force = np.empty(6)
    substeps = round(0.02 / model.opt.timestep)
    for tick in range(len(reference["times"]) - 1):
        control, correction = running_control(
            model,
            data,
            reference["qpos"][tick],
            reference["qvel"][tick],
            qa,
            va,
        )
        if support_only_balance and reference["support_loads"][tick, swing] == 0:
            control[6 * swing + 4] -= correction[1]
            control[6 * swing + 5] -= correction[0]
        control += (reference["torque"][tick] - data.qfrc_bias[va]) / model.actuator_gainprm[:, 0]
        if controller is not None:
            control = controller(data, reference["qpos"][tick], reference["qvel"][tick])
        data.ctrl[:] = np.clip(control, limits[:, 0], limits[:, 1])
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            excess = np.maximum(limits[:, 0] - data.qpos[qa], data.qpos[qa] - limits[:, 1])
            if first_limit is None and excess.max() > 1e-6:
                index = int(excess.argmax())
                first_limit = {"time_s": float(data.time), "joint": SDK_JOINTS[index]}
            for index, contact in enumerate(data.contact):
                names = {model.geom(int(g)).name for g in contact.geom}
                if "ball_geom" in names:
                    peak_penetration = max(peak_penetration, -float(contact.dist))
                if "pitch" in names and any("foot" in name for name in names):
                    continue
                mujoco.mj_contactForce(model, data, index, force)
                if force[0] > 0.1:
                    unexpected.add("/".join(sorted(names)))
            held_position = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ offset
            loads = foot_loads(model, data)
            row = np.r_[
                data.time,
                loads,
                data.qpos[2],
                data.xmat[pelvis, 8],
                max(0, excess.max()),
                np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1]),
                np.linalg.norm(data.qvel[:3]),
                np.linalg.norm(data.qvel[3:6]),
                np.max(np.abs(data.qvel[va])),
                np.linalg.norm(data.sensordata[force_address : force_address + 3]),
                np.linalg.norm(data.xpos[ball] - held_position),
                np.max(np.linalg.norm(data.xpos[feet] - feet_start, axis=1)),
                data.subtree_com[0],
            ]
            if first_step:
                bounds = capsule_bounds(
                    data.geom_xpos[swing_geoms],
                    data.geom_xmat[swing_geoms].reshape(-1, 3, 3),
                    model.geom_size[swing_geoms],
                    model.geom_type[swing_geoms],
                )
                clearance = float(bounds[0, 2])
                if clearance > 0.002 and loads[swing] < 1:
                    if airborne_since is None:
                        airborne_since = data.time - model.opt.timestep
                    longest_airborne = max(longest_airborne, data.time - airborne_since)
                elif loads[swing] > 1:
                    if longest_airborne >= 0.02 and landing_time is None:
                        landing_time = float(data.time)
                    airborne_since = None
                else:
                    airborne_since = None
                row = np.r_[
                    row,
                    np.linalg.norm(data.xpos[feet[front]] - feet_start[front]),
                    clearance,
                    (data.xpos[feet[swing]] - feet_start[swing])[:2],
                    np.max(
                        np.linalg.norm(data.xpos[feet] - reference["foot_targets"][tick], axis=1)
                    ),
                ]
            steps.append(row)
            if data.warning.number.any() or not np.isfinite(data.qpos).all():
                raise RuntimeError("invalid native startup physics")
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        if data.qpos[2] < 0.48:
            break
    steps = np.asarray(steps)
    values = dict(zip(STEP_COLUMNS if first_step else COLUMNS, steps.T, strict=True))
    tail = steps[steps[:, 0] >= reference["times"][-1] - 0.5 - 1e-9]
    pre = steps[(steps[:, 0] >= 1.5) & (steps[:, 0] < 2)]
    front_column = 1 if hand == "right" else 2
    weight = float(-model.body_mass.sum() * model.opt.gravity[2])
    transfer = (
        float((1 if hand == "right" else -1) * (tail[:, 14].mean() - pre[:, 14].mean()))
        if len(tail) and len(pre)
        else None
    )
    front_fraction = (
        float(tail[:, front_column].mean() / tail[:, 1:3].sum(axis=1).mean()) if len(tail) else None
    )
    failures = []
    for label, failed in (
        ("incomplete", len(poses) != len(reference["times"])),
        ("pelvis_height", values["pelvis_z_m"].min() < 0.48),
        ("pelvis_orientation", values["pelvis_up"].min() < 0.95),
        ("joint_limit", first_limit is not None),
        ("actuator_limit", values["motor_fraction"].max() > 1 + 1e-6),
        ("unexpected_contact", bool(unexpected)),
        ("holder_error", values["holder_error_m"].max() > 0.001),
        (
            "foot_displacement",
            not first_step and values["maximum_foot_displacement_m"].max() > 0.005,
        ),
        ("ball_penetration", peak_penetration > 0.006),
        ("not_settled", not len(tail) or np.max(tail[:, 7:10]) > 0.02),
        ("initial_hold_not_settled", not len(pre) or np.max(pre[:, 7:10]) > 0.02),
        ("missing_two_foot_support", not len(tail) or tail[:, 1:3].min() <= 1),
        (
            "weight_not_supported",
            not len(tail) or abs(tail[:, 1:3].sum(axis=1).mean() / weight - 1) > 0.01,
        ),
        (
            "load_transfer_not_achieved",
            not first_step and (front_fraction is None or front_fraction < 0.75),
        ),
        (
            "com_shift_not_achieved",
            not first_step and (transfer is None or abs(transfer - 0.08) > 0.01),
        ),
    ):
        if failed:
            failures.append(label)
    if first_step:
        for label, failed in (
            ("no_airborne_step", longest_airborne < 0.02),
            ("no_landing", landing_time is None),
            ("stance_foot_displacement", values["stance_foot_displacement_m"].max() > 0.005),
            ("foot_target_error", values["foot_target_error_m"].max() > 0.02),
            ("step_distance", not len(tail) or abs(tail[:, 18].mean() - 0.16) > 0.01),
            ("step_lateral_drift", not len(tail) or abs(tail[:, 19].mean()) > 0.005),
        ):
            if failed:
                failures.append(label)
    summary = dict(
        hand=hand,
        physics_dt_s=model.opt.timestep,
        passed=not failures,
        failures=failures,
        duration_s=float(data.time),
        initial_generalized_velocity=reference["qvel"][0].tolist(),
        first_joint_limit=first_limit,
        unexpected_contacts=sorted(unexpected),
        released=not bool(data.eq_active[holder]),
        peak_ball_penetration_m=peak_penetration,
        system_weight_n=weight,
        measured_com_shift_m=transfer,
        front_load_fraction=front_fraction,
        trace_minimum=steps.min(axis=0).tolist(),
        trace_maximum=steps.max(axis=0).tolist(),
        preshift_mean=pre.mean(axis=0).tolist() if len(pre) else None,
        final_half_second_mean=tail.mean(axis=0).tolist() if len(tail) else None,
        final_half_second_maximum=tail.max(axis=0).tolist() if len(tail) else None,
    )
    if first_step:
        summary.update(longest_airborne_s=longest_airborne, landing_time_s=landing_time)
    return summary, dict(steps=steps, qpos=np.asarray(poses), qvel=np.asarray(velocities))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--first-step", action="store_true")
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["pd"],
        choices=["pd", "support_only", "contact_acceleration"],
    )
    args = parser.parse_args()
    if len(set(args.controllers)) != len(args.controllers):
        parser.error("controller names must be distinct")
    if not args.first_step and args.controllers != ["pd"]:
        parser.error("controller comparisons require --first-step")
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__),
        ROBOT,
        ROBOT.parent / "scene_flat.xml",
        ROOT / "scripts/audit_g1_cricket_running_stance.py",
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows = []
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-startup-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
        reference = startup_reference(model, hand, first_step=args.first_step)
        np.savez_compressed(args.output / f"{hand}_reference.npz", **reference)
        for dt in (0.000125, 0.0000625):
            model.opt.timestep = dt
            for kind in args.controllers:
                controller = None
                if kind == "contact_acceleration":

                    def baseline(model, data, target, velocity, qa, va):
                        control, correction = running_control(model, data, target, velocity, qa, va)
                        frame = int((data.time + 1e-9) / 0.02)
                        control += (
                            reference["torque"][frame] - data.qfrc_bias[va]
                        ) / model.actuator_gainprm[:, 0]
                        return control, correction

                    controller = ContactAccelerationControl(
                        model, reference["qvel"], baseline, foot_reference=reference["qpos"]
                    )
                summary, trace = replay_startup(
                    model,
                    reference,
                    hand,
                    first_step=args.first_step,
                    support_only_balance=kind == "support_only",
                    controller=controller,
                )
                name = f"{hand}_{dt * 1e6:g}us"
                if args.first_step:
                    name += f"_{kind}"
                summary.update(trace=f"{name}.npz", controller=kind)
                if controller is not None:
                    trace["optimizer_scores"] = np.asarray(
                        [
                            [row["time_s"], row["initial_score"], row["final_score"]]
                            for row in controller.trace
                        ]
                    )
                np.savez_compressed(args.output / summary["trace"], **trace)
                rows.append(summary)
                print(hand, dt, kind, summary["passed"], summary["failures"], flush=True)
                if args.render and dt == 0.0000625:
                    movie = f"{hand}_{kind}.mp4" if args.first_step else f"{hand}.mp4"
                    render_poses(
                        model,
                        trace["qpos"],
                        hand,
                        args.output / movie,
                        f"FIRST STEP {kind}, NOT RUNNING OR PPO"
                        if args.first_step
                        else "PHYSICAL STARTUP ONLY, NOT RUNNING OR PPO",
                    )
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("startup inputs changed")
    report = dict(
        scope="native_first_step_only"
        if args.first_step
        else "native_two_foot_settling_and_lateral_transfer_only",
        mujoco_version=mujoco.__version__,
        input_sha256=hashes,
        control_period_s=0.02,
        first_step=args.first_step,
        controllers=args.controllers,
        com_lateral_shift_m=None if args.first_step else 0.08,
        com_lateral_target="stance ankle" if args.first_step else "80 mm from neutral",
        step_distance_m=0.16 if args.first_step else 0,
        foot_lift_m=0.05 if args.first_step else 0,
        com_lowering_m=0.03 if args.first_step else 0,
        settle_duration_s=2,
        transfer_duration_s=2 if args.first_step else 1.5,
        final_hold_duration_s=2,
        swing_start_s=4.5 if args.first_step else None,
        swing_end_s=6.5 if args.first_step else None,
        recenter_end_s=8.5 if args.first_step else None,
        columns=STEP_COLUMNS if args.first_step else COLUMNS,
        rows=rows,
        sample_timing="Each row has integrated end time, qpos and qvel; contacts, force sensors and kinematics are the solved frame at the start of that substep.",
        guard="Static foot loads are a feedforward hypothesis, not contact certification. Native replay retains the finite-compliance mechanical ball holder, original geometry, joint limits, actuator caps and all substeps. Initial placement is reset-only; no live root or joint writes, applied support forces, release, run-up, learned policy, delivery qualification or showcase promotion.",
    )
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
