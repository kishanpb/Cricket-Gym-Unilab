"""Compare fixed running PD with/without ankle balance, retaining substep loads."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from retarget_g1_cricket_running import ROBOT, ROOT, running_control

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running import RELEASE_TIME


def foot_loads(model, data):
    loads = np.zeros(2)
    wrench = np.empty(6)
    for index, contact in enumerate(data.contact):
        names = [model.geom(int(g)).name for g in contact.geom]
        if "pitch" not in names:
            continue
        for side, hand in enumerate(("left", "right")):
            if any(name.startswith(hand) and "foot" in name for name in names):
                mujoco.mj_contactForce(model, data, index, wrench)
                loads[side] += max(0.0, wrench[0])
    return loads


def replay(model, reference, gain, *, controller=None):
    times, poses, velocity = reference["times"], reference["qpos"], reference["qvel"]
    np.testing.assert_allclose(np.diff(times), 0.02, atol=1e-15)
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    limits = model.jnt_range[joints]
    ankle = np.array([4, 10])
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = poses[0], velocity[0]
    mujoco.mj_forward(model, data)
    holder = model.equality("ball_holder").id
    rows, physical = [], [data.qpos.copy()]
    first_limit = None
    for tick in range(len(times) - 1):
        control, correction = running_control(
            model, data, poses[tick], velocity[tick], qa, va, balance_gain=gain
        )
        if times[tick] >= RELEASE_TIME:
            data.eq_active[holder] = False
        if controller is not None:
            control, correction = controller(data, poses[tick], velocity[tick]), np.zeros(2)
        data.ctrl[:] = np.clip(control, limits[:, 0], limits[:, 1])
        min_height = float(data.qpos[2])
        for _ in range(round(0.02 / model.opt.timestep)):
            start_time = data.time
            q, v = data.qpos[qa[ankle]].copy(), data.qvel[va[ankle]].copy()
            mujoco.mj_step(model, data)
            excess = np.maximum(limits[:, 0] - data.qpos[qa], data.qpos[qa] - limits[:, 1])
            if first_limit is None and excess.max() > 1e-6:
                index = int(excess.argmax())
                first_limit = {
                    "time_s": float(data.time),
                    "joint": SDK_JOINTS[index],
                    "q_rad": float(data.qpos[qa[index]]),
                    "reference_rad": float(poses[tick, qa[index]]),
                    "command_rad": float(data.ctrl[index]),
                    "limit_rad": limits[index].tolist(),
                }
            rows.append(
                np.r_[
                    start_time,
                    data.time,
                    q,
                    v,
                    poses[tick, qa[ankle]],
                    control[ankle],
                    data.ctrl[ankle],
                    correction[1],
                    data.actuator_force[ankle],
                    data.qfrc_bias[va[ankle]],
                    data.qfrc_constraint[va[ankle]],
                    data.qacc[va[ankle]],
                    foot_loads(model, data),
                    data.qpos[qa[ankle]],
                    max(0.0, excess.max()),
                    np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1]),
                    data.qpos[2],
                ]
            )
            min_height = min(min_height, float(data.qpos[2]))
            if (
                data.warning.number.any()
                or not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                raise RuntimeError("invalid stance replay")
        physical.append(data.qpos.copy())
        if min_height < 0.48:
            break
    return (
        {
            "balance_gain": gain,
            "duration_s": float(data.time),
            "first_joint_limit": first_limit,
            "released": not bool(data.eq_active[holder]),
            "completed_horizon": len(physical) == len(poses) and min_height >= 0.48,
        },
        np.asarray(rows),
        np.asarray(physical),
    )


COLUMNS = [
    "step_start_s",
    "step_end_s",
    "left_q_start",
    "right_q_start",
    "left_velocity_start",
    "right_velocity_start",
    "left_reference",
    "right_reference",
    "left_raw_command",
    "right_raw_command",
    "left_command",
    "right_command",
    "balance_pitch_correction",
    "left_motor_torque",
    "right_motor_torque",
    "left_bias_torque",
    "right_bias_torque",
    "left_constraint_torque",
    "right_constraint_torque",
    "left_acceleration",
    "right_acceleration",
    "left_foot_normal_load",
    "right_foot_normal_load",
    "left_q_end",
    "right_q_end",
    "maximum_joint_limit_excess",
    "motor_force_fraction",
    "pelvis_height",
]


def render_stance_review(output, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for column, hand in enumerate(("right", "left")):
        side = "left" if hand == "right" else "right"
        for result in (row for row in results if row["hand"] == hand):
            with np.load(output / result["trace"]) as saved:
                steps = saved["steps"]
            first_stance = steps[steps[:, 1] <= 0.22]
            values = dict(zip(COLUMNS, first_stance.T, strict=True))
            label = f"balance gain {result['balance_gain']:g}"
            axes[0, column].plot(values["step_end_s"], values[f"{side}_q_end"], label=label)
            axes[1, column].plot(
                values["step_start_s"], values[f"{side}_motor_torque"], label=label
            )
            axes[2, column].plot(
                values["step_start_s"], values[f"{side}_foot_normal_load"], label=label
            )
            if result["balance_gain"] == 4:
                axes[0, column].plot(
                    values["step_start_s"], values[f"{side}_reference"], "--", label="reference"
                )
                axes[0, column].plot(
                    values["step_start_s"], values[f"{side}_command"], ":", label="motor command"
                )
                axes[1, column].plot(
                    values["step_start_s"],
                    values[f"{side}_constraint_torque"],
                    "--",
                    label="constraint torque, gain 4",
                )
                axes[0, column].axhline(
                    result["first_joint_limit"]["limit_rad"][0],
                    color="black",
                    linewidth=1,
                    label="original joint stop",
                )
        axes[0, column].set_title(f"{hand}-hand delivery: {side} stance ankle")
        axes[2, column].set_xlabel("Simulation time (s)")
    for row, units in enumerate(
        ("Ankle pitch (rad)", "Generalized torque (Nm)", "Foot normal load (N)")
    ):
        for axis in axes[row]:
            axis.set_ylabel(units)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    fig.suptitle("Actual native PD replay: first stance, neither controller completes delivery")
    fig.tight_layout()
    fig.savefig(output / "first_stance_review.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__).resolve(),
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROBOT,
        ROBOT.parent / "scene_flat.xml",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [(args.reference / f"{hand}_reference.npz").resolve() for hand in ("right", "left")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    results = []
    with TemporaryDirectory(prefix="g1-stance-") as temporary:
        for hand in ("right", "left"):
            scene = Path(temporary) / f"{hand}.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.opt.timestep = 0.0000625
            source = args.reference / f"{hand}_reference.npz"
            with np.load(source) as reference:
                for gain in (4.0, 0.0):
                    summary, steps, poses = replay(model, reference, gain)
                    summary["hand"] = hand
                    summary["max_joint_limit_excess_rad"] = float(steps[:, -3].max())
                    summary["peak_motor_fraction"] = float(steps[:, -2].max())
                    name = f"{hand}_balance_{gain:g}"
                    np.savez_compressed(args.output / f"{name}.npz", steps=steps, qpos=poses)
                    summary["trace"] = f"{name}.npz"
                    results.append(summary)
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("reference inputs changed during evaluation")
    result = {
        "mujoco_version": mujoco.__version__,
        "scope": "Fixed full-reference PD comparison, not learned bowling or promotion evidence.",
        "sampling": "62.5us native steps. Force, acceleration and contact caches belong to step start; q_end is after integration. Commands are held for 20ms.",
        "columns": COLUMNS,
        "units": "SI: seconds, radians, rad/s, rad/s2, Nm, N and meters; motor_force_fraction is dimensionless.",
        "input_sha256": hashes,
        "results": results,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    render_stance_review(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
