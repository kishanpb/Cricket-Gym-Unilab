"""Retarget both-arm cricket choreography, then test actual floating-base PD motion."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from unilab.tasks.manipulation.g1_cricket.bimanual import build_bimanual_scene, retarget_batting
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.tracking import export_reference

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "src/unilab/assets/robots/g1/g1.xml"


def source_hashes():
    paths = [Path(__file__), ROBOT, ROBOT.parent / "scene_flat.xml"]
    paths += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def grip_force(data, eq_id):
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq_id)
    if np.count_nonzero(rows) != 3:
        raise RuntimeError("active second-hand connect must have three constraint rows")
    return float(np.linalg.norm(data.efc_force[rows]))


def run(hand, output, render):
    with TemporaryDirectory(prefix="g1-bimanual-") as temporary:
        scene = Path(temporary) / "scene.xml"
        build_bimanual_scene(ROBOT, scene, hand)
        model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.00025
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        model.vis.global_.offwidth = 960
        model.vis.global_.offheight = 540
        times = np.linspace(0, 3, 151)
        reference = retarget_batting(model, times, hand)
        data = mujoco.MjData(model)
        data.qpos[:] = reference["qpos"][0]
        # Ball is out of play: first establish a two-hand swing, not a hit claim.
        ball_address = model.jnt_qposadr[model.joint("ball_free").id]
        data.qpos[ball_address : ball_address + 3] = [8, 0, 0.036]
        mujoco.mj_forward(model, data)
        ids = np.array([model.joint(name).id for name in SDK_JOINTS])
        addresses = model.jnt_qposadr[ids]
        dofs = model.jnt_dofadr[ids]
        limits = model.jnt_range[ids]
        eq_id = model.equality("second_hand_grip").id
        force = np.empty(6)
        rows, frames, snapshots, velocities, controls = [], [], [], [], []
        root_rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True)
        renderer = mujoco.Renderer(model, height=540, width=960) if render else None
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.1, 0, 0.75]
        camera.distance, camera.azimuth, camera.elevation = 2.7, -65, -12
        try:
            for index, target in enumerate(reference["qpos"][:-1]):
                peak_grip_force = 0.0
                peak_torque_fraction = 0.0
                collisions = set()
                excess = 0.0
                minimum_height = float(data.qpos[2])
                maximum_grip_error = 0.0
                control_excess = 0.0
                for _ in range(80):
                    data.ctrl[:] = (
                        target[addresses] + data.qfrc_bias[dofs] / model.actuator_gainprm[:, 0]
                    )
                    tilt = (
                        root_rotation.inv() * Rotation.from_quat(data.qpos[3:7], scalar_first=True)
                    ).as_rotvec()
                    correction = np.clip(0.7 * tilt[:2] + 0.1 * data.qvel[3:5], -0.18, 0.18)
                    data.ctrl[[4, 10]] += correction[1]
                    data.ctrl[[5, 11]] += correction[0]
                    control_excess = max(
                        control_excess,
                        float(np.maximum(limits[:, 0] - data.ctrl, data.ctrl - limits[:, 1]).max()),
                    )
                    mujoco.mj_step(model, data)
                    peak_grip_force = max(peak_grip_force, grip_force(data, eq_id))
                    minimum_height = min(minimum_height, float(data.qpos[2]))
                    maximum_grip_error = max(
                        maximum_grip_error,
                        float(
                            np.linalg.norm(
                                data.site("bat_lower_grip").xpos - data.site(f"{hand}_palm").xpos
                            )
                        ),
                    )
                    peak_torque_fraction = max(
                        peak_torque_fraction,
                        float(
                            np.max(np.abs(data.actuator_force) / model.actuator_forcerange[:, 1])
                        ),
                    )
                    q = data.qpos[addresses]
                    excess = max(
                        excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max())
                    )
                    for contact_index, contact in enumerate(data.contact):
                        if contact.dist >= 0:
                            continue
                        names = {model.geom(int(g)).name for g in contact.geom}
                        if "pitch" in names and any("foot" in name for name in names):
                            continue
                        if names == {"ball_geom", "pitch"}:
                            continue
                        mujoco.mj_contactForce(model, data, contact_index, force)
                        if force[0] > 0.1:
                            collisions.add("/".join(sorted(names)))
                mujoco.mj_forward(model, data)
                separation = np.linalg.norm(
                    data.site("bat_lower_grip").xpos - data.site(f"{hand}_palm").xpos
                )
                rows.append(
                    {
                        "time_s": float(data.time),
                        "pelvis_height_m": float(data.qpos[2]),
                        "minimum_substep_pelvis_height_m": minimum_height,
                        "second_grip_error_m": float(separation),
                        "maximum_solve_grip_error_m": maximum_grip_error,
                        "second_grip_peak_constraint_force_n": peak_grip_force,
                        "maximum_motor_fraction": peak_torque_fraction,
                        "joint_limit_excess_rad": excess,
                        "control_target_limit_excursion_rad": control_excess,
                        "unexpected_loaded_contacts": sorted(collisions),
                        "bat_center_m": data.site("bat_center").xpos.tolist(),
                    }
                )
                snapshots.append(data.qpos.copy())
                velocities.append(data.qvel.copy())
                controls.append(data.ctrl.copy())
                if renderer is not None:
                    renderer.update_scene(data, camera)
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 960, 64), fill="#17201d")
                    font = ImageFont.load_default(size=18)
                    draw.text(
                        (12, 8),
                        f"G1 {hand}-hand batting | two-hand mechanical grip | t={data.time:.2f}s",
                        font=font,
                    )
                    draw.text(
                        (12, 34),
                        "PD + gravity/ankle feedback | not RL trained | ball out of play | 0.5x",
                        font=font,
                    )
                    frames.append(np.asarray(frame))
                if data.warning.number.any():
                    raise RuntimeError("MuJoCo warning during bimanual tracking")
        finally:
            if renderer is not None:
                renderer.close()
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / f"{hand}_reference.npz", times=times, qpos=reference["qpos"])
        export_reference(model, reference["qpos"], 50, output / f"{hand}_tracking.npz")
        np.savez_compressed(
            output / f"{hand}_physics.npz",
            qpos=np.asarray(snapshots),
            qvel=np.asarray(velocities),
            final_substep_ctrl=np.asarray(controls),
            times=[row["time_s"] for row in rows],
        )
        if frames:
            imageio.mimwrite(
                output / f"{hand}_bimanual_diagnostic.mp4", frames, fps=25, macro_block_size=1
            )
            sheet = Image.new("RGB", (1440, 540))
            for tile, index in enumerate((0, 30, 57, 72, 90, 140)):
                sheet.paste(
                    Image.fromarray(frames[index]).resize((480, 270)),
                    ((tile % 3) * 480, (tile // 3) * 270),
                )
            sheet.save(output / f"{hand}_contact_sheet.png")
        return {
            "hand": hand,
            "scope": "physics_PD_reference_diagnostic_not_learned_cricket",
            "reference_errors": reference["errors"],
            "trace": rows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    (args.output / "preflight.json").write_text(
        json.dumps(
            {
                "source_sha256": hashes,
                "hands": ["right", "left"],
                "scope": "dry_swing_reference_PD_not_RL_or_batting_success",
            },
            indent=2,
        )
        + "\n"
    )
    reports = [run(hand, args.output, args.render) for hand in ("right", "left")]
    if source_hashes() != hashes:
        raise RuntimeError("retargeting sources changed during execution")
    report = {
        "rows": reports,
        "source_sha256": hashes,
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "numpy", "scipy", "imageio")
        },
        "engine": "serial_mujoco_PD_diagnostic_no_batch_parity_claim",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    for row in reports:
        print(
            row["hand"],
            {
                "min_pelvis": min(r["pelvis_height_m"] for r in row["trace"]),
                "max_grip_error": max(r["second_grip_error_m"] for r in row["trace"]),
                "contacts": sorted(
                    {c for r in row["trace"] for c in r["unexpected_loaded_contacts"]}
                ),
            },
        )


if __name__ == "__main__":
    main()
