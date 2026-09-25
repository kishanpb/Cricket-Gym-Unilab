"""Audit native motor/root/held-ball requirements of the full fixed run-up."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from audit_g1_cricket_running_rotation import ReferenceCurve
from g1_cricket_delivery_trial import capsule_bounds

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.reference_dynamics import ReferenceDynamics, curve_state
from unilab.tasks.manipulation.g1_cricket.running_support import (
    ForeAftSupportCOM,
    LateralSupportCOM,
)

ROOT = Path(__file__).resolve().parents[1]


def audit(directory):
    output = directory / "inverse_dynamics_audit.json"
    if output.exists() or output.with_suffix(".json.gz").exists():
        raise FileExistsError(output)
    source = json.loads((directory / "evaluation.json").read_text())
    suffix = "dense_reference.npz" if source.get("retarget_substeps", 1) > 1 else "reference.npz"
    inputs = [
        Path(__file__),
        ROOT / "scripts/audit_g1_cricket_running_rotation.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [
        ROOT / f"src/unilab/assets/robots/g1/{name}" for name in ("g1.xml", "scene_flat.xml")
    ]
    inputs += [directory / "evaluation.json"]
    inputs += [directory / f"{hand}_{suffix}" for hand in ("right", "left")]
    hashes = {
        str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    rows, summaries, names = [], [], None
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-inverse-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
                ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.opt.timestep = 0.0000625
        with np.load(directory / f"{hand}_{suffix}") as reference:
            curve = ReferenceCurve(model, reference["times"], reference["qpos"], hand)
        if source.get("lateral_support_com", False):
            lane = (1 if hand == "right" else -1) * (0.5 + source["outward_lane_offset_m"])
            support = ForeAftSupportCOM if source.get("fore_aft_support_com") else LateralSupportCOM
            curve.ballistic = support(curve.ballistic, hand, lane)
        helper = ReferenceDynamics(model)
        names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
        ball_v = int(model.joint("ball_free").dofadr[0])
        holder = (model.body(f"{hand}_wrist_yaw_link").id, model.body("cricket_ball").id)
        feet = []
        for side in ("left", "right"):
            body = model.body(f"{side}_ankle_roll_link").id
            geoms = np.flatnonzero(
                (model.geom_bodyid == body)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            feet.append((body, geoms))
        wrench = np.empty(6)
        for tick in range(1, 120):
            time = tick * 0.01
            for step in (0.00125, 0.000625):
                state = curve_state(model, curve, time, step)
                result = helper.evaluate(*state)
                patches = []
                for body, geoms in feet:
                    box = capsule_bounds(
                        helper.data.geom_xpos[geoms],
                        helper.data.geom_xmat[geoms].reshape(-1, 3, 3),
                        model.geom_size[geoms],
                        model.geom_type[geoms],
                    )
                    if box[0, 2] <= 0.002:
                        friction = max(
                            model.geom_friction[geoms, 0].max(),
                            model.geom_friction[model.geom("pitch").id, 0],
                        )
                        patches.append((body, box, float(friction)))
                allocations = {
                    name: helper.support_allocation(state[2], patches, bounds, holder)
                    for name, bounds in (
                        ("unbounded_motors", [(None, None)] * model.nu),
                        ("force_caps", model.actuator_forcerange),
                        ("bounded_commands", result["reachable_motor_force_bounds"]),
                    )
                }
                allocations["minimum_motor_scale"] = helper.support_allocation(
                    state[2], patches, model.actuator_forcerange, holder, minimum_motor_scale=True
                )
                contacts = []
                for index, contact in enumerate(helper.data.contact):
                    mujoco.mj_contactForce(model, helper.data, index, wrench)
                    contacts.append(
                        {
                            "geoms": [model.geom(int(g)).name for g in contact.geom],
                            "distance_m": float(contact.dist),
                            "inverse_wrench_contact_frame": wrench.copy().tolist(),
                        }
                    )
                phase = time % 0.3
                rows.append(
                    {
                        "hand": hand,
                        "time_s": time,
                        "derivative_step_s": step,
                        "crosses_phase_boundary": min(phase, abs(phase - 0.22), 0.3 - phase)
                        < 2 * step,
                        "root_force_norm_n": float(
                            np.linalg.norm(result["unactuated_residual"][:3])
                        ),
                        "root_torque_norm_nm": float(
                            np.linalg.norm(result["unactuated_residual"][3:6])
                        ),
                        "ball_force_norm_n": float(
                            np.linalg.norm(result["unactuated_residual"][ball_v : ball_v + 3])
                        ),
                        "ball_torque_norm_nm": float(
                            np.linalg.norm(result["unactuated_residual"][ball_v + 3 : ball_v + 6])
                        ),
                        "contacts": contacts,
                        "optimistic_support_patches": [
                            {
                                "body": model.body(body).name,
                                "bounds_m": box.tolist(),
                                "friction": mu,
                            }
                            for body, box, mu in patches
                        ],
                        "optimistic_allocations": allocations,
                        **{key: value.tolist() for key, value in result.items()},
                    }
                )
        for step in (0.00125, 0.000625):
            subset = [r for r in rows if r["hand"] == hand and r["derivative_step_s"] == step]
            excess = np.array([r["reachable_force_excess"] for r in subset])
            index = np.unravel_index(excess.argmax(), excess.shape)
            summaries.append(
                {
                    "hand": hand,
                    "derivative_step_s": step,
                    "samples": len(subset),
                    "motor_limit_exceeded_samples": sum(
                        max(r["motor_limit_excess"]) > 1e-6 for r in subset
                    ),
                    "command_reachability_exceeded_samples": int(np.sum(excess.max(axis=1) > 1e-6)),
                    "peak_reachability_excess_nm": float(excess[index]),
                    "peak_reachability_joint": names[index[1]],
                    "peak_reachability_time_s": subset[index[0]]["time_s"],
                    "max_root_force_n": max(r["root_force_norm_n"] for r in subset),
                    "max_root_torque_nm": max(r["root_torque_norm_nm"] for r in subset),
                    "max_ball_force_n": max(r["ball_force_norm_n"] for r in subset),
                    "optimistic_feasible_samples": {
                        name: sum(r["optimistic_allocations"][name]["feasible"] for r in subset)
                        for name in ("unbounded_motors", "force_caps", "bounded_commands")
                    },
                }
            )
            scaled = [
                r for r in subset if r["optimistic_allocations"]["minimum_motor_scale"]["feasible"]
            ]
            if not scaled:
                summaries[-1]["max_minimum_motor_scale"] = None
                continue
            worst = max(
                scaled,
                key=lambda r: r["optimistic_allocations"]["minimum_motor_scale"][
                    "minimum_motor_scale"
                ],
            )
            worst_allocation = worst["optimistic_allocations"]["minimum_motor_scale"]
            fractions = np.abs(worst_allocation["motor_forces"]) / model.actuator_forcerange[:, 1]
            summaries[-1].update(
                max_minimum_motor_scale=worst_allocation["minimum_motor_scale"],
                max_minimum_motor_scale_time_s=worst["time_s"],
                max_minimum_motor_scale_joint=names[int(fractions.argmax())],
            )
    for name, digest in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("inverse audit input changed")
    report = {
        "scope": "offline_native_inverse_dynamics_not_a_physical_rollout",
        "mujoco_version": mujoco.__version__,
        "physics_timestep_s": 0.0000625,
        "guard": "Required forces under the native soft-contact and holder model, not measured telemetry or applied assistance. Nonzero unactuated forces cannot be supplied by joint motors. This checks the fixed interpolated curve, not feasibility of every nearby motion; no locomotion, release or promotion success is claimed.",
        "equations_source": "https://mujoco.readthedocs.io/en/latest/computation/",
        "control_bounds": "Existing running controller joint-range clipping; actuator force caps unchanged.",
        "allocation_guard": "Separate optimistic ideal-contact LP: original mass/bias/passive forces and all 41 DOFs; feet within 2 mm get horizontal outer-box patches, outer-square friction bounds, unrestricted yaw torque, an ideal unbounded six-axis internal holder wrench, and joint friction forces inside the original frictionloss bounds without enforcing opposition to velocity. No joint-stop or forbidden-contact support. Feasibility is only a necessary instantaneous bound, not native soft-contact feasibility, a controller or physical success. Infeasibility applies to this fixed sampled curve, not every nearby motion.",
        "motor_scale_guard": "Minimum motor scale is an optimistic LP diagnostic with uniformly scaled force caps and no position-command bound; no model limit is changed and no resulting forces are applied. Rows infeasible even with unbounded motors have no scale value.",
        "sample_scope": "Every 10 ms from 0.01 through 1.19 s, both hands, both finite-difference spacings, including phase boundaries; holder active throughout run-up.",
        "reference_file_suffix": suffix,
        "motor_joint_order": names,
        "generalized_order": "root world translation, root local rotation, 29 robot joints, ball world translation, ball local rotation",
        "input_sha256": hashes,
        "summaries": summaries,
        "rows": rows,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
