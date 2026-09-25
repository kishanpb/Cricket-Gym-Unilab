"""Retain full-pool static/dynamic inverse diagnostics; never run a policy."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from unilab.tasks.manipulation.g1_cricket.batting_dynamics import (
    evaluate_modes,
    independent_ball_dofs,
    reference_derivatives,
)
from unilab.tasks.manipulation.g1_cricket.bimanual_contact import G1BimanualContactCfg
from unilab.tasks.manipulation.g1_cricket.reference_dynamics import ReferenceDynamics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "g1_cricket_results/bimanual_grounded_v2"


def audit(
    output,
    right_reference=DEFAULT / "right_reference.npz",
    left_reference=DEFAULT / "left_reference.npz",
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    references = {"right": Path(right_reference).resolve(), "left": Path(left_reference).resolve()}
    inputs = [Path(__file__).resolve(), *references.values()]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [
        ROOT / f"src/unilab/assets/robots/g1/{name}" for name in ("g1.xml", "scene_flat.xml")
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    rows = []
    for hand, reference in references.items():
        with np.load(reference) as saved:
            poses, times = saved["qpos"].copy(), saved["times"].copy()
        if poses.shape[0] != 151 or not np.allclose(
            times, np.arange(151) * 0.02, atol=1e-12, rtol=0
        ):
            raise ValueError("audit requires all 151 reference poses at 50 Hz")
        with TemporaryDirectory(prefix="g1-batting-dynamics-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1BimanualContactCfg(handedness=hand).build_scene(
                ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
        ball_dofs = independent_ball_dofs(model)
        helper = ReferenceDynamics(model)
        for physics_dt in (0.00003125, 0.000015625):
            model.opt.timestep = physics_dt
            for stencil in (1, 2):
                velocity, acceleration, endpoint = reference_derivatives(
                    model, poses, times, stencil
                )
                velocity[:, ball_dofs], acceleration[:, ball_dofs] = 0, 0
                for index, time in enumerate(times):
                    result = evaluate_modes(
                        helper, poses[index], velocity[index], acceleration[index]
                    )
                    data = helper.data
                    gaps = {
                        side: [
                            float(
                                mujoco.mj_geomDistance(
                                    model,
                                    data,
                                    model.geom(f"{side}_foot{i}_collision").id,
                                    model.geom("pitch").id,
                                    1,
                                    None,
                                )
                            )
                            for i in range(1, 8)
                        ]
                        for side in ("left", "right")
                    }
                    ball_body = model.body("cricket_ball").id
                    ball_robot_contacts = [
                        [model.geom(int(geom)).name for geom in contact.geom]
                        for contact in data.contact
                        if ball_body in model.geom_bodyid[contact.geom]
                        and any(
                            body not in (0, ball_body) for body in model.geom_bodyid[contact.geom]
                        )
                    ]
                    if ball_robot_contacts:
                        raise ValueError(
                            "offline reference ball contacts robot; independent-ball exclusion is invalid"
                        )
                    rows.append(
                        {
                            "hand": hand,
                            "frame": index,
                            "time_s": float(time),
                            "physics_dt": physics_dt,
                            "derivative_half_stencil_s": 0.02 * stencil,
                            "endpoint_stencil": bool(endpoint[index]),
                            "crosses_target_knot": bool(
                                np.any(
                                    np.abs(np.array([0.55, 1.15, 1.45, 1.8, 2.45]) - time)
                                    <= 0.04 * stencil
                                )
                            ),
                            "foot_gaps_m": gaps,
                            "grip_gap_m": float(
                                np.linalg.norm(
                                    data.site("bat_lower_grip").xpos
                                    - data.site(f"{hand}_palm").xpos
                                )
                            ),
                            "excluded_ball_dofs": ball_dofs,
                            **result,
                        }
                    )
    summaries = []
    for hand in references:
        for dt in (0.00003125, 0.000015625):
            for stencil in (0.02, 0.04):
                subset = [
                    row
                    for row in rows
                    if (row["hand"], row["physics_dt"], row["derivative_half_stencil_s"])
                    == (hand, dt, stencil)
                ]
                summaries.append(
                    {
                        "hand": hand,
                        "physics_dt": dt,
                        "derivative_half_stencil_s": stencil,
                        "frames": len(subset),
                        "modes": {
                            mode: {
                                "peak_motor_excess": max(
                                    row["modes"][mode]["peak_motor_excess"] for row in subset
                                ),
                                "peak_reachable_force_excess": max(
                                    row["modes"][mode]["peak_reachable_force_excess"]
                                    for row in subset
                                ),
                                "peak_root_force_n": max(
                                    float(np.linalg.norm(row["modes"][mode]["root_residual"][:3]))
                                    for row in subset
                                ),
                                "peak_root_torque_nm": max(
                                    float(np.linalg.norm(row["modes"][mode]["root_residual"][3:]))
                                    for row in subset
                                ),
                            }
                            for mode in ("static", "velocity", "dynamic")
                        },
                    }
                )
    result = {
        "mujoco_version": mujoco.__version__,
        "scope": "offline_native_inverse_decomposition_no_rollout_no_promotion",
        "interpretation_guard": "Native soft-contact inverse alone does not prove physical infeasibility; separate contact/grip stabilization from inertial demand. No external root/ball force is applied.",
        "ball_exclusion": "Independent parked-ball DOFs are omitted from robot residuals and decomposition; no robot-ball contact is permitted. This is not an impact audit.",
        "derivative_convention": "mj_differentiatePos matches export_reference for 20 ms half-stencil; 40 ms is sensitivity only. Acceleration differences those velocities; endpoint flags cover both derivative stencils. Constant root quaternion required.",
        "input_hashes": hashes,
        "row_count": len(rows),
        "summaries": summaries,
        "rows": rows,
    }
    for relative, digest in hashes.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"audit input changed: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result, default=lambda value: value.tolist(), allow_nan=False) + "\n"
    ).encode()
    output.write_bytes(gzip.compress(payload, mtime=0) if output.suffix == ".gz" else payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--right-reference", type=Path, default=DEFAULT / "right_reference.npz")
    parser.add_argument("--left-reference", type=Path, default=DEFAULT / "left_reference.npz")
    args = parser.parse_args()
    report = audit(args.output, args.right_reference, args.left_reference)
    print(
        json.dumps({"row_count": report["row_count"], "summaries": report["summaries"]}, indent=2)
    )
