"""Necessary planar support bounds for a fixed offline running reference."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from audit_g1_cricket_running_rotation import ReferenceCurve, momentum_at
from g1_cricket_delivery_trial import capsule_bounds

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.running_support import LateralSupportCOM

ROOT = Path(__file__).resolve().parents[1]


def ground_wrench(center, acceleration, angular_rate, mass, gravity):
    force = mass * (np.asarray(acceleration) - gravity)
    moment = angular_rate + np.cross(center, force)
    cop = np.array([-moment[1], moment[0]]) / force[2] if force[2] > 1 else None
    return force, cop


def audit(directory):
    output = directory / "support_audit.json"
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((directory / "evaluation.json").read_text())
    inputs = [
        Path(__file__),
        ROOT / "scripts/audit_g1_cricket_running_rotation.py",
        ROOT / "scripts/g1_cricket_delivery_trial.py",
        directory / "evaluation.json",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [
        ROOT / f"src/unilab/assets/robots/g1/{name}" for name in ("g1.xml", "scene_flat.xml")
    ]
    inputs += [directory / f"{hand}_reference.npz" for hand in ("right", "left")]
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    rows = []
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-support-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
                ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
        with np.load(directory / f"{hand}_reference.npz") as reference:
            curve = ReferenceCurve(model, reference["times"], reference["qpos"], hand)
        if source.get("lateral_support_com", False):
            lane = (1 if hand == "right" else -1) * (0.5 + source["outward_lane_offset_m"])
            curve.ballistic = LateralSupportCOM(curve.ballistic, hand, lane)
        data = mujoco.MjData(model)
        feet = [
            np.flatnonzero(
                (model.geom_bodyid == model.body(f"{side}_ankle_roll_link").id)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            for side in ("left", "right")
        ]
        for tick in range(1, 120):
            time = tick * 0.01
            data.qpos[:] = curve(time)
            mujoco.mj_forward(model, data)
            bounds = np.array(
                [
                    capsule_bounds(
                        data.geom_xpos[ids],
                        data.geom_xmat[ids].reshape(-1, 3, 3),
                        model.geom_size[ids],
                        model.geom_type[ids],
                    )
                    for ids in feet
                ]
            )
            supports = bounds[:, 0, 2] <= 0.002
            box = (
                np.stack((bounds[supports, 0, :2].min(axis=0), bounds[supports, 1, :2].max(axis=0)))
                if supports.any()
                else None
            )
            for step in (0.00125, 0.000625):
                center = curve.com(time)
                acceleration = (
                    curve.com(time + step) - 2 * center + curve.com(time - step)
                ) / step**2
                torque = (
                    momentum_at(model, curve, time + step, 1e-5)
                    - momentum_at(model, curve, time - step, 1e-5)
                ) / (2 * step)
                force, cop = ground_wrench(
                    center, acceleration, torque, model.body_mass.sum(), model.opt.gravity
                )
                outside = (
                    np.maximum(np.maximum(box[0] - cop, cop - box[1]), 0)
                    if box is not None and cop is not None
                    else None
                )
                phase = time % 0.3
                rows.append(
                    {
                        "hand": hand,
                        "time_s": time,
                        "derivative_step_s": step,
                        "crosses_phase_boundary": min(phase, abs(phase - 0.22), 0.3 - phase) < step,
                        "center_m": center.tolist(),
                        "required_force_n": force.tolist(),
                        "required_centroidal_torque_nm": torque.tolist(),
                        "required_cop_m": cop.tolist() if cop is not None else None,
                        "foot_bounds_m": bounds.tolist(),
                        "support_feet": supports.tolist(),
                        "support_outer_box_m": box.tolist() if box is not None else None,
                        "outside_outer_box_m": outside.tolist() if outside is not None else None,
                    }
                )
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("support audit input changed")
    result = {
        "scope": "offline_inferred_ground_wrench_not_a_physical_rollout",
        "guard": "The axis-aligned projected foot box overestimates planar support; outside disproves support at the sampled pose, inside does not prove feasibility. No friction, yaw-wrench, actuator or continuous-contact certificate is claimed. Boundary rows and both derivative resolutions are retained.",
        "input_sha256": hashes,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
