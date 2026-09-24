"""Check discrete run-up COM motion against gravity during aerial reference frames."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from g1_cricket_delivery_trial import capsule_bounds

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.running import GATHER_TIME

ROOT = Path(__file__).resolve().parents[1]


def aerial_residual(times, com, airborne, mass, gravity):
    dt = np.diff(times)
    np.testing.assert_allclose(dt, dt[0], rtol=0, atol=1e-12)
    acceleration = np.diff(com, n=2, axis=0) / dt[0] ** 2
    rows = []
    for i in range(1, len(times) - 1):
        if times[i + 1] <= GATHER_TIME and np.all(airborne[i - 1 : i + 2]):
            rows.append(
                {
                    "time_s": float(times[i]),
                    "com_acceleration_m_s2": acceleration[i - 1].tolist(),
                    "required_nongravity_force_n": (
                        mass * (acceleration[i - 1] - gravity)
                    ).tolist(),
                }
            )
    return rows


def aerial_momentum_residual(times, momentum, airborne):
    dt = np.diff(times)
    np.testing.assert_allclose(dt, dt[0], rtol=0, atol=1e-12)
    return [
        {
            "time_s": float(times[i]),
            "required_external_torque_nm": (
                (momentum[i + 1] - momentum[i - 1]) / (2 * dt[0])
            ).tolist(),
        }
        for i in range(2, len(times) - 2)
        if times[i + 2] <= GATHER_TIME and np.all(airborne[i - 1 : i + 2])
    ]


def audit(directory, *, flat_reference_layout=False):
    output = directory / "reference_flight_audit.json"
    if output.exists():
        raise FileExistsError(output)
    files = [Path(__file__), ROOT / "scripts/g1_cricket_delivery_trial.py"]
    files += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    files += [ROOT / f"src/unilab/assets/robots/g1/{name}" for name in ("g1.xml", "scene_flat.xml")]
    references = {
        hand: directory
        / (f"{hand}_reference.npz" if flat_reference_layout else f"{hand}/reference.npz")
        for hand in ("right", "left")
    }
    files += list(references.values())
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    rows = []
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-running-flight-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
                ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
        if model.opt.density or model.opt.viscosity or np.any(model.body_gravcomp):
            raise ValueError("gravity-only aerial check excludes fluid and gravity compensation")
        data = mujoco.MjData(model)
        feet = [
            np.flatnonzero(
                (model.geom_bodyid == model.body(f"{side}_ankle_roll_link").id)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            for side in ("left", "right")
        ]
        com, airborne, momentum = [], [], []
        with np.load(references[hand]) as reference:
            times = reference["times"]
            poses = reference["qpos"]
            for i, pose in enumerate(poses):
                data.qpos[:] = pose
                before, after = max(0, i - 1), min(len(poses) - 1, i + 1)
                mujoco.mj_differentiatePos(
                    model, data.qvel, times[after] - times[before], poses[before], poses[after]
                )
                mujoco.mj_forward(model, data)
                mujoco.mj_subtreeVel(model, data)
                momentum.append(data.subtree_angmom[0].copy())
                com.append(data.subtree_com[0].copy())
                clearance = [
                    capsule_bounds(
                        data.geom_xpos[ids],
                        data.geom_xmat[ids].reshape(-1, 3, 3),
                        model.geom_size[ids],
                        model.geom_type[ids],
                    )[0, 2]
                    for ids in feet
                ]
                world_contact = any(
                    np.any(model.geom_bodyid[contact.geom] == 0) and contact.dist <= 0
                    for contact in data.contact
                )
                airborne.append(min(clearance) > 0.002 and not world_contact)
        rows.append(
            {
                "hand": hand,
                "total_mass_kg": float(model.body_mass.sum()),
                "gravity_m_s2": model.opt.gravity.tolist(),
                "samples": aerial_residual(
                    times,
                    np.asarray(com),
                    np.asarray(airborne),
                    model.body_mass.sum(),
                    model.opt.gravity,
                ),
                "angular_momentum_samples": aerial_momentum_residual(
                    times, np.asarray(momentum), np.asarray(airborne)
                ),
            }
        )
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result = {
        "scope": "offline_runup_reference_consistency_not_measured_contact_forces",
        "interpretation": "Central COM differences over three aerial samples. A ballistic segment requires zero nongravity force; unsampled contact is not certified absent. This does not qualify a physical rollout.",
        "momentum_interpretation": "World-frame angular momentum about the whole-system COM from mj_subtreeVel and centered pose velocities. Torque estimates use three aerial momentum frames spanning five poses; the outer velocity samples can include stance/flight boundaries. These coarse offline estimates require temporal refinement, are not measured/applied torque and cannot qualify a rollout.",
        "input_sha256": hashes,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--flat-reference-layout", action="store_true")
    args = parser.parse_args()
    audit(args.directory, flat_reference_layout=args.flat_reference_layout)
