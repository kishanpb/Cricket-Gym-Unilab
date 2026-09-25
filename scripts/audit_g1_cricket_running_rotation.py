"""Refine aerial momentum derivatives on a fixed, held-ball reference curve."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from g1_cricket_delivery_trial import capsule_bounds
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg
from unilab.tasks.manipulation.g1_cricket.prior import SDK_JOINTS
from unilab.tasks.manipulation.g1_cricket.running import GATHER_TIME, BallisticRunupCOM
from unilab.tasks.manipulation.g1_cricket.running_support import (
    ForeAftSupportCOM,
    LateralSupportCOM,
)

ROOT = Path(__file__).resolve().parents[1]


class ReferenceCurve:
    """Interpolate original joints/orientation; preserve COM and mechanical holder."""

    def __init__(self, model, times, poses, hand):
        self.model = model
        self.data = mujoco.MjData(model)
        self.initial = poses[0].copy()
        joints = [model.joint(name).id for name in SDK_JOINTS]
        self.addresses = model.jnt_qposadr[joints]
        self.joints = CubicSpline(times, poses[:, self.addresses], axis=0)
        self.position = CubicSpline(times, poses[:, :3], axis=0)
        self.rotation = RotationSpline(times, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))
        self.wrist = model.body(f"{hand}_wrist_yaw_link").id
        self.ball = model.body("cricket_ball").id
        self.ball_q = int(model.joint("ball_free").qposadr[0])
        self.offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
        centers = []
        for pose in poses:
            self.data.qpos[:] = pose
            mujoco.mj_forward(model, self.data)
            centers.append(self.data.subtree_com[0].copy())
        self.ballistic = BallisticRunupCOM(times, centers, -model.opt.gravity[2])
        self.center = CubicSpline(times, centers, axis=0)

    def com(self, time):
        return self.ballistic(time) if time <= GATHER_TIME else self.center(time)

    def __call__(self, time):
        model, data = self.model, self.data
        data.qpos[:] = self.initial
        data.qpos[:3] = self.position(time)
        data.qpos[3:7] = self.rotation(time).as_quat()[[3, 0, 1, 2]]
        data.qpos[self.addresses] = self.joints(time)
        mujoco.mj_kinematics(model, data)
        held = data.xpos[self.wrist] + data.xmat[self.wrist].reshape(3, 3) @ self.offset
        center = (
            model.body_mass @ data.xipos
            + model.body_mass[self.ball] * (held - data.xipos[self.ball])
        ) / model.body_mass.sum()
        translation = self.com(time) - center
        data.qpos[:3] += translation
        data.qpos[self.ball_q : self.ball_q + 3] = held + translation
        data.qpos[self.ball_q + 3 : self.ball_q + 7] = data.xquat[self.wrist]
        return data.qpos.copy()


def momentum_at(model, curve, time, velocity_step):
    data = mujoco.MjData(model)
    before, after = curve(time - velocity_step), curve(time + velocity_step)
    data.qpos[:] = curve(time)
    mujoco.mj_differentiatePos(model, data.qvel, 2 * velocity_step, before, after)
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_angmom[0].copy()


def audit(directory):
    output = directory / "reference_rotation_refinement.json"
    if output.exists():
        raise FileExistsError(output)
    source = json.loads((directory / "evaluation.json").read_text())
    suffix = "dense_reference.npz" if source.get("retarget_substeps", 1) > 1 else "reference.npz"
    files = [Path(__file__), ROOT / "scripts/g1_cricket_delivery_trial.py"]
    files += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    files += [ROOT / f"src/unilab/assets/robots/g1/{name}" for name in ("g1.xml", "scene_flat.xml")]
    files += [directory / "evaluation.json"]
    files += [directory / f"{hand}_{suffix}" for hand in ("right", "left")]
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    rows = []
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-rotation-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(
                ROOT / "src/unilab/assets/robots/g1/g1.xml", scene
            )
            model = mujoco.MjModel.from_xml_path(str(scene))
        with np.load(directory / f"{hand}_{suffix}") as reference:
            curve = ReferenceCurve(model, reference["times"], reference["qpos"], hand)
        if source.get("lateral_support_com", False):
            lane = (1 if hand == "right" else -1) * (0.5 + source["outward_lane_offset_m"])
            support = ForeAftSupportCOM if source.get("fore_aft_support_com") else LateralSupportCOM
            curve.ballistic = support(curve.ballistic, hand, lane)
        feet = [
            np.flatnonzero(
                (model.geom_bodyid == model.body(f"{side}_ankle_roll_link").id)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            for side in ("left", "right")
        ]
        data = mujoco.MjData(model)
        for time in (0.26, 0.56, 0.86, 1.16):
            for step in (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625):
                clearance, world_contacts = [], 0
                for sample in np.linspace(time - step - 1e-4, time + step + 1e-4, 33):
                    data.qpos[:] = curve(sample)
                    mujoco.mj_forward(model, data)
                    clearance.extend(
                        capsule_bounds(
                            data.geom_xpos[ids],
                            data.geom_xmat[ids].reshape(-1, 3, 3),
                            model.geom_size[ids],
                            model.geom_type[ids],
                        )[0, 2]
                        for ids in feet
                    )
                    world_contacts += sum(
                        np.any(model.geom_bodyid[c.geom] == 0) and c.dist <= 0 for c in data.contact
                    )
                for velocity_step in (1e-4, 1e-5):
                    torque = (
                        momentum_at(model, curve, time + step, velocity_step)
                        - momentum_at(model, curve, time - step, velocity_step)
                    ) / (2 * step)
                    rows.append(
                        {
                            "hand": hand,
                            "time_s": time,
                            "torque_step_s": step,
                            "velocity_step_s": velocity_step,
                            "required_external_torque_nm": torque.tolist(),
                            "sampled_minimum_foot_clearance_m": float(min(clearance)),
                            "sampled_world_contacts": int(world_contacts),
                        }
                    )
    for path, digest in hashes.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("rotation audit input changed")
    result = {
        "scope": "fixed_curve_offline_momentum_refinement_not_a_physical_rollout",
        "curve": "Cubic joint/position and rotation splines, exact run-up COM, ball held at wrist; not new IK samples or training",
        "guard": "Inferred angular-momentum derivative, not applied/measured torque or promotion evidence. Sampled clearance does not certify the entire continuous interval.",
        "input_sha256": hashes,
        "reference_file_suffix": suffix,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        [row for row in rows if row["torque_step_s"] == 0.000625 and row["velocity_step_s"] == 1e-5]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory)
