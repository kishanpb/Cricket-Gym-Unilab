"""Two-hand mechanical grip and cold-path G1 cricket reference retargeting."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .prior import SDK_DEFAULT, SDK_JOINTS, build_prior_scene


def build_bimanual_scene(source: Path, destination: Path, hand: str) -> tuple[str, ...]:
    if hand not in {"right", "left"}:
        raise ValueError("hand must be right or left")
    top = "left" if hand == "right" else "right"
    guards = build_prior_scene(source, destination, top)
    tree = ET.parse(destination)
    root = tree.getroot()
    bat = root.find(".//body[@name='cricket_bat']")
    sensors = root.find("sensor")
    key = root.find("keyframe/key")
    assert bat is not None and sensors is not None and key is not None
    sign = 1 if hand == "right" else -1
    bat.set("quat", f"{np.sqrt(0.5)} 0 0 {sign * np.sqrt(0.5)}")
    # Mechanical palm fixtures, not an articulated-finger grasp. Only the handle
    # is non-colliding; the blade and both arms retain their collision geometry.
    handle = bat.find("geom[@name='bat_handle']")
    assert handle is not None
    handle.set("contype", "0")
    handle.set("conaffinity", "0")
    ET.SubElement(bat, "site", name="bat_lower_grip", pos="0 0 -0.11", size="0.008")
    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "connect",
        name="second_hand_grip",
        site1="bat_lower_grip",
        site2=f"{hand}_palm",
        solref="0.012 1",
        solimp="0.95 0.99 0.001",
    )
    for name, site in (
        ("lower_grip_world", "bat_lower_grip"),
        ("lower_palm_world", f"{hand}_palm"),
    ):
        ET.SubElement(sensors, "framepos", name=name, objtype="site", objname=site)
    qpos = np.fromstring(key.attrib["qpos"], sep=" ")
    qpos[:3] = [0, sign * 0.28, 0.79]
    qpos[3:7] = [np.sqrt(0.5), 0, 0, -sign * np.sqrt(0.5)]
    key.set("qpos", " ".join(map(str, qpos)))
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return guards


def batting_targets(times: np.ndarray, hand: str) -> tuple[np.ndarray, np.ndarray]:
    """Guard, backlift, downswing, drive, recovery; scaled to G1 reach."""
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation

    if hand not in {"right", "left"}:
        raise ValueError("hand must be right or left")
    knots = [0, 0.55, 1.15, 1.45, 1.8, 2.45, 3.0]
    grips = PchipInterpolator(
        knots,
        [
            [0.03, 0, 0.90],
            [-0.02, 0, 0.94],
            [-0.04, 0, 1.00],
            [0.14, 0, 0.92],
            [0.24, 0, 1.04],
            [0.08, 0, 0.94],
            [0.03, 0, 0.90],
        ],
    )(times)
    angles = PchipInterpolator(knots, [0.10, 0.55, 0.95, -0.10, -0.95, -0.25, 0.10])(times)
    rotations = Rotation.from_euler("y", angles[:, None]).as_matrix()
    return grips, rotations


def retarget_batting(model: mujoco.MjModel, times: np.ndarray, hand: str) -> dict:
    """Offline IK only. The returned poses are targets, not physics evidence."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joint_ids = np.array([model.joint(name).id for name in SDK_JOINTS])
    addresses = model.jnt_qposadr[joint_ids]
    active = np.arange(29)
    lower, upper = model.jnt_range[joint_ids[active]].T
    lower = lower.copy()
    lower[[3, 9]] = 0.15
    top_id = model.site("bat_fixture").id
    lower_id = model.site("bat_lower_grip").id
    palm_id = model.site(f"{hand}_palm").id
    bat_id = model.body("cricket_bat").id
    grips, rotations = batting_targets(times, hand)
    mujoco.mj_forward(model, data)
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    feet_pos = data.xpos[feet].copy()
    feet_rotation = data.xmat[feet].reshape(2, 3, 3).copy()
    root_xy = data.qpos[:2].copy()
    pelvis_id = model.body("pelvis").id
    pairs = [
        (model.geom(f"{side}_{part}_collision").id, model.geom("torso_collision").id)
        for side in ("left", "right")
        for part in ("shoulder_yaw", "elbow_yaw", "wrist", "hand")
    ]
    pairs += [
        (model.geom("bat_blade").id, model.geom(f"{side}_thigh_collision").id)
        for side in ("left", "right")
    ]
    pairs += [
        (model.geom(f"left_{left}_collision").id, model.geom(f"right_{right}_collision").id)
        for left in ("elbow_yaw", "wrist", "hand")
        for right in ("elbow_yaw", "wrist", "hand")
    ]
    previous = np.r_[SDK_DEFAULT, data.qpos[:3]]
    lower, upper = np.r_[lower, root_xy - 0.12, 0.68], np.r_[upper, root_xy + 0.12, 0.81]
    reference, errors = [], []
    for grip, rotation in zip(grips, rotations, strict=True):

        def residual(angles):
            data.qpos[addresses] = angles[:29]
            data.qpos[:3] = angles[29:]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            actual_rotation = data.xmat[bat_id].reshape(3, 3)
            orientation = Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec()
            return np.concatenate(
                (
                    15 * (data.site_xpos[top_id] - grip),
                    20 * (data.site_xpos[lower_id] - data.site_xpos[palm_id]),
                    orientation,
                    0.025 * (angles - previous),
                    0.3 * (angles[12:15] - [0, 0, 0.15]),
                    25 * (data.xpos[feet] - feet_pos).ravel(),
                    2
                    * Rotation.from_matrix(
                        feet_rotation.transpose(0, 2, 1) @ data.xmat[feet].reshape(2, 3, 3)
                    )
                    .as_rotvec()
                    .ravel(),
                    15 * (data.subtree_com[pelvis_id, :2] - feet_pos[:, :2].mean(axis=0)),
                    np.array(
                        [
                            30
                            * max(0.012 - mujoco.mj_geomDistance(model, data, a, b, 0.1, None), 0)
                            for a, b in pairs
                        ]
                    ),
                )
            )

        solved = least_squares(
            residual,
            previous,
            bounds=(lower + 0.01, upper - 0.01),
            max_nfev=120,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        residual(solved.x)
        previous = solved.x.copy()
        reference.append(data.qpos.copy())
        errors.append(
            {
                "maximum_foot_position_error_m": float(
                    np.linalg.norm(data.xpos[feet] - feet_pos, axis=1).max()
                ),
                "center_of_mass_support_error_m": float(
                    np.linalg.norm(data.subtree_com[pelvis_id, :2] - feet_pos[:, :2].mean(axis=0))
                ),
                "minimum_tracked_clearance_m": min(
                    float(mujoco.mj_geomDistance(model, data, a, b, 0.1, None)) for a, b in pairs
                ),
                "top_grip_error_m": float(np.linalg.norm(data.site_xpos[top_id] - grip)),
                "second_grip_error_m": float(
                    np.linalg.norm(data.site_xpos[lower_id] - data.site_xpos[palm_id])
                ),
                "bat_rotation_error_rad": float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            rotation.T @ data.xmat[bat_id].reshape(3, 3)
                        ).as_rotvec()
                    )
                ),
                "optimizer_success": bool(solved.success),
            }
        )
    return {"times": times, "qpos": np.asarray(reference), "errors": errors}
