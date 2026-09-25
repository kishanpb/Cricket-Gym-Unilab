"""Offline whole-body retargeting of the earlier cricket running delivery."""

import mujoco
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .prior import SDK_DEFAULT, SDK_JOINTS
from .running_momentum import HeldBallMomentum

GATHER_TIME = 1.20
RELEASE_TIME = 1.82
END_TIME = 2.70


class BallisticRunupCOM:
    """Whole-system COM target with ballistic flight and C1 transitions, offline only."""

    def __init__(self, times, centers, gravity):
        self.parent = PchipInterpolator(times, centers, axis=0)
        self.start = np.asarray(centers[0])
        self.height = float(self.start[2])
        self.horizontal_velocity = (self.parent(GATHER_TIME)[:2] - self.start[:2]) / GATHER_TIME
        self.gravity = gravity
        self.takeoff_velocity = gravity * 0.08 / 2
        self.stance = CubicHermiteSpline(
            [0, 0.22],
            [self.height, self.height],
            [-self.takeoff_velocity, self.takeoff_velocity],
        )
        self.gather = CubicHermiteSpline(
            [GATHER_TIME, 1.32],
            [np.r_[self.parent(GATHER_TIME)[:2], self.height], self.parent(1.32)],
            [
                np.r_[self.horizontal_velocity, -self.takeoff_velocity],
                self.parent.derivative()(1.32),
            ],
        )

    def __call__(self, time):
        if time >= 1.32:
            return self.parent(time)
        if time >= GATHER_TIME:
            return self.gather(time)
        phase = time % 0.3
        if phase <= 0.22:
            height = float(self.stance(phase))
        else:
            flight = phase - 0.22
            height = self.height + self.takeoff_velocity * flight - self.gravity * flight**2 / 2
        return np.r_[self.start[:2] + self.horizontal_velocity * time, height]


class RunningDeliveryTargets:
    """G1-sized targets, not prescribed simulation poses or achieved dynamics."""

    def __init__(self, hand, lane_offset=0.0):
        if hand not in {"right", "left"}:
            raise ValueError("hand must be right or left")
        self.hand = hand
        self.side = 1 if hand == "right" else -1
        self.lane = self.side * (0.5 + lane_offset)
        self.root_x = PchipInterpolator(
            [0, 0.3, 0.6, 0.9, 1.2, 1.42, 1.65, RELEASE_TIME, 2.1, 2.5, END_TIME],
            np.array([-3, -2.45, -1.9, -1.35, -0.8, -0.4, -0.14, 0, 0.27, 0.85, 1]) * 0.55 - 0.45,
        )
        self.root_z = PchipInterpolator(
            [0, 1.2, 1.32, 1.42, 1.65, RELEASE_TIME, 2.1, 2.5, END_TIME],
            0.73 + 0.55 * (np.array([0.79, 0.79, 0.86, 0.78, 0.72, 0.81, 0.78, 0.79, 0.79]) - 0.79),
        )
        self.arm_angle = CubicHermiteSpline(
            [GATHER_TIME, 1.66, RELEASE_TIME, 1.90, 2.14, END_TIME],
            [2.4, 1.6, 0.18, 1.2, 2.5, 3.1],
            [0, -4, 10, 5, 0, 0],
        )
        self.flexion = PchipInterpolator(
            [0, GATHER_TIME, 1.66, 1.72, 1.90, 2.14, END_TIME],
            np.radians([65, 55, 15, 8, 8, 55, 65]),
        )
        self.lean = PchipInterpolator(
            [0, GATHER_TIME, 1.42, 1.65, RELEASE_TIME, 2.05, 2.4, END_TIME],
            [0.07, 0.07, 0.03, 0.04, 0.09, 0.24, 0.11, 0.07],
        )

    def root(self, time):
        bounce = 0.015 * np.sin(np.pi * time / 0.3) ** 2 if time < GATHER_TIME else 0
        return np.array([self.root_x(time), self.lane, self.root_z(time) + bounce])

    def foot(self, side, time):
        front = side != self.hand
        landings = (
            [
                (-0.6, -3.95),
                (0, -2.85),
                (0.6, -1.75),
                (1.2, -0.65),
                (1.65, 0.30),
                (2.5, 1.15),
                (3.1, 1.15),
            ]
            if front
            else [
                (-0.3, -3.4),
                (0.3, -2.3),
                (0.9, -1.2),
                (1.42, -0.45),
                (2.1, 0.50),
                (2.7, 0.90),
                (3.3, 0.90),
            ]
        )
        index = max(i for i, (t, _) in enumerate(landings[:-1]) if t <= time)
        start, x = landings[index]
        end, next_x = landings[index + 1]
        stance = 0.22 if start < 1.6 else 0.30
        fraction = np.clip((time - start - stance) / (end - start - stance), 0, 1)
        blend = fraction**2 * (3 - 2 * fraction)
        return np.array(
            [
                0.55 * (x + (next_x - x) * blend) - 0.45,
                self.lane + (0.12 if side == "left" else -0.12),
                0.035 + 0.11 * np.sin(np.pi * fraction) ** 2,
            ]
        )

    def arm(self, side, time):
        bowling = side == self.hand
        if time < GATHER_TIME:
            angle = np.pi + (0.5 if bowling else -0.5) * np.sin(2 * np.pi * time / 0.6)
            blend = np.clip((time - 1) / 0.2, 0, 1)
            blend = blend**2 * (3 - 2 * blend)
            angle = (1 - blend) * angle + blend * (2.4 if bowling else 2.1)
        elif bowling:
            angle = float(self.arm_angle(time))
        else:
            angle = float(
                np.interp(
                    time,
                    [GATHER_TIME, 1.65, RELEASE_TIME, 2.05, END_TIME],
                    [2.1, 1.1, 2.5, 3.4, 3.1],
                )
            )
        flexion = float(self.flexion(time)) if bowling else np.radians(65)
        lateral = 1 if side == "left" else -1
        upper = np.array([np.sin(angle), lateral * 0.35, np.cos(angle)])
        lower = np.array([np.sin(angle - flexion), lateral * 0.12, np.cos(angle - flexion)])
        upper /= np.linalg.norm(upper)
        lower /= np.linalg.norm(lower)
        y = np.array([0.0, 1.0, 0.0]) - lower[1] * lower
        y /= np.linalg.norm(y)
        return upper, lower, np.column_stack((lower, y, np.cross(lower, y)))


def retarget_running_delivery(
    model,
    times,
    hand,
    *,
    com_target=None,
    lane_offset=0.0,
    conserve_momentum=False,
    momentum_target=None,
):
    """Solve original G1 joints offline; do not use this loop as a physics rollout."""
    target = RunningDeliveryTargets(hand, lane_offset)
    if conserve_momentum and com_target is None:
        raise ValueError("momentum-conserving rotation requires a COM target")
    if momentum_target is not None and not conserve_momentum:
        raise ValueError("a momentum target requires momentum-conserving rotation")
    momentum = HeldBallMomentum(model, hand) if conserve_momentum else None
    was_tracking, desired_momentum, omega = False, None, np.zeros(3)
    recovery, landing_rotation = None, None
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    joints = np.array([model.joint(name).id for name in SDK_JOINTS])
    addresses = model.jnt_qposadr[joints]
    lower, upper = model.jnt_range[joints].T.copy()
    lower[[3, 9]] = 0.12
    previous = np.clip(SDK_DEFAULT, lower + 0.03, upper - 0.03)
    mujoco.mj_forward(model, data)
    arms = {}
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    for side in ("left", "right"):
        ids = [
            model.body(f"{side}_{part}_link").id
            for part in ("shoulder_roll", "elbow", "wrist_roll")
        ]
        positions = data.xpos[ids]
        arms[side] = ids, np.linalg.norm(np.diff(positions, axis=0), axis=1)
    pairs = [
        (model.geom(f"{side}_{part}_collision").id, model.geom("torso_collision").id)
        for side in ("left", "right")
        for part in ("elbow_yaw", "wrist", "hand")
    ]
    pairs += [
        (model.geom(f"{side}_{part}_collision").id, model.geom(f"{side}_hip_collision").id)
        for side in ("left", "right")
        for part in ("wrist", "hand")
    ]
    ball_joint = model.joint("ball_free")
    ball_qa = int(ball_joint.qposadr[0])
    ball_body = model.body("cricket_ball").id
    wrist_id = model.body(f"{hand}_wrist_yaw_link").id
    holder_offset = np.array([0.15, 0.06 if hand == "left" else -0.06, 0])
    poses, errors = [], []
    for frame, time in enumerate(times):
        dt = float(time - times[frame - 1]) if frame else 0.0
        flying = bool(
            conserve_momentum
            and frame
            and time <= GATHER_TIME + 1e-12
            and (time - dt / 2) % 0.3 > 0.22
        )
        tracking = flying or bool(
            momentum_target is not None and frame and time <= GATHER_TIME + 1e-12
        )
        if tracking and momentum_target is not None:
            desired_momentum = momentum_target(time - dt / 2)
        elif flying and not was_tracking:
            desired_momentum = momentum.measure(poses[-2], poses[-1], dt)
        if was_tracking and not tracking:
            landing = float(times[frame - 1])
            landing_rotation = Rotation.from_quat(poses[-1][[4, 5, 6, 3]])
            recovery = CubicHermiteSpline(
                [landing, landing + 0.22],
                [np.zeros(3), landing_rotation.inv().as_rotvec()],
                [omega, np.zeros(3)],
            )
        root_quaternion = np.array([1.0, 0, 0, 0])
        if recovery is not None and time < recovery.x[-1]:
            root_quaternion = (landing_rotation * Rotation.from_rotvec(recovery(time))).as_quat()[
                [3, 0, 1, 2]
            ]
        data.qpos[:3] = target.root(float(time))
        data.qpos[3:7] = root_quaternion
        foot_targets = np.array([target.foot(side, time) for side in ("left", "right")])
        arm_targets = {side: target.arm(side, time) for side in arms}

        def residual(q):
            data.qpos[addresses] = q
            if tracking:
                data.qpos[3:7] = momentum.advance(poses[-1], q, desired_momentum, dt)[0]
            if com_target is not None:
                data.qpos[:3] = target.root(time)
            mujoco.mj_kinematics(model, data)
            if com_target is not None:
                # Eliminate offline root translation using the exact held-ball COM.
                held = data.xpos[wrist_id] + data.xmat[wrist_id].reshape(3, 3) @ holder_offset
                center = (
                    model.body_mass @ data.xipos
                    + model.body_mass[ball_body] * (held - data.xipos[ball_body])
                ) / model.body_mass.sum()
                data.qpos[:3] += com_target(time) - center
                mujoco.mj_kinematics(model, data)
            result = [
                50 * (data.xpos[feet] - foot_targets).ravel(),
                3 * Rotation.from_matrix(data.xmat[feet].reshape(2, 3, 3)).as_rotvec().ravel(),
                0.1 * (q - previous),
                np.array([0.1, 2, 2]) * (q[12:15] - [0, 0, target.lean(time)]),
            ]
            for side, (ids, lengths) in arms.items():
                up, down, rotation = arm_targets[side]
                shoulder, elbow, wrist = data.xpos[ids]
                result.extend(
                    [
                        25 * (elbow - shoulder - lengths[0] * up),
                        25 * (wrist - elbow - lengths[1] * down),
                        0.15
                        * Rotation.from_matrix(
                            rotation.T @ data.body(f"{side}_wrist_yaw_link").xmat.reshape(3, 3)
                        ).as_rotvec(),
                    ]
                )
            result.append(
                np.array(
                    [
                        100 * max(0.012 - mujoco.mj_geomDistance(model, data, a, b, 0.1, None), 0)
                        for a, b in pairs
                    ]
                )
            )
            return np.concatenate(result)

        lo, hi = lower + 0.03, upper - 0.03
        if frame:
            step = 12 * (time - times[frame - 1])
            lo, hi = np.maximum(lo, previous - step), np.minimum(hi, previous + step)
        solved = least_squares(
            residual,
            previous,
            bounds=(lo, hi),
            max_nfev=120,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        residual(solved.x)
        previous = solved.x.copy()
        data.qpos[ball_qa : ball_qa + 3] = (
            data.xpos[wrist_id] + data.xmat[wrist_id].reshape(3, 3) @ holder_offset
        )
        data.qpos[ball_qa + 3 : ball_qa + 7] = data.xquat[wrist_id]
        mujoco.mj_forward(model, data)
        momentum_error = None
        if tracking:
            _, omega = momentum.advance(poses[-1], solved.x, desired_momentum, dt)
            momentum_error = float(
                np.linalg.norm(momentum.measure(poses[-1], data.qpos, dt) - desired_momentum)
            )
        was_tracking = tracking
        unexpected = []
        for c in data.contact:
            names = {model.geom(int(g)).name for g in c.geom}
            if c.dist < -0.001 and not ("pitch" in names and any("foot" in n for n in names)):
                unexpected.append({"pair": sorted(names), "penetration_m": float(-c.dist)})
        poses.append(data.qpos.copy())
        arm_error = []
        for side, (ids, lengths) in arms.items():
            up, down, _ = arm_targets[side]
            shoulder, elbow, wrist = data.xpos[ids]
            arm_error.extend(
                [
                    np.linalg.norm(elbow - shoulder - lengths[0] * up),
                    np.linalg.norm(wrist - elbow - lengths[1] * down),
                ]
            )
        errors.append(
            {
                "time_s": float(time),
                "optimizer_success": bool(solved.success),
                "optimizer_evaluations": int(solved.nfev),
                "optimizer_optimality": float(solved.optimality),
                "flight_momentum_error_nms": momentum_error if flying else None,
                "centroidal_momentum_error_nms": momentum_error,
                "root_rotation_rad": float(Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).magnitude()),
                "arm_segment_error_m": float(max(arm_error)),
                "foot_error_m": float(np.linalg.norm(data.xpos[feet] - foot_targets, axis=1).max()),
                "minimum_tracked_clearance_m": min(
                    float(mujoco.mj_geomDistance(model, data, a, b, 0.1, None)) for a, b in pairs
                ),
                "com_position_error_m": (
                    (data.subtree_com[0] - com_target(time)).tolist()
                    if com_target is not None
                    else None
                ),
                "unexpected_penetrations": unexpected,
            }
        )
    return {"times": np.asarray(times), "qpos": np.asarray(poses), "errors": errors}
