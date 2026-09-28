"""Pinned AMP actor transfer to native MuJoCo G1; locomotion, not bowling."""

import argparse
from collections import deque
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from integrations.g1_dynamics.twist2_cpu_probe import fingerprint

COMMIT = "4a4dff791ec9ba376a39894d60832754124247de"
CHECKPOINT_SHA256 = "5de7628fe92bd74118016c592b95eb17183547a5289708622c80a7815da025ac"
# Isaac Lab order is breadth-first, unlike the native model's SDK ordering.
JOINTS = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)
TERM_WIDTHS = (3, 6, 3, 29, 29, 29)


def default_angles():
    values = dict(
        hip_pitch=-0.1, knee=0.3, ankle_pitch=-0.2, shoulder_pitch=0.3, elbow=0.97
    )
    result = []
    for name in JOINTS:
        kind = name.split("_", 1)[1]
        value = values.get(kind, 0.0)
        if kind in ("shoulder_roll", "wrist_roll"):
            value = (0.25 if kind == "shoulder_roll" else 0.15) * (
                1 if name.startswith("left_") else -1
            )
        result.append(value)
    return np.asarray(result)


def policy_gains():
    gains = []
    for name in JOINTS:
        kind = name.split("_", 1)[1]
        if kind in ("hip_pitch", "hip_roll"):
            pair = (100, 3)
        elif kind == "knee":
            pair = (150, 5)
        elif kind == "hip_yaw":
            pair = (80, 2)
        elif name == "waist_yaw":
            pair = (150, 4)
        elif name.startswith("waist_"):
            pair = (40, 4)
        elif kind.startswith("ankle_"):
            pair = (35, 2)
        elif kind in ("wrist_pitch", "wrist_yaw"):
            pair = (25, 1)
        else:
            pair = (30, 1)
        gains.append(pair)
    return np.asarray(gains, dtype=float).T


class ObservationHistory:
    def __init__(self):
        self.frames = deque(maxlen=5)

    def append(self, terms):
        if tuple(np.shape(term) for term in terms) != tuple((w,) for w in TERM_WIDTHS):
            raise ValueError("Expected six AMP observation terms with 99 total values")
        terms = tuple(np.asarray(term, dtype=np.float32).copy() for term in terms)
        if not all(np.isfinite(term).all() for term in terms):
            raise ValueError("Nonfinite AMP observation")
        if not self.frames:
            self.frames.extend([terms] * 4)
        self.frames.append(terms)
        # Isaac Lab flattens each term's oldest-to-newest history before concatenation.
        return np.concatenate(
            [np.stack([f[i] for f in self.frames]).ravel() for i in range(len(terms))]
        )


def local_rotation(quat):
    rotation = Rotation.from_quat(quat, scalar_first=True)
    matrix = rotation.as_matrix()
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    local = (Rotation.from_euler("z", -yaw) * rotation).as_matrix()
    return np.r_[local[:, 0], local[:, 2]]


def load_actor(upstream):
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError("AMP checkout differs from reviewed revision")
    checkpoint = upstream / "checkpoints/model_6200.pt"
    if fingerprint(checkpoint)["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("AMP checkpoint hash mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor = nn.Sequential(
        nn.Linear(495, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, 29),
    )
    actor.load_state_dict(
        {
            k.removeprefix("actor."): v
            for k, v in saved["model_state_dict"].items()
            if k.startswith("actor.")
        },
        strict=True,
    )
    actor.eval()
    return actor


def command_at(
    time_s, speed, heading=None, heading_gain=0.5, lane_error=None,
    *, preserve_direction=False, lateral_velocity=0.0, lane_damping=0.0,
):
    yaw_rate = 0 if heading is None else np.clip(-heading_gain * heading, -1, 1)
    forward = speed * min(time_s, 1.0) * np.clip(6 - time_s, 0, 1)
    sideways = 0.0
    if lane_error is not None:
        lateral = np.clip(-2 * lane_error - lane_damping * lateral_velocity, -0.4, 0.4)
        cosine, sine = np.cos(heading), np.sin(heading)
        forward, sideways = (
            cosine * forward + sine * lateral,
            -sine * forward + cosine * lateral,
        )
    if preserve_direction:
        scale = max(1.0, forward / 3.0, -forward / 0.5, abs(sideways) / 0.5)
        forward, sideways = forward / scale, sideways / scale
    return np.array(
        [np.clip(forward, -0.5, 3.0), np.clip(sideways, -0.5, 0.5), yaw_rate]
    )


def run_case(
    actor,
    scene,
    output,
    speed,
    dt,
    *,
    heading_control=False,
    heading_gain=0.5,
    lane_control=False,
):
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    joints = np.array([model.joint(name + "_joint").id for name in JOINTS])
    q, v = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    ids = np.array([model.actuator(name + "_joint").id for name in JOINTS])
    native_kp = model.actuator_gainprm[ids, 0].copy()
    native_kd = -model.actuator_biasprm[ids, 2].copy()
    caps = model.actuator_forcerange[ids, 1].copy()
    np.testing.assert_array_equal(model.actuator_biasprm[ids, 1], -native_kp)
    assert np.all(model.actuator_forcelimited[ids])
    kp, kd = policy_gains()
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0.8, 1, 0, 0, 0]
    data.qpos[q] = default_angles()
    mujoco.mj_forward(model, data)
    history = ObservationHistory()
    previous = np.zeros(29)
    target = default_angles()
    period = round(0.02 / dt)
    assert np.isclose(period * dt, 0.02)
    states, observations, actions, metrics = [], [], [], []
    floor = model.geom("floor").id
    foot_bodies = [
        model.body(side + "_ankle_roll_link").id for side in ("left", "right")
    ]
    state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    terminal = "complete"
    for step in range(round(8 / dt)):
        if step % period == 0:
            rotation = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()
            heading = (
                np.arctan2(rotation[1, 0], rotation[0, 0])
                if heading_control or lane_control
                else None
            )
            obs = history.append(
                (
                    data.qvel[3:6],
                    local_rotation(data.qpos[3:7]),
                    command_at(
                        data.time,
                        speed,
                        heading,
                        heading_gain,
                        data.qpos[1] if lane_control else None,
                    ),
                    data.qpos[q],
                    data.qvel[v],
                    previous,
                )
            )
            with torch.no_grad():
                previous = actor(torch.from_numpy(obs)[None])[0].numpy()
            if not np.isfinite(previous).all():
                raise ValueError("Nonfinite AMP action")
            target = default_angles() + 0.25 * previous
            observations.append(obs)
            actions.append(previous.copy())
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            states.append(state.copy())
        torque = np.clip(kp * (target - data.qpos[q]) - kd * data.qvel[v], -caps, caps)
        data.ctrl[ids] = data.qpos[q] + (torque + native_kd * data.qvel[v]) / native_kp
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        loads = np.zeros(2)
        forbidden = 0
        force = np.empty(6)
        for i, contact in enumerate(data.contact):
            if floor not in contact.geom:
                continue
            other = contact.geom[1] if contact.geom[0] == floor else contact.geom[0]
            body = model.geom_bodyid[other]
            mujoco.mj_contactForce(model, data, i, force)
            if body in foot_bodies:
                loads[foot_bodies.index(body)] += max(force[0], 0)
            elif force[0] > 1:
                forbidden += 1
        tilt = np.arccos(
            np.clip(
                Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_matrix()[2, 2],
                -1,
                1,
            )
        )
        excess = np.maximum(
            model.jnt_range[joints, 0] - data.qpos[q],
            data.qpos[q] - model.jnt_range[joints, 1],
        ).max(initial=0)
        force_excess = np.maximum(np.abs(data.actuator_force[ids]) - caps, 0).max()
        support = max(np.abs(data.qfrc_applied).max(), np.abs(data.xfrc_applied).max())
        metrics.append(
            np.r_[
                data.time,
                data.qpos[:3],
                data.qvel[:3],
                tilt,
                loads,
                excess,
                force_excess,
                support,
                forbidden,
            ]
        )
        if not np.isfinite(np.r_[data.qpos, data.qvel, metrics[-1]]).all():
            raise ValueError("Nonfinite AMP dynamics")
        if data.qpos[2] < 0.5 or tilt > 0.8 or forbidden:
            terminal = "fall_or_nonfoot_floor_contact"
            break
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    states.append(state.copy())
    m = np.asarray(metrics)
    cruise = (m[:, 0] >= 2) & (m[:, 0] <= 5)
    flight = (m[:, 8:10] < 1).all(axis=1) & (m[:, 0] > 1)
    boundaries = np.flatnonzero(np.diff(np.r_[False, flight, False]))
    flight_durations = (boundaries[1::2] - boundaries[::2]) * dt
    case = f"speed{speed:.1f}_dt{dt:g}"
    trace = output / f"{case}.npz"
    np.savez_compressed(
        trace,
        states=states,
        observations=observations,
        actions=actions,
        metrics=metrics,
    )
    return dict(
        case=case,
        requested_speed_m_s=speed,
        timestep=dt,
        terminal=terminal,
        duration_s=float(data.time),
        terminal_root_position_m=data.qpos[:3].tolist(),
        cruise_forward_velocity_m_s=float(m[cruise, 4].mean())
        if cruise.any()
        else None,
        max_lateral_drift_m=float(np.abs(m[:, 2]).max()),
        minimum_root_height_m=float(m[:, 3].min()),
        max_tilt_rad=float(m[:, 7].max()),
        max_joint_limit_excess_rad=float(m[:, 10].max()),
        max_native_force_excess_nm=float(m[:, 11].max()),
        max_artificial_support=float(m[:, 12].max()),
        flight_intervals_over_10ms=int((flight_durations > 0.01).sum()),
        max_flight_duration_s=float(flight_durations.max(initial=0)),
        native_hip_pitch_cap_nm=float(caps[0]),
        trace={"path": str(trace), **fingerprint(trace)},
    )


def run(
    upstream,
    scene,
    output,
    *,
    speeds=(0.0, 0.8, 1.5),
    heading_control=False,
    heading_gain=0.5,
    lane_control=False,
):
    torch.set_num_threads(1)
    actor = load_actor(upstream)
    output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        Path("tests/test_amp_running_probe.py"),
        upstream / "checkpoints/model_6200.pt",
        upstream / "LICENCE",
    ]
    files += sorted((upstream / "source/legged_lab/legged_lab").rglob("*.py"))
    files += sorted(p for p in scene.parent.rglob("*") if p.is_file())
    inputs = {str(p): fingerprint(p) for p in files}
    rows = []
    for speed in speeds:
        for dt in (0.001, 0.0005):
            row = run_case(
                actor,
                scene,
                output,
                speed,
                dt,
                heading_control=heading_control,
                heading_gain=heading_gain,
                lane_control=lane_control,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    for path, expected in inputs.items():
        assert fingerprint(Path(path)) == expected, path
    result = dict(
        scope="Native-plant AMP locomotion transfer only, not a bowling policy or showcase",
        upstream_commit=COMMIT,
        heading_control=heading_control,
        heading_gain=heading_gain,
        lane_control=lane_control,
        requested_speeds=list(speeds),
        inputs=inputs,
        rows=rows,
        observation_contract="495 values; six term-major five-frame histories; absolute joints; no actor normalization",
        transfer="Upstream requested PD impedance, native force caps and existing native inertia/contact model; not exact Isaac Lab reproduction",
        promotion_allowed=False,
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "scene", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--speeds",
        nargs="+",
        type=float,
        choices=(0.0, 0.8, 1.5, 2.5),
        default=[0.0, 0.8, 1.5],
    )
    parser.add_argument("--heading-control", action="store_true")
    parser.add_argument("--heading-gain", type=float, choices=(0.5, 2.0), default=0.5)
    parser.add_argument("--lane-control", action="store_true")
    args = parser.parse_args()
    run(
        args.upstream,
        args.scene,
        args.output,
        speeds=args.speeds,
        heading_control=args.heading_control,
        heading_gain=args.heading_gain,
        lane_control=args.lane_control,
    )
