"""All-feed native one-pitch batting screen with frozen GR00T balance."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np

from integrations.g1_dynamics.groot_articulated_probe import (
    GrootSwingController,
    NativeFingerServo,
    load_policy,
)
from integrations.g1_dynamics.groot_swing_timing import (
    arrival_timing,
    retime_swing_knots,
)
from integrations.g1_dynamics.twist2_cpu_probe import fingerprint
from integrations.g1_dynamics.impact_schedule import ImpactSchedule
from integrations.g1_dynamics.impact_grip import ImpactGripRecorder
from scripts.evaluate_g1_articulated_swing import run_case
from scripts.evaluate_g1_incoming_swing import BALL_COLUMNS, delivery_diagnostic
from unilab.tasks.manipulation.g1_cricket.articulated_learning import (
    ARTICULATED_DELIVERY_POOL,
    add_articulated_ball_contacts,
)
from unilab.tasks.manipulation.g1_cricket.articulated_hands import hand_contact_state
from unilab.tasks.manipulation.g1_cricket.boundary_batting import BoundaryBattingReward
from unilab.tasks.manipulation.g1_cricket.one_pitch_delivery import (
    ResetOnePitchDelivery,
    launch_vertical_speed,
)
from unilab.tasks.manipulation.g1_cricket.scene import (
    BALL_CONTACT_NAMES,
    CONTACT_SLOTS,
    CONTACT_WIDTH,
)


def incoming_scene(source):
    root = ET.parse(source).getroot()
    add_articulated_ball_contacts(root)
    sensors = root.find("sensor")
    sensors.find("contact[@name='ball_bat']").set("num", str(CONTACT_SLOTS))
    for name, kind, obj in (
        ("ball_world_position", "body", "cricket_ball"),
        ("bat_center_world", "geom", "bat_blade"),
    ):
        ET.SubElement(sensors, "framepos", name=name, objtype=kind, objname=obj)
    return ET.tostring(root, encoding="unicode")


def reset_delivery(model, data, hand, feed_index):
    speed, line, height, _ = ARTICULATED_DELIVERY_POOL[feed_index]
    ball = model.joint("ball_free")
    q, v = int(ball.qposadr[0]), int(ball.dofadr[0])
    data.qpos[q : q + 7] = [
        ResetOnePitchDelivery.start_x,
        line if hand == "right" else -line,
        height,
        1,
        0,
        0,
        0,
    ]
    vertical = launch_vertical_speed(
        speed,
        height,
        distance=ResetOnePitchDelivery.start_x - ResetOnePitchDelivery.bounce_x,
        surface_height=model.geom("pitch").pos[2] + model.geom("ball_geom").size[0],
        gravity=-model.opt.gravity[2],
    )
    data.qvel[v : v + 6] = [-speed, 0, vertical, 0, 0, 0]


class IncomingAudit:
    def __init__(self, model, *, time_based_sampling=False):
        self.model = model
        self.ball = model.geom("ball_geom").id
        self.allowed = {model.geom(name).id for name in ("pitch", "bat_blade")}
        self.ball_v = int(model.joint("ball_free").dofadr[0])
        self.width = len(BALL_CONTACT_NAMES) * CONTACT_SLOTS * CONTACT_WIDTH
        self.sensor_names = (*BALL_CONTACT_NAMES, "ball_world_position")
        env = SimpleNamespace(
            num_envs=1,
            scene=SimpleNamespace(env_origins=np.zeros((1, 3))),
            set_substep_observer=lambda *args: None,
        )
        self.score = BoundaryBattingReward(None, env)
        self.sensors, self.velocity, self.invalid = [], [], []
        self.metrics, self.invalid_metrics = [], []
        self.period = round(0.02 / model.opt.timestep)
        self.time_based_sampling = time_based_sampling
        self.sample_times = []
        self.integrator_codes = []
        self.next_sample_tick = 1

    def __call__(self, data):
        if self.time_based_sampling:
            self.sample_times.append(float(data.time))
            self.integrator_codes.append(int(self.model.opt.integrator))
        sensors = np.concatenate([data.sensor(name).data for name in self.sensor_names])
        velocity = data.qvel[self.ball_v : self.ball_v + 3].copy()
        contacts = sensors[: self.width].reshape(5, CONTACT_SLOTS, CONTACT_WIDTH)
        active = contacts[..., 0] > 0
        normal = np.where(active, contacts[..., 1], 0).sum(axis=-1)
        depth = np.maximum(0, np.where(active, -contacts[..., 7], 0).max(axis=-1))
        invalid_force, invalid_depth = 0.0, 0.0
        for index, contact in enumerate(data.contact):
            if self.ball not in contact.geom:
                continue
            other = next(int(g) for g in contact.geom if g != self.ball)
            if other not in self.allowed:
                wrench = np.empty(6)
                mujoco.mj_contactForce(self.model, data, index, wrench)
                invalid_force += max(0.0, float(wrench[0]))
                invalid_depth = max(invalid_depth, -float(contact.dist))
        self.sensors.append(sensors)
        self.velocity.append(velocity)
        self.invalid.append(invalid_force > 0 or invalid_depth > 0.006)
        self.invalid_metrics.append((invalid_force, invalid_depth))
        self.metrics.append(
            np.r_[
                sensors[self.width :],
                velocity,
                normal,
                depth,
                data.sensor("bat_center_world").data,
            ]
        )
        due = (
            data.time + 1e-10 >= self.next_sample_tick * 0.02
            if self.time_based_sampling
            else len(self.sensors) == self.period
        )
        if due:
            self.flush()
            self.next_sample_tick += 1

    def flush(self):
        if self.sensors:
            self.score.observe(
                np.asarray(self.sensors)[None],
                np.asarray(self.velocity)[None],
                invalid_contacts=np.asarray(self.invalid)[None],
            )
            self.sensors.clear()
            self.velocity.clear()
            self.invalid.clear()

    def result(self):
        self.flush()
        return {
            "delivery": delivery_diagnostic(
                np.asarray(self.metrics),
                self.model.opt.timestep,
                sample_times=self.sample_times if self.time_based_sampling else None,
            ),
            "valid_one_pitch_hit": bool(
                self.score.valid_hit[0] and not self.score.disqualified[0]
            ),
            "disqualified": bool(self.score.disqualified[0]),
            "boundary_runs": int(self.score.boundary_runs[0]),
            "separated": bool(self.score.separated[0]),
            "separation_velocity_m_s": self.score.separation_velocity_m_s[0].tolist(),
            "maximum_radius_after_hit_m": float(self.score.maximum_radius_m[0]),
            "invalid_ball_contact_force_n": float(
                np.max(self.invalid_metrics, axis=0)[0]
            ),
            "invalid_ball_contact_depth_m": float(
                np.max(self.invalid_metrics, axis=0)[1]
            ),
        }


def swing_reference(checkout, hand, delay=0.0, *, reference_file=None, swing_power=0.0):
    if not np.isfinite(delay) or not 0 <= delay <= 0.2:
        raise ValueError("Swing delay must be finite and between zero and 0.2 seconds")
    with np.load(
        checkout / f"g1_cricket_results/articulated_swing_compact/{hand}_reference.npz"
    ) as saved:
        poses, times = saved["qpos"], saved["times"]
    if reference_file is not None:
        with np.load(reference_file) as saved:
            candidate, candidate_times = saved["qpos"], saved["times"]
        if (
            candidate.shape != poses.shape
            or not np.isfinite(candidate).all()
            or not np.array_equal(candidate_times, times)
            or not np.array_equal(candidate[0], poses[0])
        ):
            raise ValueError(
                "Reference override must preserve finite poses, shape, clock and initial stance"
            )
        poses = candidate
    if delay:
        poses = np.vstack([poses[0], poses])
        times = np.r_[0.0, times + delay]
    return np.vstack([poses, poses[-1]]), retime_swing_knots(
        np.r_[times, 8.0], swing_power, delay
    )


class FingerRecoveryAudit:
    def __init__(self, model, fingers):
        self.model, self.fingers = model, fingers
        self.peak = None

    def __call__(self, data):
        if data.time < 4.5:
            return
        fingers = self.fingers
        index = int(np.argmax(np.abs(data.qvel[fingers.v])))
        velocity = float(data.qvel[fingers.v[index]])
        if self.peak is not None and abs(velocity) <= abs(
            self.peak["signed_velocity_rad_s"]
        ):
            return
        actuator = fingers.actuators[index]
        self.peak = {
            "time_s": float(data.time),
            "joint": self.model.joint(int(fingers.joints[index])).name,
            "signed_velocity_rad_s": velocity,
            "joint_position_rad": float(data.qpos[fingers.q[index]]),
            **(
                {"target_position_rad": float(data.ctrl[actuator])}
                if isinstance(fingers, NativeFingerServo)
                else {"command_nm": float(data.ctrl[actuator])}
            ),
            "actuator_force_nm": float(data.actuator_force[actuator]),
            "force_limits_nm": fingers.force_limits[index].tolist(),
            "hand_contacts": hand_contact_state(self.model, data),
            "scope": "Completed physics state, all hand-handle contacts at the maximum finger recovery speed; observation only.",
        }


def evaluate_case(job):
    (
        upstream,
        checkout,
        output,
        hand,
        feed,
        finger_preload,
        swing_delay,
        adaptive,
        coarse_substeps,
        coarse_integrator,
        native_finger_servo,
        finger_damping,
        finger_stiffness,
        reference_file,
        condition_timing,
        swing_power,
        grip_feedback_gain,
        grip_feedback_waist,
        record_impact_grip,
    ) = job
    started = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    model.opt.timestep = 0.000125
    if adaptive:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        model.pair("cricket_blade_impact_v1").solref[:] = [0.001, 0.285]
    poses, times = swing_reference(
        checkout,
        hand,
        swing_delay,
        reference_file=reference_file,
        swing_power=swing_power,
    )
    timing_record = {"mode": "fixed", "delay_s": swing_delay}
    if condition_timing:
        preview = mujoco.MjData(model)
        preview.qpos[:] = poses[0]
        reset_delivery(model, preview, hand, feed)
        timing_record = arrival_timing(model, preview, swing_delay)
        poses, times = swing_reference(
            checkout,
            hand,
            timing_record["delay_s"],
            reference_file=reference_file,
            swing_power=swing_power,
        )
    controller = GrootSwingController(
        model,
        poses,
        times,
        load_policy(upstream),
        reference_waist=True,
        root_feedback_gain=4,
        finger_preload=finger_preload,
        finger_stiffness=finger_stiffness,
        finger_damping=finger_damping,
        native_finger_servo=native_finger_servo,
        time_based_policy=adaptive,
        grip_feedback_gain=grip_feedback_gain,
        grip_feedback_waist=grip_feedback_waist,
    )
    audit = IncomingAudit(model, time_based_sampling=adaptive)
    schedule = ImpactSchedule(coarse_substeps, coarse_integrator) if adaptive else None
    recovery = FingerRecoveryAudit(model, controller.fingers) if adaptive else None
    grip = (
        ImpactGripRecorder(model, poses[0], controller.fingers)
        if record_impact_grip
        else None
    )

    def observe(data):
        audit(data)
        if recovery is not None:
            recovery(data)
        if grip is not None:
            grip(data, audit.metrics[-1][6])

    def reset(data):
        reset_delivery(model, data, hand, feed)
        if (
            condition_timing
            and arrival_timing(model, data, swing_delay) != timing_record
        ):
            raise ValueError("Actual launch differs from the timing plan")

    physical, states, controls, tactile = run_case(
        model,
        poses,
        times,
        inertial=False,
        controller=controller,
        reset=reset,
        observe=observe,
        step_schedule=schedule,
    )
    result = audit.result()
    case = f"{hand}_feed{feed}"
    trace = output / f"{case}.npz"
    timing = (
        {
            "ball_time_s": np.array(audit.sample_times),
            "integrator_code": np.array(audit.integrator_codes, dtype=np.uint8),
        }
        if adaptive
        else {}
    )
    if controller.grip_feedback is not None:
        timing["grip_feedback"] = np.asarray(
            [
                [
                    r["time_s"],
                    *r["offset_rad"],
                    r["feasible"],
                    r["position_error_m"],
                    r["rotation_error_rad"],
                ]
                for r in controller.grip_feedback.rows
            ]
        )
    if grip is not None:
        timing.update(grip.arrays())
    np.savez_compressed(
        trace,
        states=states,
        controls=controls,
        policy=controller.policy_rows,
        ball_metrics=audit.metrics,
        invalid_ball_contacts=audit.invalid_metrics,
        **timing,
    )
    contacts = output / f"{case}_contacts.json"
    contacts.write_text(json.dumps(tactile, allow_nan=False) + "\n")
    row = {
        "case": case,
        "hand": hand,
        "feed_index": feed,
        "swing_timing": timing_record,
        "swing_power": swing_power,
        "grip_feedback_gain": grip_feedback_gain,
        "grip_feedback_waist": grip_feedback_waist,
        "impact_grip": grip.report() if grip is not None else None,
        "physical": physical,
        **result,
        "qualified_hit": physical["passed"] and result["valid_one_pitch_hit"],
        "trace": {"path": str(trace), **fingerprint(trace)},
        "contacts": {"path": str(contacts), **fingerprint(contacts)},
        "wall_seconds": time.monotonic() - started,
        "integration": schedule.report() if adaptive else {"timestep_s": 0.000125},
        "finger_recovery_peak": recovery.peak if recovery is not None else None,
    }
    (output / f"{case}.json").write_text(
        json.dumps(row, indent=2, allow_nan=False) + "\n"
    )
    return row


def evaluate(
    upstream,
    checkout,
    output,
    workers,
    *,
    finger_preload=0.2,
    swing_delay=0.0,
    adaptive=False,
    finger_stiffness=2.0,
    feed_indices=None,
    coarse_substeps=1,
    coarse_integrator="RK4",
    finger_damping=0.02,
    native_finger_servo=False,
    reference_files=None,
    condition_timing=False,
    swing_power=0.0,
    grip_feedback_gain=0.0,
    grip_feedback_waist=False,
    record_impact_grip=False,
):
    if record_impact_grip and (not adaptive or not native_finger_servo):
        raise ValueError(
            "Impact grip recording requires adaptive stepping and native finger servo"
        )
    if not np.isfinite(grip_feedback_gain) or not 0 <= grip_feedback_gain <= 1:
        raise ValueError("Grip feedback gain must be finite and between zero and one")
    if not np.isfinite(swing_power) or not 0 <= swing_power <= 1:
        raise ValueError("Swing power must be finite and between zero and one")
    if finger_damping > 0.08 and not native_finger_servo:
        raise ValueError("Finger damping above 0.08 requires the native servo")
    if finger_stiffness > 3 and not native_finger_servo:
        raise ValueError("Finger stiffness above 3 requires the native servo")
    if coarse_substeps not in (1, 2) or (coarse_substeps != 1 and not adaptive):
        raise ValueError("Coarse substeps require adaptive RK4 and must be one or two")
    if coarse_integrator not in ("RK4", "implicitfast") or (
        coarse_integrator != "RK4" and not adaptive
    ):
        raise ValueError("Coarse integrator selection requires adaptive RK4")
    feeds = (
        tuple(range(len(ARTICULATED_DELIVERY_POOL)))
        if feed_indices is None
        else tuple(feed_indices)
    )
    if (
        not feeds
        or len(set(feeds)) != len(feeds)
        or any(f not in range(len(ARTICULATED_DELIVERY_POOL)) for f in feeds)
    ):
        raise ValueError("Feed indices must be unique members of the development pool")
    references = (
        dict(zip(("right", "left"), reference_files, strict=True))
        if reference_files is not None
        else {}
    )
    for hand, path in references.items():
        swing_reference(checkout, hand, swing_delay, reference_file=path)
    output.mkdir(parents=True, exist_ok=False)
    source = checkout / "g1_cricket_results/articulated_stance/scene.xml"
    scene = output / "scene.xml"
    scene.write_text(incoming_scene(source))
    parent = Path("runs/external_g1/groot_position_feedback_v3/summary.json")
    inputs = list(json.loads(parent.read_text())["source_input_fingerprints"])
    inputs += [
        str(parent),
        str(Path(__file__)),
        "tests/test_groot_incoming_probe.py",
        "tests/test_groot_articulated_probe.py",
        "integrations/g1_dynamics/groot_swing_timing.py",
        "tests/test_groot_swing_timing.py",
        "integrations/g1_dynamics/grip_feedback.py",
        "tests/test_groot_grip_feedback.py",
        "integrations/g1_dynamics/impact_grip.py",
        "tests/test_impact_grip.py",
        "runs/unilab_contact_telemetry/src/unilab/tasks/manipulation/g1_cricket/shared_bat_pose.py",
        "runs/unilab_contact_telemetry/src/unilab/tasks/manipulation/g1_cricket/strike_timing.py",
        str(scene),
        str(checkout / "scripts/evaluate_g1_incoming_swing.py"),
    ]
    xml = ET.parse(scene).getroot()
    meshdir = Path(xml.find("compiler").get("meshdir"))
    inputs += [str(meshdir / mesh.get("file")) for mesh in xml.findall("asset/mesh")]
    inputs += [str(path) for path in references.values()]
    fingerprints = {name: fingerprint(Path(name)) for name in inputs}
    if adaptive:
        for name in (
            "integrations/g1_dynamics/impact_schedule.py",
            "tests/test_impact_schedule.py",
        ):
            fingerprints[name] = fingerprint(Path(name))
        with zipfile.ZipFile(
            "runs/external_g1/groot_rk4_damping_v1/source_bundle.zip"
        ) as parent:
            names = set(parent.namelist()) | {
                "integrations/g1_dynamics/impact_schedule.py",
                "tests/test_impact_schedule.py",
                "integrations/g1_dynamics/groot_swing_timing.py",
                "tests/test_groot_swing_timing.py",
                "integrations/g1_dynamics/grip_feedback.py",
                "tests/test_groot_grip_feedback.py",
                "integrations/g1_dynamics/impact_grip.py",
                "tests/test_impact_grip.py",
                "runs/unilab_contact_telemetry/src/unilab/tasks/manipulation/g1_cricket/shared_bat_pose.py",
                "runs/unilab_contact_telemetry/src/unilab/tasks/manipulation/g1_cricket/strike_timing.py",
            }
        with zipfile.ZipFile(
            output / "source_bundle.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for name in sorted(names):
                archive.writestr(name, Path(name).read_bytes())
    jobs = [
        (
            upstream,
            checkout,
            output,
            hand,
            feed,
            finger_preload,
            swing_delay,
            adaptive,
            coarse_substeps,
            coarse_integrator,
            native_finger_servo,
            finger_damping,
            finger_stiffness,
            references.get(hand),
            condition_timing,
            swing_power,
            grip_feedback_gain,
            grip_feedback_waist,
            record_impact_grip,
        )
        for hand in ("right", "left")
        for feed in feeds
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(evaluate_case, jobs):
            rows.append(row)
            print(
                json.dumps(
                    {
                        "case": row["case"],
                        "qualified_hit": row["qualified_hit"],
                        "physical_pass": row["physical"]["passed"],
                        "boundary_runs": row["boundary_runs"],
                    }
                ),
                flush=True,
            )
    for name, expected in fingerprints.items():
        if fingerprint(Path(name)) != expected:
            raise ValueError(f"Source/input changed during trial: {name}")
    report = {
        "scope": "Frozen GR00T balance and reference swing; no new cricket RL training",
        "cohort": "Both hands, all seven one_pitch_v2 development feeds, complete eight-second cases"
        if len(feeds) == len(ARTICULATED_DELIVERY_POOL)
        else "Partial development screen, both hands, complete eight-second cases; not full-cohort evidence",
        "feed_indices": list(feeds),
        "full_cohort": len(feeds) == len(ARTICULATED_DELIVERY_POOL),
        "physics": f"Experimental blade-local RK4, coarse integrator {coarse_integrator}, blade solref [0.001, 0.285]; other native contact parameters unchanged"
        if adaptive
        else "Existing articulated batting blade/pitch pairs; otherwise retained native scene",
        "dt_s": None if adaptive else 0.000125,
        "sampling": "Actual per-step ball timestamps, 50 Hz policy/states, completed-step contact samples"
        if adaptive
        else "Fixed timestep; legacy contact sampling",
        "finger_preload_rad": finger_preload,
        "finger_stiffness_nm_per_rad": finger_stiffness,
        "finger_damping_nm_s_per_rad": finger_damping,
        "finger_control_mode": "native_position_servo"
        if native_finger_servo
        else "sampled_pd_torque",
        "finger_control_input_unit": "rad" if native_finger_servo else "Nm",
        "coarse_substeps": coarse_substeps,
        "coarse_integrator": coarse_integrator,
        "swing_delay_s": swing_delay,
        "timing_mode": "launch_velocity" if condition_timing else "fixed",
        "swing_power": swing_power,
        "grip_feedback_gain": grip_feedback_gain,
        "grip_feedback_waist": grip_feedback_waist,
        "record_impact_grip": record_impact_grip,
        "grip_feedback_scope": "Optional measured bat-relative wrist feedback at 50 Hz; arm and optional waist target offsets capped at 0.05 rad and 0.005 rad per update, unchanged native torque authority. IK residuals precede gain/rate limiting and are not achieved physical errors. Not a weld or learned control.",
        "grip_feedback_columns": [
            "time_s",
            *(
                [f"waist_offset_{i}_rad" for i in range(3)]
                if grip_feedback_waist
                else []
            ),
            *[f"arm_offset_{i}_rad" for i in range(14)],
            "ik_feasible",
            "ik_position_error_m",
            "ik_rotation_error_rad",
        ],
        "power_scope": "Shared pose knots retained; inverse StrikeTiming clock, linearly interpolated by the native controller. Not learned power or a guarantee of stronger contact.",
        "workers": workers,
        "reference_files": {hand: str(path) for hand, path in references.items()},
        "mujoco_version": mujoco.__version__,
        "boundary_radius_m": 55.0,
        "ball_columns": BALL_COLUMNS,
        "invalid_ball_contact_columns": ["normal_force_n", "penetration_m"],
        "source_input_fingerprints": fingerprints,
        "rows": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--finger-preload", type=float, choices=(0.2, 0.4, 0.6), default=0.2
    )
    parser.add_argument("--swing-delay", type=float, choices=(0.0, 0.12), default=0.0)
    parser.add_argument("--arrival-timing", action="store_true")
    parser.add_argument("--swing-power", type=float, default=0.0)
    parser.add_argument("--grip-feedback-gain", type=float, default=0.0)
    parser.add_argument("--grip-feedback-waist", action="store_true")
    parser.add_argument("--record-impact-grip", action="store_true")
    parser.add_argument("--adaptive-rk4", action="store_true")
    parser.add_argument(
        "--finger-stiffness", type=float, choices=(2.0, 3.0, 6.0), default=2.0
    )
    parser.add_argument("--feed-indices", type=int, nargs="+", choices=range(7))
    parser.add_argument("--native-finger-servo", action="store_true")
    parser.add_argument(
        "--reference-files", type=Path, nargs=2, metavar=("RIGHT", "LEFT")
    )
    parser.add_argument(
        "--finger-damping", type=float, choices=(0.02, 0.08, 0.2, 0.3), default=0.02
    )
    parser.add_argument("--coarse-substeps", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--coarse-integrator", choices=("RK4", "implicitfast"), default="RK4"
    )
    args = parser.parse_args()
    evaluate(
        args.upstream,
        args.checkout,
        args.output,
        args.workers,
        finger_preload=args.finger_preload,
        swing_delay=args.swing_delay,
        adaptive=args.adaptive_rk4,
        finger_stiffness=args.finger_stiffness,
        finger_damping=args.finger_damping,
        native_finger_servo=args.native_finger_servo,
        reference_files=args.reference_files,
        condition_timing=args.arrival_timing,
        swing_power=args.swing_power,
        grip_feedback_gain=args.grip_feedback_gain,
        grip_feedback_waist=args.grip_feedback_waist,
        record_impact_grip=args.record_impact_grip,
        feed_indices=args.feed_indices,
        coarse_substeps=args.coarse_substeps,
        coarse_integrator=args.coarse_integrator,
    )
