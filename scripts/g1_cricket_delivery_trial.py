"""Independent full-episode robot delivery gate, not an umpiring certification."""

import mujoco
import numpy as np

from unilab.tasks.manipulation.g1_cricket.delivery import POPPING_X, RETURN_Y, TARGET_POPPING_X


def capsule_bounds(centres, rotations, sizes, kinds):
    radii = sizes[:, 0, None]
    half = np.where(kinds == mujoco.mjtGeom.mjGEOM_CAPSULE, sizes[:, 1], 0)
    extent = np.abs(rotations[:, :, 2]) * half[:, None] + radii
    return np.stack(((centres - extent).min(axis=0), (centres + extent).max(axis=0)))


def arm_geometry(shoulder, elbow, wrist):
    upper, forearm = shoulder - elbow, wrist - elbow
    lengths = np.linalg.norm(upper), np.linalg.norm(forearm)
    if min(lengths) <= 1e-8:
        raise ValueError("degenerate delivery arm")
    angle = np.arccos(np.clip(np.dot(upper, forearm) / np.prod(lengths), -1, 1))
    return float(-upper[2] / lengths[0]), float(angle)


def feet_failures(front, back, side, release_time):
    if front is None or back is None:
        return {"missing_delivery_stride"}
    failures = set()
    if not (release_time - 1 < back["time"] < front["time"] <= release_time):
        failures.add("invalid_delivery_stride_order")
    if release_time - front["time"] > 0.5:
        failures.add("stale_front_foot_landing")
    f, b = np.array(front["bounds"]), np.array(back["bounds"])
    if f[0, 0] >= POPPING_X:
        failures.add("front_foot_over_popping_crease")
    if not (b[0, 1] > -RETURN_Y and b[1, 1] < RETURN_Y):
        failures.add("back_foot_outside_return_crease")
    if (side > 0 and min(f[0, 1], b[0, 1]) <= 0) or (side < 0 and max(f[1, 1], b[1, 1]) >= 0):
        failures.add("feet_cross_declared_wicket_side")
    return failures


class DeliveryEvents:
    def __init__(self, hand):
        self.hand, self.side = hand, 1 if hand == "right" else -1
        self.front = "left" if hand == "right" else "right"
        self.failures = set()
        self.landings = {"left": [], "right": []}
        self.airborne_since = {"left": None, "right": None}
        self.landing_armed = {"left": False, "right": False}
        self.horizontal_time = None
        self.previous_upper = None
        self.minimum_angle = np.inf
        self.maximum_extension = 0.0
        self.release_record = None
        self.first_bounce = self.crossing = None
        self.pitch_previous = False
        self.bounces_before_target = 0
        self.ball_contacts = {}
        self.ball_peak_forces = {}
        self.peak_penetration = 0.0
        self.height, self.up = np.inf, np.inf
        self.limit_excess = self.force_fraction = 0.0
        self.peak_holder_force = 0.0
        self.holder_impulse = np.zeros(3)
        self.previous_ball = None

    def observe_arm(self, time, upper_z, angle):
        if self.release_record is not None:
            return
        if (
            self.horizontal_time is None
            and self.previous_upper is not None
            and self.previous_upper < 0 <= upper_z
        ):
            self.horizontal_time = time
        self.previous_upper = upper_z
        if self.horizontal_time is not None:
            self.minimum_angle = min(self.minimum_angle, angle)
            self.maximum_extension = max(self.maximum_extension, angle - self.minimum_angle)

    def observe_support(self, time, side, loaded, bounds):
        if not loaded and bounds[0, 2] > 0.002:
            if self.airborne_since[side] is None:
                self.airborne_since[side] = time
            if time - self.airborne_since[side] >= 0.02:
                self.landing_armed[side] = True
        else:
            self.airborne_since[side] = None
        if self.release_record is None and loaded and self.landing_armed[side] and time > 0.2:
            self.landings[side].append({"time": time, "bounds": bounds.tolist()})
        if loaded:
            self.landing_armed[side] = False

    def release(self, time, position, velocity, shoulder_z, upper_z, angle):
        if self.release_record is not None:
            raise ValueError("second release in one episode")
        self.observe_arm(time, upper_z, angle)
        front = self.landings[self.front][-1] if self.landings[self.front] else None
        back = self.landings[self.hand][-1] if self.landings[self.hand] else None
        self.failures.update(feet_failures(front, back, self.side, time))
        for name, bad in (
            ("no_upward_shoulder_level_crossing", self.horizontal_time is None),
            ("elbow_extension_above_15_degrees", self.maximum_extension > np.deg2rad(15)),
            ("release_not_overarm", upper_z < 0 or position[2] < shoulder_z + 0.12),
            ("release_forward_speed_not_above_6_m_s", velocity[0] <= 6),
            ("release_lateral_speed_above_2_m_s", abs(velocity[1]) > 2),
        ):
            if bad:
                self.failures.add(name)
        self.release_record = dict(
            time=time,
            position=position.tolist(),
            velocity=velocity.tolist(),
            shoulder_z=shoulder_z,
            upper_z=upper_z,
            elbow_angle_rad=angle,
            maximum_elbow_extension_rad=self.maximum_extension,
            horizontal_time=self.horizontal_time,
            front_landing=front,
            back_landing=back,
        )

    def observe_ball(self, time, position, contacts):
        pitch = False
        for name, distance, force in contacts:
            self.ball_contacts[name] = self.ball_contacts.get(name, 0) + 1
            self.ball_peak_forces[name] = max(self.ball_peak_forces.get(name, 0.0), force)
            self.peak_penetration = max(self.peak_penetration, -distance)
            pitch |= name == "pitch"
            if self.release_record is None or name not in {
                "pitch",
                "wicket_0",
                "wicket_1",
                "wicket_2",
            }:
                self.failures.add(f"ball_contact:{name}")
        if (
            pitch
            and not self.pitch_previous
            and self.release_record is not None
            and self.crossing is None
        ):
            self.bounces_before_target += 1
            if self.first_bounce is None:
                self.first_bounce = dict(time=time, position=position.tolist())
        self.pitch_previous = pitch
        if (
            self.release_record is not None
            and self.crossing is None
            and self.previous_ball is not None
        ):
            old = self.previous_ball
            if old[0] < TARGET_POPPING_X <= position[0]:
                fraction = (TARGET_POPPING_X - old[0]) / (position[0] - old[0])
                self.crossing = dict(
                    time=time, position=(old + fraction * (position - old)).tolist()
                )
        self.previous_ball = position.copy()

    def finish(self, complete):
        failures = self.failures.copy()
        bounce = self.first_bounce
        crossing = self.crossing
        for name, bad in (
            ("episode_incomplete", not complete),
            ("no_release", self.release_record is None),
            ("pelvis_height", self.height < 0.48),
            ("pelvis_orientation", self.up < 0.65),
            ("joint_limit", self.limit_excess > 1e-6),
            ("actuator_limit", self.force_fraction > 1 + 1e-6),
            ("ball_penetration_above_6_mm", self.peak_penetration > 0.006),
            (
                "first_bounce_outside_delivery_zone",
                bounce is None
                or not (
                    4 < bounce["position"][0] < TARGET_POPPING_X
                    and abs(bounce["position"][1]) < RETURN_Y
                ),
            ),
            ("not_exactly_one_bounce_before_target", self.bounces_before_target != 1),
            (
                "target_corridor_missed",
                crossing is None
                or not (
                    abs(crossing["position"][1]) <= 0.5 and 0.04 < crossing["position"][2] <= 1.2
                ),
            ),
        ):
            if bad:
                failures.add(name)
        return dict(
            passed=not failures,
            failures=sorted(failures),
            release=self.release_record,
            first_bounce=bounce,
            target_crossing=crossing,
            bounces_before_target=self.bounces_before_target,
            ball_contact_counts=self.ball_contacts,
            ball_contact_peak_force_n=self.ball_peak_forces,
            maximum_ball_penetration_m=self.peak_penetration,
            minimum_pelvis_height_m=self.height,
            minimum_pelvis_up=self.up,
            maximum_joint_limit_excess_rad=self.limit_excess,
            maximum_actuator_limit_fraction=self.force_fraction,
            peak_holder_force_n=self.peak_holder_force,
            holder_impulse_world_ns=self.holder_impulse.tolist(),
        )


class DeliveryReplay:
    def __init__(self, env):
        self.model = env.get_playback_model()
        self.data, self.pose = mujoco.MjData(self.model), mujoco.MjData(self.model)
        m = self.model
        self.steps = env.cfg.sim_substeps
        self.sensors = env.scene.bind_sensor_data(tuple(m.sensor(i).name for i in range(m.nsensor)))
        self.foot_geoms = {}
        for side in ("left", "right"):
            ids = np.flatnonzero(
                (m.geom_bodyid == m.body(f"{side}_ankle_roll_link").id)
                & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
            )
            if (
                len(ids) != 11
                or not np.isin(
                    m.geom_type[ids], [mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE]
                ).all()
            ):
                raise ValueError("unexpected G1 foot collision geometry")
            self.foot_geoms[side] = ids
        self.foot_sets = {side: set(ids.tolist()) for side, ids in self.foot_geoms.items()}
        self.ball = m.geom("ball_geom").id
        self.pitch = m.geom("pitch").id
        self.ball_qadr = m.jnt_qposadr[m.body("cricket_ball").jntadr[0]]
        self.ball_vadr = m.jnt_dofadr[m.body("cricket_ball").jntadr[0]]
        self.arm = [
            m.body(f"{env.cfg.handedness}_{name}_link").id
            for name in ("shoulder_roll", "elbow", "wrist_roll")
        ]
        self.joints = m.actuator_trnid[:, 0]
        self.joint_qadr = m.jnt_qposadr[self.joints]
        self.site = m.site("holder_site").id
        self.holder_adr = m.sensor("holder_force").adr[0]

    def step(self, env, action, events, *, observer=None):
        m, d = self.model, self.data
        initial = env.get_physics_state_snapshot()[0].copy()
        state = env.step(action)
        term = env.action_manager.get_term("residual")
        mujoco.mj_resetData(m, d)
        mujoco.mj_setState(m, d, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.eq_active[:] = env.equality_constraints.get_equality_active()[0]
        d.ctrl[:] = term.processed_action[0]
        if term.just_released[0]:
            mujoco.mj_setState(m, self.pose, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_kinematics(m, self.pose)
            upper, angle = arm_geometry(*self.pose.xpos[self.arm])
            events.release(
                float(initial[0]),
                initial[1 + self.ball_qadr : 4 + self.ball_qadr],
                initial[1 + m.nq + self.ball_vadr : 4 + m.nq + self.ball_vadr],
                float(self.pose.xpos[self.arm[0], 2]),
                upper,
                angle,
            )
        for _ in range(self.steps):
            mujoco.mj_step(m, d)
            if observer is not None:
                observer(m, d)
            solved_time = float(d.time - m.opt.timestep)
            upper, angle = arm_geometry(*d.xpos[self.arm])
            events.observe_arm(solved_time, upper, angle)
            loaded = {"left": False, "right": False}
            ball_contacts = []
            for i, contact in enumerate(d.contact):
                if contact.efc_address < 0:
                    continue
                g1, g2 = map(int, contact.geom)
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, wrench)
                for side, geoms in self.foot_sets.items():
                    if self.pitch in (g1, g2) and (g1 in geoms or g2 in geoms) and wrench[0] > 1:
                        loaded[side] = True
                if self.ball in (g1, g2):
                    other = g2 if g1 == self.ball else g1
                    ball_contacts.append(
                        (m.geom(other).name, float(contact.dist), float(np.linalg.norm(wrench[:3])))
                    )
                elif self.pitch in (g1, g2):
                    other = g2 if g1 == self.pitch else g1
                    if not any(other in geoms for geoms in self.foot_sets.values()):
                        events.failures.add("nonfoot_ground_contact")
                else:
                    events.failures.add("robot_self_or_wicket_contact")
            for side, ids in self.foot_geoms.items():
                bounds = capsule_bounds(
                    d.geom_xpos[ids],
                    d.geom_xmat[ids].reshape(-1, 3, 3),
                    m.geom_size[ids],
                    m.geom_type[ids],
                )
                events.observe_support(solved_time, side, loaded[side], bounds)
            events.observe_ball(solved_time, d.geom_xpos[self.ball], ball_contacts)
            events.height = min(events.height, float(d.qpos[2]))
            events.up = min(events.up, float(1 - 2 * np.square(d.qpos[4:6]).sum()))
            q = d.qpos[self.joint_qadr]
            events.limit_excess = max(
                events.limit_excess,
                float(
                    np.maximum(
                        m.jnt_range[self.joints, 0] - q, q - m.jnt_range[self.joints, 1]
                    ).max()
                ),
            )
            events.force_fraction = max(
                events.force_fraction,
                float((np.abs(d.actuator_force) / m.actuator_forcerange[:, 1]).max()),
            )
            force = d.sensordata[self.holder_adr : self.holder_adr + 3]
            events.peak_holder_force = max(events.peak_holder_force, float(np.linalg.norm(force)))
            events.holder_impulse += d.site_xmat[self.site].reshape(3, 3) @ force * m.opt.timestep
        expected = np.empty_like(initial, dtype=np.float64)
        mujoco.mj_getState(m, d, expected, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        actual = env.get_physics_state_snapshot()[0]
        np.testing.assert_array_equal(expected.astype(actual.dtype), actual)
        native_sensors = self.sensors.read()[0]
        np.testing.assert_array_equal(d.sensordata.astype(native_sensors.dtype), native_sensors)
        if (
            d.warning.number.any()
            or not np.isfinite(expected).all()
            or not np.isfinite(d.sensordata).all()
        ):
            raise RuntimeError("invalid delivery replay")
        return state


def trial(env, replay, action_at, seed):
    env.reset(seed=seed)
    events = DeliveryEvents(env.cfg.handedness)
    total = 0.0
    for tick in range(env.max_episode_length):
        state = replay.step(env, action_at(tick), events)
        total += float(state.reward[0])
        if state.terminated[0] or state.truncated[0]:
            break
    result = events.finish(bool(state.truncated[0] and not state.terminated[0]))
    result.update(steps=tick + 1, episode_return=total, exact_endpoint_replay=True)
    return result
