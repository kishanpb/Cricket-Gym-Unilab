"""Additional signed-extension gate; never clears an existing delivery failure."""

import mujoco
import numpy as np
from g1_cricket_delivery_trial import DeliveryReplay, arm_geometry

from unilab.tasks.manipulation.g1_cricket.arm_geometry import signed_elbow_angle
from unilab.tasks.manipulation.g1_cricket.bowling import RELEASE_THRESHOLD


class SignedElbowAudit:
    def __init__(self):
        self.previous_upper = None
        self.horizontal_time = None
        self.minimum_angle = np.inf
        self.maximum_extension = 0.0
        self.maximum_angle = -np.inf
        self.release_record = None
        self.samples = 0
        self.previous_angle = None
        self.unwrapped_angle = 0.0

    def observe(self, time, upper, angle):
        if self.release_record is not None:
            return
        self.samples += 1
        if self.previous_angle is None:
            self.unwrapped_angle = angle
        else:
            difference = angle - self.previous_angle
            self.unwrapped_angle += np.arctan2(np.sin(difference), np.cos(difference))
        self.previous_angle = angle
        if self.horizontal_time is None and self.previous_upper is not None:
            if self.previous_upper < 0 <= upper:
                self.horizontal_time = time
        self.previous_upper = upper
        if self.horizontal_time is not None:
            self.minimum_angle = min(self.minimum_angle, self.unwrapped_angle)
            self.maximum_angle = max(self.maximum_angle, self.unwrapped_angle)
            self.maximum_extension = max(
                self.maximum_extension, self.unwrapped_angle - self.minimum_angle
            )

    def release(self, time, upper, angle):
        self.observe(time, upper, angle)
        self.release_record = dict(
            time=time,
            upper_z=upper,
            signed_elbow_angle_rad=angle,
            unwrapped_elbow_angle_rad=float(self.unwrapped_angle),
        )

    def result(self, legacy):
        failures = set(legacy["failures"])
        if legacy["release"] is not None:
            if self.release_record is None:
                failures.add("missing_signed_elbow_release_evidence")
            if self.horizontal_time is None:
                failures.add("no_signed_elbow_shoulder_level_crossing")
            if self.maximum_extension > np.deg2rad(15):
                failures.add("signed_elbow_extension_above_15_degrees")
        return dict(
            version="signed_elbow_v1_additional_gate",
            passed=not failures,
            failures=sorted(failures),
            release=self.release_record,
            horizontal_time=self.horizontal_time,
            maximum_extension_rad=self.maximum_extension,
            maximum_angle_rad=None if self.horizontal_time is None else self.maximum_angle,
            samples_through_release=self.samples,
            legacy_failures_preserved=set(legacy["failures"]).issubset(failures),
            scope="simulated_G1_hinge_extension_not_umpiring_certification",
        )


class SignedDeliveryReplay(DeliveryReplay):
    def __init__(self, env):
        super().__init__(env)
        self.elbow_joint = self.model.joint(f"{env.cfg.handedness}_elbow_joint").id
        self.proximal = self.model.body(f"{env.cfg.handedness}_shoulder_yaw_link").id
        self.audit = SignedElbowAudit()

    def sample(self, data, time, *, release=False):
        shoulder, elbow, wrist = data.xpos[self.arm]
        upper, _ = arm_geometry(shoulder, elbow, wrist)
        # Use a landmark rigidly attached to the elbow's parent, not across shoulder yaw.
        angle = signed_elbow_angle(
            data.xpos[self.proximal], elbow, wrist, data.xaxis[self.elbow_joint]
        )
        if release:
            self.audit.release(time, upper, angle)
        else:
            self.audit.observe(time, upper, angle)

    def step(self, env, action, events, *, observer=None):
        initial = env.get_physics_state_snapshot()[0]
        mujoco.mj_setState(self.model, self.pose, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_kinematics(self.model, self.pose)
        term = env.action_manager.get_term("residual")
        releasing = not term.released[0] and action[0, 7] > RELEASE_THRESHOLD
        self.sample(self.pose, float(initial[0]), release=releasing)

        def observe(m, d):
            if observer is not None:
                observer(m, d)
            self.sample(d, float(d.time - m.opt.timestep))

        return super().step(env, action, events, observer=observe)
