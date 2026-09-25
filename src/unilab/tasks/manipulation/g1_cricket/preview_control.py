"""Short native-physics motor preview; prediction is isolated from the live robot."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from mujoco import rollout

from .contact_control import ContactAccelerationControl
from .running import RELEASE_TIME


def build_preview_model(source, destination, timestep):
    tree = ET.parse(source)
    root = tree.getroot()
    for sensors in root.findall("sensor"):
        root.remove(sensors)
    sensors = ET.SubElement(root, "sensor")
    for side in ("left", "right"):
        ET.SubElement(
            sensors,
            "framepos",
            name=f"preview_{side}_foot",
            objtype="xbody",
            objname=f"{side}_ankle_roll_link",
        )
    for left, right in (("thigh", "thigh"), ("thigh", "shin"), ("shin", "thigh"), ("shin", "shin")):
        ET.SubElement(
            sensors,
            "distance",
            name=f"preview_{left}_{right}",
            geom1=f"left_{left}_collision",
            geom2=f"right_{right}_collision",
            cutoff="0.05",
        )
    ET.SubElement(
        sensors,
        "distance",
        name="preview_feet",
        body1="left_ankle_roll_link",
        body2="right_ankle_roll_link",
        cutoff="0.05",
    )
    tree.write(destination)
    model = mujoco.MjModel.from_xml_path(str(destination))
    model.opt.timestep = timestep
    return model


class PreviewControl:
    def __init__(self, model, preview_model, reference, baseline, *, seed=1, samples=24, horizon=6):
        self.model, self.preview = model, preview_model
        self.poses, self.velocity = reference["qpos"].copy(), reference["qvel"].copy()
        self.anchor = ContactAccelerationControl(
            model, self.velocity, baseline, foot_reference=self.poses
        )
        self.qa, self.va = self.anchor.qa, self.anchor.va
        self.limits = self.anchor.limits
        self.samples, self.horizon = samples, horizon
        self.substeps = round(0.02 / model.opt.timestep)
        self.rng = np.random.default_rng(seed)
        self.pool = rollout.Rollout(nthread=2)
        self.workers = [mujoco.MjData(preview_model) for _ in range(2)]
        self.control_spec = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE
        self.plan = None
        self.trace = []
        self.nominal = []
        data = mujoco.MjData(model)
        for pose, velocity in zip(self.poses, self.velocity, strict=True):
            data.qpos[:], data.qvel[:] = pose, velocity
            mujoco.mj_forward(model, data)
            command, _ = baseline(model, data, pose, velocity, self.qa, self.va)
            self.nominal.append(np.clip(command, self.limits[:, 0], self.limits[:, 1]))
        self.nominal = np.asarray(self.nominal)

    def close(self):
        self.pool.close()

    def predict(self, data, plans):
        initial = np.empty(mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.model, data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        commands = np.repeat(plans, self.substeps, axis=1)
        steps = commands.shape[1]
        active = (
            data.time + np.arange(steps) * self.model.opt.timestep < RELEASE_TIME - 1e-9
        ).astype(float)
        active *= bool(data.eq_active[self.model.equality("ball_holder").id])
        controls = np.concatenate(
            (commands, np.broadcast_to(active[None, :, None], (*commands.shape[:2], 1))), axis=2
        )
        return self.pool.rollout(
            self.preview,
            self.workers,
            initial[None],
            control=controls,
            control_spec=self.control_spec,
            initial_warmstart=data.qacc_warmstart[None],
        )

    def costs(self, data, plans):
        state, sensors = self.predict(data, plans)
        nq = self.model.nq
        poses = state[..., 1 : 1 + nq]
        sample = np.arange(self.substeps - 1, state.shape[1], self.substeps)
        frame = int((data.time + 1e-9) / 0.02)
        target_frames = np.minimum(frame + np.arange(1, self.horizon + 1), len(self.poses) - 1)
        target = self.poses[target_frames]
        observed = poses[:, sample]
        position = np.sum((observed[..., :3] - target[None, :, :3]) ** 2, axis=-1)
        dot = np.sum(observed[..., 3:7] * target[None, :, 3:7], axis=-1)
        orientation = 1 - np.clip(dot**2, 0, 1)
        joint = np.mean((observed[..., self.qa] - target[None, :, self.qa]) ** 2, axis=-1)
        feet = sensors[:, sample, :6].reshape(len(plans), self.horizon, 2, 3)
        foot_error = np.sum(
            (feet - self.anchor.foot_positions[target_frames][None]) ** 2, axis=(-1, -2)
        )
        cost = np.mean(200 * position + 40 * orientation + 2 * joint + 100 * foot_error, axis=1)
        excess = np.maximum(
            self.limits[:, 0] - poses[..., self.qa], poses[..., self.qa] - self.limits[:, 1]
        )
        peak_limit = np.maximum(excess.max(axis=(1, 2)), 0)
        minimum_leg_clearance = sensors[..., 6:].min(axis=(1, 2))
        minimum_height = poses[..., 2].min(axis=1)
        stalled = np.any(np.diff(state[..., 0], axis=1) <= 0, axis=1)
        nonfinite = ~np.isfinite(state).all(axis=(1, 2)) | ~np.isfinite(sensors).all(axis=(1, 2))
        unsafe = (
            (peak_limit > 1e-6)
            | (minimum_leg_clearance < -0.001)
            | (minimum_height < 0.48)
            | stalled
            | nonfinite
        )
        cost += 1e4 * unsafe + 1e4 * peak_limit**2 + 1e4 * np.minimum(minimum_leg_clearance, 0) ** 2
        cost[nonfinite] = 1e12
        return cost, {
            "costs": cost.tolist(),
            "predicted_unsafe": unsafe.tolist(),
            "peak_joint_limit_excess_rad": peak_limit.tolist(),
            "minimum_leg_clearance_m": minimum_leg_clearance.tolist(),
            "minimum_pelvis_height_m": minimum_height.tolist(),
        }

    def __call__(self, data, target, velocity):
        frame = int((data.time + 1e-9) / 0.02)
        indices = np.minimum(frame + np.arange(self.horizon), len(self.poses) - 1)
        nominal = self.nominal[indices]
        anchor = self.anchor(data, target, velocity)
        anchored = np.clip(nominal + anchor - nominal[0], self.limits[:, 0], self.limits[:, 1])
        center = anchored if self.plan is None else np.vstack((self.plan[1:], nominal[-1]))
        stages = []
        interpolation = np.linspace(0, 1, self.horizon)[None, :, None]
        for scale in (0.12, 0.06):
            knots = self.rng.normal(0, scale, (self.samples - 3, 2, self.model.nu))
            noise = (1 - interpolation) * knots[:, :1] + interpolation * knots[:, 1:]
            candidates = np.concatenate(
                (center[None], nominal[None], anchored[None], center[None] + noise)
            )
            candidates = np.clip(candidates, self.limits[:, 0], self.limits[:, 1])
            costs, audit = self.costs(data, candidates)
            selected = int(np.argmin(costs))
            center = candidates[selected]
            stages.append({"selected": selected, **audit})
        self.plan = center.copy()
        self.trace.append({"time_s": float(data.time), "stages": stages})
        return center[0].copy()
