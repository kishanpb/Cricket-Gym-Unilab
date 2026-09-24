"""Attribute the retained reach failures without changing controls or selecting rows."""

import json

import mujoco
import numpy as np
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import sha256
from probe_g1_cricket_overarm import DIRECTORY, PLAN, owner_config, reach_trial, sources
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env, runtime_sources


def phase(tick):
    for end, name in (
        (10, "neutral"),
        (30, "settle"),
        (80, "raise"),
        (120, "hold"),
        (170, "recover"),
    ):
        if tick < end:
            return name
    return "final_settle"


class FailureAudit:
    def __init__(self, env):
        m = env.get_playback_model()
        self.selected = set(env.action_manager.get_term("residual").arm_ids.tolist())
        self.joints = m.actuator_trnid[:, 0]
        self.qadr, self.vadr = m.jnt_qposadr[self.joints], m.jnt_dofadr[self.joints]
        self.limits = m.jnt_range[self.joints]
        self.pitch = m.geom("pitch").id
        self.feet = {m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
        self.first_joint = self.worst_joint = self.first_contact = None
        self.context = {}
        self.samples = 0

    def __call__(self, tick, m, d):
        self.samples += 1
        if tick not in self.context:
            self.context[tick] = dict(
                tick=tick,
                time=float(d.time),
                phase=phase(tick),
                pelvis_up=float(1 - 2 * np.square(d.qpos[4:6]).sum()),
                root_qpos=d.qpos[:7].tolist(),
                root_qvel=d.qvel[:6].tolist(),
                targets=d.ctrl.tolist(),
            )
        q = d.qpos[self.qadr]
        excess = np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1])
        k = int(excess.argmax())
        if excess[k] > 1e-6 and (
            self.worst_joint is None or excess[k] > self.worst_joint["excess_rad"]
        ):
            event = dict(
                tick=tick,
                phase=phase(tick),
                time=float(d.time),
                joint=m.joint(self.joints[k]).name,
                selected_arm=k in self.selected,
                q=float(q[k]),
                qvel=float(d.qvel[self.vadr[k]]),
                limits=self.limits[k].tolist(),
                target=float(d.ctrl[k]),
                torque_nm=float(d.actuator_force[k]),
                excess_rad=float(excess[k]),
            )
            if self.first_joint is None:
                self.first_joint = event
            self.worst_joint = event
        if self.first_contact is not None:
            return
        for index, contact in enumerate(d.contact):
            if contact.efc_address < 0:
                continue
            geoms = list(map(int, contact.geom))
            if self.pitch in geoms:
                other = geoms[1] if geoms[0] == self.pitch else geoms[0]
                if m.geom_bodyid[other] in self.feet:
                    continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, force)
            self.first_contact = dict(
                tick=tick,
                phase=phase(tick),
                solved_time=float(d.time - m.opt.timestep),
                geoms=[m.geom(g).name for g in geoms],
                bodies=[m.body(m.geom_bodyid[g]).name for g in geoms],
                distance_m=float(contact.dist),
                normal_force_n=float(force[0]),
            )
            break

    def result(self):
        events = [
            x for x in (self.first_joint, self.worst_joint, self.first_contact) if x is not None
        ]
        ticks = {t for event in events for t in range(event["tick"] - 2, event["tick"] + 3)}
        return dict(
            substeps=self.samples,
            first_joint=self.first_joint,
            worst_joint=self.worst_joint,
            first_forbidden_contact=self.first_contact,
            context=[v for k, v in sorted(self.context.items()) if k in ticks],
            joint_state_phase="integrated",
            contact_phase="preceding_solved_state",
            context_phase="first_integrated_substep_of_control_interval",
        )


def audit_inputs():
    files = sources()
    for name in (
        "scripts/audit_g1_cricket_overarm_failures.py",
        "tests/scripts/test_g1_cricket_overarm_failures.py",
    ):
        files[name] = sha256(ROOT / name)
    return dict(
        sources=files,
        parent_sha256=sha256(DIRECTORY / "evaluation.json"),
        runtime_sources=runtime_sources(),
        executor=executor_manifest(),
    )


def run():
    output = DIRECTORY / "failure_attribution.json"
    if output.exists():
        raise FileExistsError("inspect retained failure attribution instead of restarting")
    inputs = audit_inputs()
    parents = json.loads((DIRECTORY / "evaluation.json").read_text())["rows"]
    rows = []
    for parent in parents:
        env = make_env(owner_config(), parent["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
        try:
            observer = FailureAudit(env)
            result = reach_trial(env, parent["pitch"], parent["elbow"], observer=observer)
            if result != parent:
                raise ValueError("failure diagnostic changed a retained reach outcome")
            row = dict(
                hand=parent["hand"],
                pitch=parent["pitch"],
                elbow=parent["elbow"],
                **observer.result(),
            )
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "context"}), flush=True)
        finally:
            env.close()
    if inputs != audit_inputs():
        raise ValueError("failure audit inputs changed")
    write_json(output, dict(inputs=inputs, rows=rows, exact_parent_rows=True, new_training=False))


if __name__ == "__main__":
    run()
