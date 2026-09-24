"""Fixed positive-shoulder launch/braking study with unchanged motor authority."""

import itertools
import json
from copy import deepcopy

import numpy as np
from evaluate_g1_cricket_residual import sha256
from probe_g1_cricket_overarm import smooth, target_at
from probe_g1_cricket_overarm_guard import GuardAudit, owner_config
from search_g1_cricket_shoulder import DIRECTORY as PARENT
from search_g1_cricket_shoulder import inputs as search_inputs
from search_g1_cricket_shoulder import trajectory_trial
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, make_env

DIRECTORY = ROOT / "g1_cricket_results/positive_arc_v1"
PLAN = dict(
    hand="left",
    seed=6301,
    dt=0.0000625,
    engine="mjbatch",
    preload_pitch=1.0,
    drive_pitch=2.6,
    brake_pitch=1.6,
    drive_ticks=[94, 98, 102],
    brake_ticks=[112, 114],
    release_tick=114,
    recovery=[150, 190],
    rows=6,
    new_training=False,
    policy_promoted=False,
)


def candidates():
    return list(itertools.product(PLAN["drive_ticks"], PLAN["brake_ticks"]))


def phase(tick, candidate):
    drive, brake = candidate
    for end, name in (
        (10, "neutral"),
        (30, "settle"),
        (80, "preload"),
        (drive, "hold"),
        (brake, "drive"),
        (150, "brake"),
        (190, "recover"),
    ):
        if tick < end:
            return name
    return "final_settle"


def launch_target(neutral, candidate, tick):
    drive, brake = candidate
    if tick < drive:
        return target_at(neutral, PLAN["hand"], PLAN["preload_pitch"], 1.4, tick)
    target = np.array(
        [PLAN["drive_pitch"] if tick < brake else PLAN["brake_pitch"], 0.35, 0, 1.4, 0, 0, 0]
    )
    return target + smooth(tick, *PLAN["recovery"]) * (neutral - target)


def corrected_attribution(result, candidate):
    result = deepcopy(result)
    result["first_contact_excluding_foot_pitch"] = result.pop("first_forbidden_contact")
    events = [
        result[key] for key in ("first_joint", "worst_joint", "first_contact_excluding_foot_pitch")
    ]
    for event in events + result["context"]:
        if event is not None:
            event["phase"] = phase(event["tick"], candidate)
    result["contact_scope"] = (
        "first_contact_excluding_foot_pitch_not_a_violation_gate_use_full_outcome"
    )
    return result


class LaunchAudit(GuardAudit):
    def __init__(self, env, candidate):
        super().__init__(env)
        self.candidate = candidate
        self.index = int(self.term.arm_ids[0])
        self.maximum_q = -np.inf
        self.maximum_forward_qvel = -np.inf
        self.previous_qvel = None
        self.turning_point = None
        self.torques = {}

    def __call__(self, tick, m, d):
        super().__call__(tick, m, d)
        q, qvel = float(d.qpos[self.qadr[self.index]]), float(d.qvel[self.vadr[self.index]])
        self.maximum_q = max(self.maximum_q, q)
        self.maximum_forward_qvel = max(self.maximum_forward_qvel, qvel)
        if (
            tick >= self.candidate[1]
            and self.turning_point is None
            and self.previous_qvel is not None
            and self.previous_qvel > 0 >= qvel
        ):
            self.turning_point = dict(tick=tick, time=float(d.time), q=q, qvel=qvel)
        self.previous_qvel = qvel
        torque = float(d.actuator_force[self.index])
        stats = self.torques.setdefault(
            phase(tick, self.candidate),
            dict(minimum_nm=torque, maximum_nm=torque, saturated_substeps=0, substeps=0),
        )
        stats["minimum_nm"] = min(stats["minimum_nm"], torque)
        stats["maximum_nm"] = max(stats["maximum_nm"], torque)
        stats["saturated_substeps"] += int(
            abs(torque) >= m.actuator_forcerange[self.index, 1] - 1e-6
        )
        stats["substeps"] += 1

    def result(self):
        result = corrected_attribution(super().result(), self.candidate)
        result["shoulder_braking"] = dict(
            maximum_q_rad=self.maximum_q,
            minimum_positive_stop_margin_rad=float(self.limits[self.index, 1] - self.maximum_q),
            maximum_forward_qvel_rad_s=self.maximum_forward_qvel,
            first_post_brake_turning_point=self.turning_point,
            torque_by_phase=self.torques,
            state_phase="integrated_q_and_qvel_torque_from_preceding_solve",
        )
        return result


def inputs():
    record = search_inputs()
    for name in (
        "scripts/probe_g1_cricket_positive_arc.py",
        "tests/scripts/test_g1_cricket_positive_arc.py",
        "docs/g1_cricket_positive_arc_v1.md",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record.pop("initial_population")
    record.update(plan=PLAN, parent_shoulder_search_sha256=sha256(PARENT / "evaluation.json"))
    return record


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained positive-arc study instead of restarting")
    record = inputs()
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", record)
    rows = []
    env = make_env(owner_config(), PLAN["hand"], dt=PLAN["dt"], engine=PLAN["engine"])
    try:
        for candidate in candidates():
            audits = []

            def factory(env):
                audit = LaunchAudit(env, candidate)
                audits.append(audit)
                return audit

            row = trajectory_trial(
                env,
                candidate,
                target_fn=launch_target,
                seed=PLAN["seed"],
                release_tick=PLAN["release_tick"],
                audit_factory=factory,
            )
            row["attribution"] = audits[0].result()
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k not in {"trace", "attribution"}}),
                flush=True,
            )
    finally:
        env.close()
    if record != inputs():
        raise ValueError("positive-arc study inputs changed")
    assert len(rows) == PLAN["rows"]
    write_json(
        DIRECTORY / "evaluation.json",
        dict(
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            rows=rows,
            full_signed_gate_passes=sum(r["signed_audit"]["passed"] for r in rows),
            scope="six_scripted_left_development_launch_braking_cases_not_learning",
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    run()
