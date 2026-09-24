"""Isolated contact-model sensitivity, not robot control or material calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "g1_cricket_results/compliance_v1"
PLAN = {
    "time_constants_s": [0.004, 0.002],
    "incident_speeds_m_s": [2.5, 4.67],
    "timesteps_s": [0.00025, 0.000125, 0.0000625, 0.00003125],
    "duration_s": 1.0,
    "expected_rows": 16,
    "changed_axis": "positive_solref_time_constant_at_fixed_damping_ratio_one",
    "transverse_speed_tolerance_m_s": 1e-6,
    "angular_speed_tolerance_rad_s": 1e-4,
}
PAIR = {
    "name": "cricket_blade_impact_v1",
    "geom1": "ball_geom",
    "geom2": "bat_blade",
    "condim": "3",
    "solref": "0.004 1",
    "solimp": "0.9 0.95 0.001 0.5 2",
    "friction": "0.6 0.6 0.01 0.001 0.001",
    "margin": "0",
    "gap": "0",
}
TOLERANCES = {
    "maximum_penetration_m": 0.0001,
    "peak_normal_force_n": 1.0,
    "contact_seconds": 0.000125,
    "loaded_seconds": 0.000125,
    "first_exit_vx_m_s": 0.01,
    "force_norm_integral_ns": 0.001,
    "final_kinetic_energy_j": 0.005,
}
INPUTS = (
    "scripts/probe_g1_cricket_compliance.py",
    "tests/scripts/test_g1_cricket_compliance.py",
    "docs/g1_cricket_compliance_v1.md",
    "src/unilab/tasks/manipulation/g1_cricket/impact.py",
    "src/unilab/tasks/manipulation/g1_cricket/scene.py",
    "scripts/probe_g1_cricket_impact.py",
    "g1_cricket_results/residual_v3/isolated_impact.json",
    "g1_cricket_results/elbow_v1/evaluation.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_xml(dt, time_constant, side=1):
    pair = {**PAIR, "solref": f"{time_constant} 1"}
    attributes = " ".join(f'{key}="{value}"' for key, value in pair.items())
    return f"""<mujoco>
      <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <geom name="bat_blade" type="box" size=".025 .055 .2"/>
        <body name="ball" pos="{side * 0.1} 0 0"><freejoint/>
          <geom name="ball_geom" type="sphere" size=".036" mass=".156"/>
        </body>
      </worldbody>
      <contact><pair {attributes}/></contact>
    </mujoco>"""


def compiled_parameters(model):
    return {
        "solref": model.pair_solref[0].tolist(),
        "solimp": model.pair_solimp[0].tolist(),
        "friction": model.pair_friction[0].tolist(),
        "condim": int(model.pair_dim[0]),
        "margin": float(model.pair_margin[0]),
        "gap": float(model.pair_gap[0]),
        "geom_order": [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i))
            for i in (model.pair_geom1[0], model.pair_geom2[0])
        ],
        "ball_mass_kg": float(model.body_mass[model.body("ball").id]),
        "gravity": model.opt.gravity.tolist(),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "tolerance": float(model.opt.tolerance),
        "cone": int(model.opt.cone),
        "disableflags": int(model.opt.disableflags),
    }


def probe(dt, time_constant, speed, side=1):
    xml = model_xml(dt, time_constant, side)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    ball = model.geom("ball_geom").id
    mass = float(model.body_mass[model.body("ball").id])
    data.qvel[0] = -side * speed
    initial_velocity = data.qvel[:3].copy()
    initial_energy = 0.5 * mass * speed**2
    impulse = np.zeros(3)
    work = integral = peak = depth = momentum_error = 0.0
    loaded = 0
    maximum_transverse = maximum_angular = 0.0
    finite = True
    trace = []
    separation = None
    previous_contact = False
    episodes = 0
    for step in range(round(PLAN["duration_s"] / dt)):
        before = data.qvel[:3].copy()
        mujoco.mj_step(model, data)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        maximum_transverse = max(maximum_transverse, float(np.linalg.norm(data.qvel[1:3])))
        maximum_angular = max(maximum_angular, float(np.linalg.norm(data.qvel[3:])))
        total_force = np.zeros(3)
        if data.ncon:
            assert data.ncon == 1
            contact = data.contact[0]
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, 0, wrench)
            # Contact-frame wrench acts on geom2; transform before integrating.
            sign = 1 if contact.geom[1] == ball else -1
            total_force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
            force_norm = float(np.linalg.norm(total_force))
            loaded += wrench[0] > 0
            integral += force_norm * dt
            peak = max(peak, float(wrench[0]))
            depth = max(depth, -float(contact.dist))
            episodes += not previous_contact
            trace.append(
                {
                    "solve_time_s": step * dt,
                    "integrated_time_s": (step + 1) * dt,
                    "distance_m": float(contact.dist),
                    "force_world_on_ball_n": total_force.tolist(),
                    "normal_force_n": float(wrench[0]),
                    "velocity_before_m_s": before.tolist(),
                    "velocity_after_m_s": data.qvel[:3].tolist(),
                    "angular_velocity_after_rad_s": data.qvel[3:].tolist(),
                }
            )
        elif previous_contact and separation is None:
            separation = {"solve_time_s": step * dt, "vx_m_s": float(data.qvel[0])}
        previous_contact = bool(data.ncon)
        impulse += total_force * dt
        work += float(total_force @ (before + data.qvel[:3]) / 2) * dt
        momentum_error = max(
            momentum_error,
            float(np.max(np.abs(mass * (data.qvel[:3] - before) - total_force * dt))),
        )
    final_energy = float(0.5 * mass * (data.qvel[:3] @ data.qvel[:3]))
    inertia = model.body_inertia[model.body("ball").id]
    rotational_energy = float(0.5 * np.sum(inertia * data.qvel[3:] ** 2))
    momentum_delta = mass * (data.qvel[:3] - initial_velocity)
    failures = []
    for check, passed in {
        "single_completed_contact": episodes == 1 and separation is not None and not data.ncon,
        "finite_states": finite,
        "loaded_contact": 0 < loaded <= len(trace),
        "no_solver_warnings": not np.any(data.warning.number),
        "momentum_balance": np.max(np.abs(momentum_delta - impulse)) <= 1e-10,
        "step_momentum_balance": momentum_error <= 1e-10,
        "work_energy_balance": abs(final_energy - initial_energy - work) <= 1e-10,
        "no_net_energy_gain": final_energy + rotational_energy <= initial_energy + 1e-10,
        "near_normal_motion": maximum_transverse <= PLAN["transverse_speed_tolerance_m_s"]
        and maximum_angular <= PLAN["angular_speed_tolerance_rad_s"],
        "nonnegative_normal_forces": all(t["normal_force_n"] >= 0 for t in trace),
    }.items():
        if not passed:
            failures.append(check)
    return {
        "sim_dt": dt,
        "time_constant_s": time_constant,
        "incident_speed_m_s": speed,
        "side": side,
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "compiled_parameters": compiled_parameters(model),
        "physics_steps": round(PLAN["duration_s"] / dt),
        "contact_episodes": episodes,
        "first_contact_s": trace[0]["solve_time_s"] if trace else None,
        "first_separation": separation,
        "first_exit_vx_m_s": separation["vx_m_s"] if separation else None,
        "rebound_ratio": side * separation["vx_m_s"] / speed if separation else None,
        "contact_seconds": len(trace) * dt,
        "loaded_seconds": int(loaded) * dt,
        "maximum_penetration_m": depth,
        "peak_normal_force_n": peak,
        "force_norm_integral_ns": integral,
        "world_impulse_on_ball_ns": impulse.tolist(),
        "momentum_change_kg_m_s": momentum_delta.tolist(),
        "maximum_step_momentum_error_ns": momentum_error,
        "initial_kinetic_energy_j": initial_energy,
        "final_kinetic_energy_j": final_energy,
        "final_rotational_kinetic_energy_j": rotational_energy,
        "net_energy_removed_j": initial_energy - final_energy - rotational_energy,
        "maximum_transverse_speed_m_s": maximum_transverse,
        "maximum_angular_speed_rad_s": maximum_angular,
        "contact_translational_work_j": work,
        "work_energy_error_j": abs(final_energy - initial_energy - work),
        "validation_failures": failures,
        "contact_trace": trace,
    }


def compare(coarse, fine):
    return [
        key
        for key, absolute in TOLERANCES.items()
        if coarse[key] is None
        or fine[key] is None
        or abs(coarse[key] - fine[key]) > max(absolute, 0.05 * abs(fine[key]))
    ]


def main(preflight):
    contract_path = DIRECTORY / "preflight.json"
    output_path = DIRECTORY / "evaluation.json"
    if preflight:
        if DIRECTORY.exists():
            raise FileExistsError("compliance evidence already retained")
        contract = {
            "scope": "isolated_fixed_blade_model_sensitivity_not_robot_evaluation",
            "plan": PLAN,
            "pair": PAIR,
            "absolute_tolerances": TOLERANCES,
            "relative_tolerance": 0.05,
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
            "input_sha256": {name: sha256(ROOT / name) for name in INPUTS},
            "robot_task_changed": False,
            "physical_calibration_validated": False,
            "policy_promoted": False,
        }
        DIRECTORY.mkdir()
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        return
    if output_path.exists():
        raise FileExistsError("compliance results already retained")
    contract = json.loads(contract_path.read_text())
    assert contract["plan"] == PLAN and contract["pair"] == PAIR
    assert contract["mujoco_version"] == mujoco.__version__
    assert contract["numpy_version"] == np.__version__
    pins = {**contract["input_sha256"], str(contract_path.relative_to(ROOT)): sha256(contract_path)}
    assert all(sha256(ROOT / name) == digest for name, digest in pins.items())
    rows, comparisons = [], []
    for tc in PLAN["time_constants_s"]:
        for speed in PLAN["incident_speeds_m_s"]:
            group = [probe(dt, tc, speed) for dt in PLAN["timesteps_s"]]
            rows.extend(group)
            for a, b in zip(group, group[1:]):
                comparisons.append(
                    {
                        "time_constant_s": tc,
                        "incident_speed_m_s": speed,
                        "coarse_dt": a["sim_dt"],
                        "fine_dt": b["sim_dt"],
                        "failed_checks": compare(a, b),
                    }
                )
            print(
                json.dumps(
                    [
                        {
                            k: r[k]
                            for k in (
                                "sim_dt",
                                "time_constant_s",
                                "incident_speed_m_s",
                                "maximum_penetration_m",
                                "peak_normal_force_n",
                                "rebound_ratio",
                                "validation_failures",
                            )
                        }
                        for r in group
                    ]
                ),
                flush=True,
            )
    assert len(rows) == PLAN["expected_rows"]
    assert all(sha256(ROOT / name) == digest for name, digest in pins.items())
    report = {**contract, "input_sha256": pins, "rows": rows, "comparisons": comparisons}
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    main(parser.parse_args().preflight)
