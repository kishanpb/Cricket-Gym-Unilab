"""Isolate numerical bat/ball compliance; this is not a robot-policy evaluation."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def probe(dt, time_constant):
    xml = f"""<mujoco>
      <option timestep="{dt}" gravity="0 0 0" integrator="implicitfast"/>
      <default><geom solref="{time_constant} 1"/></default>
      <worldbody>
        <geom name="bat" type="box" size=".025 .055 .2"/>
        <body pos=".1 0 0"><freejoint/>
          <geom name="ball" type="sphere" size=".036" mass=".156"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[0] = -2.5
    first = separation = None
    depth = peak = impulse = 0.0
    for step in range(round(1 / dt)):
        mujoco.mj_step(model, data)
        if data.ncon:
            if first is None:
                first = (step + 1) * dt
            depth = max(depth, -float(data.contact[0].dist))
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, 0, wrench)
            peak = max(peak, float(wrench[0]))
            impulse += float(wrench[0]) * dt
        elif first is not None and separation is None:
            separation = {"seconds": (step + 1) * dt, "vx_m_s": float(data.qvel[0])}
    np.testing.assert_allclose(impulse, 0.156 * (data.qvel[0] + 2.5), atol=1e-12)
    return {
        "sim_dt": dt,
        "solref_time_constant_s": time_constant,
        "damping_ratio": 1,
        "first_contact_s": first,
        "first_separation": separation,
        "maximum_penetration_m": depth,
        "peak_normal_force_n": peak,
        "normal_impulse_ns": impulse,
        "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
    }


if __name__ == "__main__":
    rows = [
        probe(dt, tc)
        for tc in (0.02, 0.004)
        for dt in (0.002, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625)
    ]
    report = {
        "scope": "isolated_fixed_blade_normal_impact_not_robot_policy_evidence",
        "setup": "0.156 kg, 36 mm sphere at -2.5 m/s against fixed blade; no gravity or tangential motion; one second",
        "changed_axis": "contact time constant only; six-step numerical sensitivity per value; two finest steps added after initial four-step peak sensitivity",
        "candidate_status": "uncalibrated_design_probe_not_applied_to_native_environment",
        "mujoco_version": mujoco.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
    }
    destination = (
        Path(__file__).resolve().parents[1] / "g1_cricket_results/residual_v3/isolated_impact.json"
    )
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))
