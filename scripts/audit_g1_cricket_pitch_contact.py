"""Ball/pitch force accounting and resolution, not material calibration or learning."""

import json
import xml.etree.ElementTree as ET
from importlib.metadata import version

import mujoco
import numpy as np
from evaluate_g1_cricket_mjbatch import executor_manifest
from evaluate_g1_cricket_residual import sha256
from g1_cricket_delivery_trial import DeliveryReplay, trial
from hydra import compose, initialize_config_dir
from mjbatch.held_control import FULL, HeldControlRollout
from train_g1_cricket_bc import write_json
from train_g1_cricket_delivery import ROOT, VERSIONS, make_env, runtime_sources

from unilab.tasks.manipulation.g1_cricket.pitch_contact import add_pitch_pair
from unilab.tasks.manipulation.g1_cricket.prior import ASSET_HASHES

DIRECTORY = ROOT / "g1_cricket_results/pitch_contact_v2"
DTS = (0.00025, 0.000125, 0.0000625, 0.00003125)
VELOCITIES = ((0, 0, 0), (8, 0, -4), (12, 0, -8))


def drop_model(dt, revised):
    root = ET.fromstring(f"""
    <mujoco><option timestep="{dt}" integrator="implicitfast"/>
      <worldbody><geom name="pitch" type="plane" size="25 25 .05" friction=".7 .01 .001"/>
        <body name="ball" pos="0 0 1.2"><freejoint/>
          <geom name="ball_geom" type="sphere" size=".036" mass=".156" friction=".5 .01 .001"/>
        </body>
      </worldbody>
      <sensor><contact name="impact" geom1="ball_geom" geom2="pitch" num="1"
        data="found force torque dist pos normal tangent"/></sensor>
    </mujoco>""")
    if revised:
        add_pitch_pair(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def drop(dt, velocity, revised):
    model = drop_model(dt, revised)
    data = mujoco.MjData(model)
    data.qvel[:3] = velocity
    initial = np.empty(mujoco.mj_stateSize(model, FULL))
    mujoco.mj_getState(model, data, initial, FULL)
    steps = round(0.6 / dt)
    recorder = HeldControlRollout([model])
    try:
        native, sensors = recorder.rollout(
            [model], [data], initial[None], np.zeros((1, steps, 0)), control_spec=0, nstep=steps
        )
    finally:
        recorder.close()
    impulse = np.zeros(3)
    peak = depth = 0.0
    first_in = first_out = None
    active_previous = False
    contacts = 0
    maximum_momentum_error = 0.0
    snapshot = np.empty_like(initial)
    for tick in range(steps):
        before = data.qvel[:3].copy()
        mujoco.mj_step(model, data)
        mujoco.mj_getState(model, data, snapshot, FULL)
        np.testing.assert_array_equal(snapshot, native[0, tick])
        np.testing.assert_array_equal(data.sensordata, sensors[0, tick])
        active = False
        for index, contact in enumerate(data.contact):
            if contact.efc_address < 0:
                continue
            active = True
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            sign = 1 if contact.geom[1] == model.geom("ball_geom").id else -1
            world = sign * contact.frame.reshape(3, 3).T @ force[:3]
            impulse += world * dt
            peak = max(peak, float(np.linalg.norm(world)))
            depth = max(depth, float(-contact.dist))
        if active and not active_previous:
            contacts += 1
            if first_in is None:
                first_in = before.tolist()
        if not active and active_previous and first_out is None:
            first_out = before.tolist()
        active_previous = active
        momentum = 0.156 * (data.qvel[:3] - velocity - model.opt.gravity * data.time)
        maximum_momentum_error = max(
            maximum_momentum_error, float(np.max(np.abs(momentum - impulse)))
        )
    if data.warning.number.any() or not np.isfinite(native).all() or not np.isfinite(sensors).all():
        raise RuntimeError("invalid isolated pitch impact")
    return dict(
        dt=dt,
        initial_velocity=list(velocity),
        revised=revised,
        exact_native_state_sensors=True,
        peak_force_n=peak,
        maximum_penetration_m=depth,
        contact_onsets=contacts,
        first_in_velocity=first_in,
        first_out_velocity=first_out,
        impulse_world_ns=impulse.tolist(),
        maximum_momentum_residual_ns=maximum_momentum_error,
        first_rebound_energy_ratio=None
        if first_out is None
        else float(np.dot(first_out, first_out) / np.dot(first_in, first_in)),
    )


def source_hashes():
    paths = set()
    for folder in ("base", "managers", "envs", "training", "tasks/manipulation/g1_cricket"):
        paths.update((ROOT / "src/unilab" / folder).rglob("*.py"))
    for name in (
        "src/unilab/tasks/__init__.py",
        "src/unilab/utils/rotation.py",
        "src/unilab/utils/nan_guard.py",
        "src/unilab/assets/robots/g1/g1.xml",
        "src/unilab/assets/robots/g1/scene_flat.xml",
        "scripts/audit_g1_cricket_pitch_contact.py",
        "scripts/g1_cricket_delivery_trial.py",
        "scripts/train_g1_cricket_delivery.py",
        "scripts/train_g1_cricket_bc.py",
        "scripts/evaluate_g1_cricket_residual.py",
        "scripts/evaluate_g1_cricket_mjbatch.py",
        "scripts/retain_g1_training_diagnostics.py",
        "tests/scripts/test_g1_cricket_pitch_contact.py",
        "docs/g1_cricket_pitch_contact_v2.md",
    ):
        paths.add(ROOT / name)
    paths.add(ROOT / "src/unilab/conf/ppo/config.yaml")
    for task in ("prior_v1", "bowling_v1", "delivery_v1", "delivery_pitch_v2"):
        paths.update((ROOT / "src/unilab/conf/ppo/task" / f"g1_cricket_{task}").glob("*.yaml"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def owner_config():
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=["task=g1_cricket_delivery_pitch_v2/mjbatch"])


def run():
    if DIRECTORY.exists():
        raise FileExistsError("inspect retained contact audit instead of restarting")
    inputs = dict(
        sources=source_hashes(),
        executor=executor_manifest(),
        versions={name: version(name) for name in VERSIONS},
        runtime_sources=runtime_sources(),
        external_prior=ASSET_HASHES,
    )
    DIRECTORY.mkdir()
    write_json(DIRECTORY / "preflight.json", inputs)
    drops = [
        drop(dt, velocity, revised)
        for revised in (False, True)
        for velocity in VELOCITIES
        for dt in DTS
    ]
    owner = owner_config()
    rows = []
    for hand in ("right", "left"):
        for dt in DTS[-2:]:
            for engine in ("mujoco", "mjbatch"):
                env = make_env(owner, hand, dt=dt, engine=engine)
                try:

                    def action_at(tick):
                        action = np.zeros((1, 8), np.float32)
                        action[0, 7] = float(tick >= 50)
                        return action

                    outcome = trial(env, DeliveryReplay(env), action_at, 6301)
                    row = dict(hand=hand, dt=dt, engine=engine, outcome=outcome)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                finally:
                    env.close()
    if inputs["sources"] != source_hashes() or inputs["runtime_sources"] != runtime_sources():
        raise ValueError("contact audit inputs changed")
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        if a["outcome"] != b["outcome"]:
            raise ValueError("pitch robot executor mismatch")
    write_json(
        DIRECTORY / "evaluation.json",
        dict(
            preflight_sha256=sha256(DIRECTORY / "preflight.json"),
            isolated_drops=drops,
            robot_rows=rows,
            physical_calibration_validated=False,
            new_training=False,
            policy_promoted=False,
        ),
    )


if __name__ == "__main__":
    run()
