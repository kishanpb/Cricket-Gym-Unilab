"""Project both complete batting references; preserve bat targets, not simulated outcomes."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from unilab.tasks.manipulation.g1_cricket.batting_projection import BattingProjection
from unilab.tasks.manipulation.g1_cricket.bimanual import support_feedforward
from unilab.tasks.manipulation.g1_cricket.bimanual_contact import G1BimanualContactCfg
from unilab.tasks.manipulation.g1_cricket.tracking import export_reference

ROOT = Path(__file__).resolve().parents[1]


def project(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [Path(__file__)]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    rows = []
    for hand in ("right", "left"):
        config_path = (
            ROOT / f"g1_cricket_results/bimanual_batting_learning_v1/ppo_{hand}/run_config.json"
        )
        config = json.loads(config_path.read_text())["config"]
        reference = ROOT / config["env"]["actions"]["reference"]["reference_file"]
        robot = ROOT / config["env"]["scene"]["model_file"]
        inputs += [config_path, reference, robot]
        with TemporaryDirectory(prefix="g1-batting-projection-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1BimanualContactCfg(handedness=hand).build_scene(robot, scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.00003125
        with np.load(reference) as saved:
            times, original = saved["times"], saved["qpos"]
        helper = BattingProjection(model, original[0], hand)
        data = mujoco.MjData(model)
        poses = []
        for time, pose in zip(times, original, strict=True):
            projected, row = helper.project(pose)
            poses.append(projected)
            torque, residual = support_feedforward(model, projected)
            data.qpos[:] = projected
            mujoco.mj_forward(model, data)
            overlaps = []
            for contact in data.contact:
                names = [model.geom(int(index)).name for index in contact.geom]
                foot_support = "pitch" in names and any("foot" in name for name in names)
                if contact.dist < -1e-6 and not foot_support:
                    overlaps.append({"geoms": names, "penetration_m": float(-contact.dist)})
            row.update(
                hand=hand,
                time_s=float(time),
                static_ideal_support_residual=float(np.linalg.norm(residual)),
                static_ideal_support_motor_force=torque.tolist(),
                unintended_overlaps=overlaps,
            )
            row["kinematic_pass"] = (
                row["optimizer_success"]
                and row["max_constraint_error"] < 1e-8
                and row["static_ideal_support_residual"] < 1e-6
                and not overlaps
            )
            rows.append(row)
        np.savez_compressed(output / f"{hand}_reference.npz", times=times, qpos=np.asarray(poses))
        export_reference(model, np.asarray(poses), 50, output / f"{hand}_tracking.npz")
    report = {
        "scope": "offline_kinematic_projection_not_physical_feasibility",
        "contract": "all frames, same bat SE3 and timing, fixed initial feet, exact lower grip, unchanged model/limits; root translation bounded within 5 cm per axis of each old pose",
        "kinematic_pass": all(row["kinematic_pass"] for row in rows),
        "rows": rows,
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        },
        "output_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.glob("*.npz"))
        },
    }
    (output / "projection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frames": len(rows), "kinematic_pass": report["kinematic_pass"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    project(parser.parse_args().output.resolve())
