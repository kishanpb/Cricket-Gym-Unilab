"""Compare absolute damping and reference-rate tracking on both frozen deliveries."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from audit_g1_cricket_running_stance import COLUMNS, replay
from retarget_g1_cricket_running import ROBOT, ROOT, render_poses

from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [
        Path(__file__),
        ROOT / "scripts/audit_g1_cricket_running_stance.py",
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROBOT,
        ROBOT.parent / "scene_flat.xml",
    ]
    inputs += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    inputs += [
        args.reference / f"{hand}_{kind}.npz"
        for hand in ("right", "left")
        for kind in ("reference", "physical")
    ]
    hashes = {
        str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    }
    rows = []
    for hand in ("right", "left"):
        with TemporaryDirectory(prefix="g1-rate-") as temporary:
            scene = Path(temporary) / "scene.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
        model.opt.timestep = 0.0000625
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
        with np.load(args.reference / f"{hand}_reference.npz") as reference:
            for relative in (False, True):
                summary, steps, poses = replay(model, reference, 4, track_root_velocity=relative)
                summary.update(hand=hand, track_root_velocity=relative)
                summary["max_joint_limit_excess_rad"] = float(
                    steps[:, COLUMNS.index("maximum_joint_limit_excess")].max()
                )
                summary["peak_motor_fraction"] = float(
                    steps[:, COLUMNS.index("motor_force_fraction")].max()
                )
                if not relative:
                    with np.load(args.reference / f"{hand}_physical.npz") as baseline:
                        np.testing.assert_array_equal(poses, baseline["qpos"])
                    summary["parent_poses_identical"] = True
                name = f"{hand}_{'relative' if relative else 'absolute'}"
                np.savez_compressed(args.output / f"{name}.npz", steps=steps, qpos=poses)
                if args.render and relative:
                    render_poses(
                        model,
                        poses,
                        hand,
                        args.output / f"{name}.mp4",
                        "PHYSICAL RATE TRACKING, NOT PPO",
                    )
                rows.append(summary)
                print(summary, flush=True)
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("rate comparison inputs changed")
    (args.output / "evaluation.json").write_text(
        json.dumps(
            {
                "scope": "fixed_reference_native_PD_rate_comparison_not_learned_bowling",
                "mujoco_version": mujoco.__version__,
                "control_period_s": 0.02,
                "columns": COLUMNS,
                "input_sha256": hashes,
                "rows": rows,
                "guard": "Retain all four episodes. Relative-rate feedback does not certify full cricket, contact or promotion gates.",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
