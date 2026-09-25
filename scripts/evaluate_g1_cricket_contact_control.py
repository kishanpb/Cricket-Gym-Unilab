"""Fixed both-hand contact-acceleration controller comparison; no training."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
from audit_g1_cricket_running_stance import COLUMNS, replay
from retarget_g1_cricket_running import ROBOT, ROOT, render_poses, running_control

from unilab.tasks.manipulation.g1_cricket.contact_control import ContactAccelerationControl
from unilab.tasks.manipulation.g1_cricket.pitch_contact import G1CricketDeliveryPitchV2Cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/audit_g1_cricket_running_stance.py",
        ROOT / "scripts/retarget_g1_cricket_running.py",
        ROBOT,
        ROBOT.parent / "scene_flat.xml",
    ]
    sources += sorted((ROOT / "src/unilab/tasks/manipulation/g1_cricket").glob("*.py"))
    sources += [(args.reference / f"{hand}_reference.npz").resolve() for hand in ("right", "left")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results = []
    start = time.monotonic()
    with TemporaryDirectory(prefix="g1-contact-control-") as temporary:
        for hand in ("right", "left"):
            scene = Path(temporary) / f"{hand}.xml"
            G1CricketDeliveryPitchV2Cfg(handedness=hand).build_scene(ROBOT, scene)
            model = mujoco.MjModel.from_xml_path(str(scene))
            model.opt.timestep = 0.0000625
            model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
            with np.load(args.reference / f"{hand}_reference.npz") as reference:
                controller = ContactAccelerationControl(model, reference["qvel"], running_control)
                summary, steps, poses = replay(model, reference, 4, controller=controller)
                summary.update(
                    hand=hand, controller="contact_acceleration", optimization=controller.trace
                )
                summary["max_joint_limit_excess_rad"] = float(
                    steps[:, COLUMNS.index("maximum_joint_limit_excess")].max()
                )
                summary["peak_motor_fraction"] = float(
                    steps[:, COLUMNS.index("motor_force_fraction")].max()
                )
                np.savez_compressed(args.output / f"{hand}.npz", steps=steps, qpos=poses)
                if args.render:
                    render_poses(
                        model,
                        poses,
                        hand,
                        args.output / f"{hand}.mp4",
                        "PHYSICAL CONTACT CONTROL, NOT PPO",
                    )
                results.append(summary)
                print(
                    {key: value for key, value in summary.items() if key != "optimization"},
                    flush=True,
                )
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("inputs changed during evaluation")
    report = {
        "scope": "Native physical motor-only experiment, not learned bowling or promotion evidence.",
        "mujoco_version": mujoco.__version__,
        "elapsed_seconds": time.monotonic() - start,
        "columns": COLUMNS,
        "input_sha256": hashes,
        "results": results,
        "guard": "The local acceleration objective is a controller diagnostic, not a physical success metric. Full cricket/contact gates remain required.",
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
