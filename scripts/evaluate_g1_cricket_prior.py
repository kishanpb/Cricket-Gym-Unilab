"""Complete native zero-command transfer audit of an external Unitree policy."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import onnxruntime as ort
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.tasks.manipulation.g1_cricket.prior import (
    ASSET_HASHES,
    POLICY_DEFAULT,
    POLICY_TO_SDK,
    REVISION,
    SDK_JOINTS,
    SDK_KD,
    SDK_KP,
)
from unilab.tasks.manipulation.g1_cricket.task import G1CricketCfg, G1CricketEnv, fallen

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(4201, 4209))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prior(directory: Path) -> ort.InferenceSession:
    for name, expected in ASSET_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f"unexpected external Unitree asset: {name}")
    config = OmegaConf.load(directory / "deploy.yaml")
    for name, expected in (
        ("joint_ids_map", POLICY_TO_SDK),
        ("default_joint_pos", POLICY_DEFAULT),
        ("stiffness", SDK_KP),
        ("damping", SDK_KD),
    ):
        np.testing.assert_array_equal(config[name], expected)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(directory / "policy.onnx"), options, providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].shape != [1, 480] or session.get_outputs()[0].shape != [1, 29]:
        raise ValueError("unexpected Unitree network contract")
    return session


def make_env(hand: str, version: str = "v1") -> G1CricketEnv:
    registry.ensure_registries()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[f"task=g1_cricket_prior_{version}/mujoco", f"env.handedness={hand}"],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    return cast(
        G1CricketEnv,
        registry.make(
            "G1CricketBatting", num_envs=1, sim_backend="mujoco", env_cfg_override=override
        ),
    )


def pose_snapshot(env: G1CricketEnv) -> np.ndarray:
    robot, ball = env.scene["robot"], env.scene["ball"]
    return np.concatenate(
        (robot.data.root_link_pose_w[0], robot.data.joint_pos[0], ball.data.root_link_pose_w[0])
    )


def evaluate(directory: Path, version: str = "v1") -> tuple[dict, dict]:
    session = load_prior(directory)
    rows, images = [], {}
    for hand in ("none", "right", "left"):
        env = make_env(hand, version)
        try:
            cfg = cast(G1CricketCfg, env.cfg)
            model = mujoco.MjModel.from_xml_path(
                str(Path(env.scene_directory.name) / "cricket.xml")
            )
            robot = env.scene["robot"]
            joints = model.actuator_trnid[:, 0]
            limits = model.jnt_range[joints]
            force_limits = model.actuator_forcerange[:, 1]
            force = env.scene.bind_sensor_data(tuple(f"prior_force_{name}" for name in SDK_JOINTS))
            contacts = [
                (name, env.scene.bind_sensor_data((name,))) for name in cfg.bat_guard_sensor_names
            ]
            for controller in ("constant_target", "unitree_onnx"):
                for seed in SEEDS:
                    obs, _ = env.reset(seed=seed)
                    start = robot.data.root_link_pos_w[0].copy()
                    min_height, drift, excess, max_force, max_action = (
                        float(start[2]),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    contact_counts = dict.fromkeys(cfg.bat_guard_sensor_names, 0)
                    contact_peaks = dict.fromkeys(cfg.bat_guard_sensor_names, 0.0)
                    frames = (
                        [(0.0, pose_snapshot(env))]
                        if controller == "unitree_onnx" and seed == SEEDS[0]
                        else []
                    )
                    for step in range(env.max_episode_length):
                        policy_action = (
                            session.run(["actions"], {"obs": obs["obs"].astype(np.float32)})[0]
                            if controller == "unitree_onnx"
                            else np.zeros((1, 29), dtype=np.float32)
                        )
                        action = np.empty_like(policy_action)
                        action[:, POLICY_TO_SDK] = policy_action
                        state = env.step(action)
                        obs = state.obs
                        if not np.isfinite(obs["obs"]).all() or not np.isfinite(force.read()).all():
                            raise RuntimeError("non-finite native G1 prior rollout")
                        min_height = min(min_height, float(robot.data.root_link_pos_w[0, 2]))
                        drift = max(
                            drift,
                            float(np.linalg.norm(robot.data.root_link_pos_w[0, :2] - start[:2])),
                        )
                        q = robot.data.joint_pos[0]
                        excess = max(
                            excess, float(np.maximum(limits[:, 0] - q, q - limits[:, 1]).max())
                        )
                        max_force = max(
                            max_force, float((np.abs(force.read()[0]) / force_limits).max())
                        )
                        max_action = max(max_action, float(np.abs(policy_action).max()))
                        for name, view in contacts:
                            records = view.read().reshape(-1, 17)
                            contact_counts[name] += int(np.any(records[:, 0] > 0))
                            contact_peaks[name] = max(
                                contact_peaks[name],
                                float(np.linalg.norm(records[:, 1:4], axis=1).max()),
                            )
                        done = bool(state.terminated[0] or state.truncated[0])
                        if frames and (step + 1 in (100, 250, 500) or done):
                            frames.append(((step + 1) * env.step_dt, pose_snapshot(env)))
                        if done:
                            break
                    else:
                        raise RuntimeError("native prior episode exceeded its declared horizon")
                    rows.append(
                        {
                            "hand": hand,
                            "controller": controller,
                            "seed": seed,
                            "seconds": (step + 1) * env.step_dt,
                            "fell": bool(fallen(env)[0]),
                            "terminated": bool(state.terminated[0]),
                            "time_limit": bool(state.truncated[0]),
                            "completed": bool(state.truncated[0] and not state.terminated[0]),
                            "minimum_pelvis_height_m": min_height,
                            "maximum_xy_drift_m": drift,
                            "maximum_joint_limit_excess_rad": excess,
                            "maximum_actuator_force_limit_fraction_at_control_snapshots": max_force,
                            "maximum_raw_policy_action": max_action,
                            "contact_presence_control_counts": contact_counts,
                            "contact_force_norm_control_snapshot_peaks_n": contact_peaks,
                            "final_root_pose": robot.data.root_link_pose_w[0].tolist(),
                        }
                    )
                    if frames:
                        images[hand] = render_frames(model, hand, frames)
        finally:
            env.close()
    sources = [Path(__file__), ROOT / "src/unilab/conf/ppo/task/g1_cricket_prior_v1/mujoco.yaml"]
    sources += [
        ROOT / "src/unilab/tasks/manipulation/g1_cricket" / name
        for name in ("prior.py", "scene.py", "task.py")
    ]
    if version == "v2":
        sources.append(ROOT / "src/unilab/conf/ppo/task/g1_cricket_prior_v2/mujoco.yaml")
    return {
        "scope": "Native UniLab transfer of externally trained locomotion; no local cricket learning or hardware claim",
        "version": version,
        "source": f"https://github.com/unitreerobotics/unitree_rl_lab/tree/{REVISION}",
        "external_asset_sha256": ASSET_HASHES,
        "external_asset_distribution": "Local cache only; no checkpoint/config redistribution",
        "contract": {
            "input_shape": [1, 480],
            "history": "term-major, oldest first, repeat first on reset",
            "imu_frame": "pelvis",
            "policy_to_sdk": POLICY_TO_SDK.tolist(),
            "native_action_order": "SDK; observer remaps previous raw action to policy order",
            "control_dt_s": 0.02,
            "physics_dt_s": 0.002,
            "root_reset_height_m": 0.8,
            "bat_mount": "legacy" if version == "v1" else "forward_down_45_degrees_at_default_pose",
            "joint_reset_jitter_rad": 0.005,
            "seeds": list(SEEDS),
            "horizon_seconds": 10,
            "root_support": False,
            "pd_gains": "All 29 kp/kv pairs replaced by official deploy.yaml SDK-order gains; native force limits unchanged",
            "pose_write_after_reset": False,
            "contact_scope": "Native end-of-control sensor snapshots, may miss between-control impacts; not integrated peaks or impulses",
            "force_scope": "Simulated uncalibrated geometry contact and actuator force, not hardware tactile pressure",
            "comparison": "Within native UniLab only; different asset, fixture and solver from mjbatch",
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("unilab-rl", "unisim-core", "mujoco", "onnxruntime", "numpy")
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "robot_xml_sha256": sha256(ROOT / "src/unilab/assets/robots/g1/g1.xml"),
        "robot_scene_flat_sha256": sha256(ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"),
        "rows": rows,
    }, images


def render_frames(model, hand, frames):
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0.3 if hand != "left" else -0.3, 0.7]
    camera.distance, camera.azimuth, camera.elevation = 3, 125, -12
    images = []
    with mujoco.Renderer(model, height=360, width=640) as renderer:
        for time, qpos in frames:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            image = renderer.render().copy()
            if np.std(image) < 10:
                raise RuntimeError("blank native prior diagnostic")
            images.append((time, image))
    return images


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    report, images = evaluate(args.assets, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for axis in axes.flat:
        axis.axis("off")
    for row, hand in enumerate(("none", "right", "left")):
        for col, (time, image) in enumerate(images[hand]):
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"{hand} bat | {time:g} s")
    fig.suptitle(
        f"Native UniLab {args.version} | external Unitree prior | seed 4201 | NOT learned cricket"
    )
    fig.tight_layout()
    fig.savefig(args.output.with_name("stance_diagnostic.png"), dpi=120)
    plt.close(fig)
    print(
        json.dumps(
            [
                {k: row[k] for k in ("hand", "controller", "seed", "seconds", "completed", "fell")}
                for row in report["rows"]
            ]
        )
    )


if __name__ == "__main__":
    main()
