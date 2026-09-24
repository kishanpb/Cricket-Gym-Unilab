"""Paired six-case selected-shoulder damping comparison; no learned policy."""

from evaluate_g1_cricket_residual import sha256
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from probe_g1_cricket_positive_arc import DIRECTORY as PARENT
from probe_g1_cricket_positive_arc import inputs as arc_inputs
from probe_g1_cricket_positive_arc import run
from train_g1_cricket_delivery import ROOT

DIRECTORY = ROOT / "g1_cricket_results/shoulder_damping_v1"


def owner_config(engine="mjbatch"):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        return compose("config", overrides=[f"task=g1_cricket_shoulder_damping_v1/{engine}"])


def inputs():
    record = arc_inputs()
    for name in (
        "scripts/probe_g1_cricket_shoulder_damping.py",
        "tests/scripts/test_g1_cricket_shoulder_damping.py",
        "docs/g1_cricket_shoulder_damping_v1.md",
        "src/unilab/tasks/manipulation/g1_cricket/impedance.py",
        "src/unilab/tasks/__init__.py",
        "src/unilab/conf/ppo/task/g1_cricket_shoulder_damping_v1/mujoco.yaml",
        "src/unilab/conf/ppo/task/g1_cricket_shoulder_damping_v1/mjbatch.yaml",
    ):
        record["sources"][name] = sha256(ROOT / name)
    record.update(
        changed_axis="selected_shoulder_pitch_controller_kd_10_to_2",
        config=OmegaConf.to_container(owner_config(), resolve=True),
        parent_positive_arc_sha256=sha256(PARENT / "evaluation.json"),
        checkpoint_scope="distinct_action_config_strict_contract_no_old_checkpoint_reuse",
    )
    return record


if __name__ == "__main__":
    run(directory=DIRECTORY, owner_factory=owner_config, input_record=inputs)
