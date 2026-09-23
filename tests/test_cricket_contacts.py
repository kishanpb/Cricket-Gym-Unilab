"""Optional cricket-plugin contact reporting and native PPO integration checks."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cricket_gym.integrations.unilab_bowling")


def test_fixed_report_keeps_full_pool_and_terminal_stage():
    from scripts.cricket_contact_report import collect

    report = collect()
    rows = report["rows"]
    assert {(row["task"], row["handedness"], row["seed"]) for row in rows} == {
        (task, hand, seed)
        for task, seeds in (("batting", (17000, 17001)), ("bowling", (4101, 4102)))
        for hand in ("right", "left")
        for seed in seeds
    }
    assert len(rows) == 8
    for row in rows:
        telemetry = row["contact_telemetry"]
        assert telemetry["sample_count"] > 0
        assert telemetry["active_sample_count"] == len(telemetry["active_samples"])
        if row["task"] == "bowling":
            terminal = telemetry["terminal_forward"]
            assert terminal["sample_stage"] == "terminal_pose_mj_forward_not_integrated"
            assert terminal["active_sample_count"] == int(row["seed"] == 4101)


def test_native_ppo_accepts_enabled_contact_owner_config(tmp_path):
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

    from unilab.base import registry
    from unilab.base.config_adapter import BackendAdapter, create_env

    registry.ensure_registries(packages=["cricket_gym.integrations"])
    conf = Path(__file__).parents[1] / "src/unilab/conf/ppo"
    with initialize_config_dir(version_base=None, config_dir=str(conf)):
        cfg = compose(
            config_name="config",
            overrides=["task=cricket_bowling/mujoco", "env.contact_telemetry=true"],
        )
    overrides = BackendAdapter(cfg, root_dir=tmp_path, algo_name="ppo").build_task_env_cfg_override()
    env = create_env(cfg, num_envs=2, env_cfg_override=overrides)
    try:
        wrapped = RslRlVecEnvWrapper(env, device="cpu")
        config = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
        config["num_steps_per_env"] = 8
        runner = OnPolicyRunner(wrapped, config, log_dir=None, device="cpu")
        policy = runner.alg.get_policy()
        before = torch.cat([p.detach().flatten().clone() for p in policy.parameters()])
        runner.learn(num_learning_iterations=2)
        after = torch.cat([p.detach().flatten() for p in policy.parameters()])
        assert torch.isfinite(after).all()
        assert torch.linalg.vector_norm(after - before) > 0
        assert np.isfinite(env.state.obs["obs"]).all()
        assert all(row["sample_count"] > 0 for row in env.state.info["contact_telemetry"])
    finally:
        env.close()
