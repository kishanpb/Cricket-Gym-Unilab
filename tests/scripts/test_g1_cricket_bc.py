"""Actor-only initialization, physical provenance and complete development gates."""

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import train_g1_cricket_bc as experiment


def test_owner_only_changes_declared_parent_fields():
    config = OmegaConf.to_container(experiment.owner_config(), resolve=True)
    reference = json.loads((ROOT / "g1_cricket_results/tanh_v1/right/run_config.json").read_text())
    experiment.validate_config(config, reference["config"])
    config["algo"]["policy"]["init_noise_std"] = 0.3
    with pytest.raises(ValueError, match="beyond contact model"):
        experiment.validate_config(config, reference["config"])


def test_bc_changes_actor_only_and_counts_real_transitions():
    torch.manual_seed(1)
    owner = experiment.owner_config()
    env = experiment.make_env(owner)
    try:
        env.reset(seed=5301)
        wrapped, runner = experiment.make_runner(owner, env)
        before = copy.deepcopy(runner.alg.get_policy().mlp.state_dict())
        obs = wrapped.get_observations()["policy"].clone()
        stored = obs.clone()
        wrapped.step(torch.zeros((1, 7)))
        assert wrapped.transitions == 1
        assert torch.equal(obs, stored)
        samples = {
            "observations": obs.repeat(3, 1),
            "ticks": torch.tensor([0, 10, 20]),
            "raw_actions": torch.zeros((3, 7)),
        }
        result = experiment.fit_actor(runner, samples, updates=100)
        assert result["phase_sample_counts"] == [1, 1, 1]
        assert result["losses"][-1] < result["losses"][0]
        assert result["critic_unchanged"] and result["distribution_unchanged"]
        assert result["ppo_optimizer_empty_unchanged"]
        assert not result["closed_loop_success_implied"]
        assert any(
            not torch.equal(before[k], v)
            for k, v in runner.alg.get_policy().mlp.state_dict().items()
        )
        np.testing.assert_allclose(
            runner.alg.get_policy().distribution.state_dict()["std_param"].numpy(), 0.2
        )
    finally:
        env.close()


def test_phases_use_actual_samples_and_reject_missing_phase():
    assert [len(p) for p in experiment.phases(torch.arange(100))] == [10, 10, 80]
    with pytest.raises(ValueError, match="every declared swing phase"):
        experiment.phases(torch.arange(9))


@pytest.mark.parametrize(
    "field", ["physical_model", "preflight_sha256", "teacher_dataset_sha256", "stage"]
)
def test_checkpoint_requires_physical_and_learning_provenance(field):
    expected = dict(
        physical_model="G1CricketImpactV2_2ms",
        preflight_sha256="a",
        teacher_dataset_sha256="b",
        stage="bc",
    )
    experiment.validate_checkpoint({"infos": expected.copy()}, expected)
    corrupt = expected.copy()
    corrupt[field] = "wrong"
    with pytest.raises(ValueError, match="provenance mismatch"):
        experiment.validate_checkpoint({"infos": corrupt}, expected)
    with pytest.raises(ValueError, match="provenance mismatch"):
        experiment.validate_checkpoint({}, expected)


def synthetic_rows():
    p = experiment.PLAN
    rows = []
    for engine, dt, hand, controller, offset, seed in itertools.product(
        p["evaluation_executors"],
        p["evaluation_timesteps_s"],
        p["evaluation_hands"],
        p["evaluation_controllers"],
        p["training_offsets_m"],
        p["evaluation_seeds"],
    ):
        rows.append(
            dict(
                engine=engine,
                sim_dt=dt,
                terminated=False,
                truncated=True,
                first_impact={"force": 10},
                outcome=dict(
                    hand=hand,
                    controller=controller,
                    offset_m=offset,
                    seed=seed,
                    seconds=2,
                    passed=True,
                    failures=[],
                    blade_contact_seen=True,
                    maximum_blade_penetration_m=0.003,
                    first_separation_ball_vx_m_s=1.2,
                    ball_contact_peak_force_norm_n={"bat_blade": 10},
                    fixture_peak_force_norm_n=20,
                    fixture_peak_torque_norm_nm=0.2,
                ),
            )
        )
    return rows


def test_all_contexts_required_without_duplicates():
    rows = synthetic_rows()
    comparisons = experiment.summarize(rows)
    assert len(rows) == experiment.PLAN["evaluation_rows"] == 576
    assert len(comparisons) == 144 and all(r["qualified"] for r in comparisons)
    with pytest.raises(ValueError, match="every unique"):
        experiment.summarize(rows[:-1])
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="every unique"):
        experiment.summarize(rows)


def test_full_impact_mismatch_or_fine_failure_rejects_qualification():
    rows = synthetic_rows()
    rows[0]["first_impact"]["force"] += 1
    comparisons = experiment.summarize(rows)
    assert sum(r["qualified"] for r in comparisons) == 143
    assert sum(not r["executor_evidence_exact"] for r in comparisons) == 1
    rows = synthetic_rows()
    for row in rows:
        if (
            row["outcome"]["hand"],
            row["outcome"]["controller"],
            row["outcome"]["seed"],
            row["outcome"]["offset_m"],
        ) == ("right", "ppo", 4301, 0.0) and row["sim_dt"] == 0.00003125:
            row["outcome"].update(
                passed=False, failures=["speed"], first_separation_ball_vx_m_s=0.999
            )
    comparisons = experiment.summarize(rows)
    assert sum(r["qualified"] for r in comparisons) == 143
    assert all(r["executor_evidence_exact"] for r in comparisons)


def test_fixture_resolution_is_not_hidden_by_executor_parity():
    rows = synthetic_rows()
    for row in rows:
        if row["sim_dt"] == 0.00003125:
            row["outcome"]["fixture_peak_force_norm_n"] *= 2
    comparisons = experiment.summarize(rows)
    assert all(r["executor_evidence_exact"] and not r["qualified"] for r in comparisons)
    assert all(
        r["fixture_resolution_failures"] == ["fixture_peak_force_norm_n"] for r in comparisons
    )


def test_scalar_retention_requires_complete_iteration_coverage(tmp_path):
    path = tmp_path / "scalars.csv"
    header = "iteration,Loss/value,Loss/surrogate,Policy/mean_std\n"
    path.write_text(header + "0,1,2,.2\n1,1,2,.2\n")
    experiment.validate_scalars(path, 2)
    path.write_text(header + "0,1,2,.2\n")
    with pytest.raises(ValueError, match="every PPO iteration"):
        experiment.validate_scalars(path, 2)
    path.write_text(header + "0,1,2,.2\n1,,2,.2\n")
    with pytest.raises(ValueError, match="missing or nonfinite"):
        experiment.validate_scalars(path, 2)


def test_counted_ppo_smoke_can_save_and_reload_actor_only(tmp_path):
    torch.manual_seed(1)
    owner = experiment.owner_config()
    env = experiment.make_env(owner, count=4, training=True)
    try:
        env.reset(seed=1)
        wrapped, runner = experiment.make_runner(owner, env)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        assert wrapped.transitions == 192
        assert runner.current_learning_iteration == 1
        path = tmp_path / "checkpoint.pt"
        runner.save(str(path), infos={"test_only": True})
        payload = torch.load(path, weights_only=True)
        assert payload["infos"] == {"test_only": True}
        _, fresh = experiment.make_runner(owner, env)
        critic = copy.deepcopy(fresh.alg.critic.state_dict())
        fresh.load(
            str(path),
            load_cfg=dict(actor=True, critic=False, optimizer=False, iteration=False, rnd=False),
        )
        assert fresh.current_learning_iteration == 0 and not fresh.alg.optimizer.state
        assert all(
            torch.equal(value, fresh.alg.critic.state_dict()[key]) for key, value in critic.items()
        )
        assert all(
            torch.equal(value, fresh.alg.get_policy().state_dict()[key])
            for key, value in runner.alg.get_policy().state_dict().items()
        )
    finally:
        env.close()


def test_nonfinite_values_stop_before_framework_sanitization():
    cfg = experiment.resolve_nan_guard_cfg(experiment.owner_config().training)
    guard = experiment.StrictNanGuard(cfg, 1, False)
    assert guard.check({"obs": np.zeros((1, 2))}, np.zeros(1)) is None
    assert guard.check_ctrl(np.zeros((1, 7))) is None
    with pytest.raises(RuntimeError, match="before sanitization"):
        guard.check({"obs": np.zeros((1, 2))}, np.array([np.nan]))
    with pytest.raises(RuntimeError, match="before sanitization"):
        guard.check({"obs": np.array([[np.inf, 0]])}, np.zeros(1))
    with pytest.raises(RuntimeError, match="before physics"):
        guard.check_ctrl(np.full((1, 7), np.nan))
    with pytest.raises(RuntimeError, match="physics state"):
        guard.capture(np.array([[np.nan]]))
