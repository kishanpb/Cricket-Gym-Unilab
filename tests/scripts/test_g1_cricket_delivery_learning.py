import copy

import numpy as np
import pytest
import torch
from g1_cricket_delivery_trial import DeliveryEvents
from train_g1_cricket_delivery import PLAN, compare_rows, make_env, make_runner, owner_config


def test_delivery_cpu_ppo_updates_and_counts_actual_transitions(tmp_path):
    torch.set_num_threads(2)
    owner = owner_config()
    owner.algo.num_steps_per_env = 4
    owner.algo.algorithm.num_mini_batches = 1
    env = make_env(owner, "right", count=2, training=True)
    try:
        env.reset(seed=1)
        wrapped, runner = make_runner(owner, env, tmp_path)
        actor = runner.alg.get_policy()
        before = [p.detach().clone() for p in actor.parameters()]
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        assert wrapped.transitions == 8
        assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters(), strict=True))
        assert all(torch.isfinite(p).all() for p in actor.parameters())
        action = runner.get_inference_policy(device="cpu")(wrapped.get_observations())
        assert action.shape == (2, 8)
        runner.logger.writer.flush()
        runner.logger.writer.close()
    finally:
        env.close()


def rows():
    event = DeliveryEvents("right")
    event.height = 0.7
    event.up = 0.9
    outcome = event.finish(True)
    return [
        dict(hand=h, controller=c, seed=s, dt=dt, engine=e, outcome=copy.deepcopy(outcome))
        for h in PLAN["hands"]
        for c in PLAN["controllers"]
        for s in PLAN["evaluation_seeds"]
        for dt in PLAN["evaluation_dt"]
        for e in PLAN["evaluation_engines"]
    ]


def test_complete_pairing_rejects_missing_duplicate_and_executor_difference():
    data = rows()
    assert len(data) == PLAN["evaluation_rows"]
    pairs = compare_rows(data)
    assert len(pairs) == 8 and all(p["exact_executors"] for p in pairs)
    assert not any(p["qualified"] for p in pairs)
    with pytest.raises(ValueError, match="pool"):
        compare_rows(data[:-1])
    with pytest.raises(ValueError, match="pool"):
        compare_rows(data[:-1] + [data[0]])
    data[0]["outcome"]["peak_holder_force_n"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        compare_rows(data)


def test_resolution_failure_is_not_hidden_by_executor_agreement():
    data = rows()
    for row in data:
        if row["dt"] == PLAN["evaluation_dt"][0]:
            row["outcome"]["peak_holder_force_n"] = 10
    assert not any(p["timestep_stable"] for p in compare_rows(data))
    assert not np.isnan(data[0]["outcome"]["minimum_pelvis_height_m"])
