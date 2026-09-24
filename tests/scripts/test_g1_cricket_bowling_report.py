import copy

import pytest
from audit_g1_cricket_bowling import summarize


def rows():
    return [
        dict(
            engine=engine,
            hand=hand,
            physics_dt_s=dt,
            row=i,
            steps=200,
            terminated=False,
            truncated=True,
            released=i >= 2,
            touch_intervals=0,
            peak_holder_force_n=2,
            minimum_pelvis_height_m=0.7,
            minimum_up_component=0.95,
        )
        for hand in ("right", "left")
        for dt in (0.00025, 0.000125)
        for engine in ("mujoco", "mjbatch")
        for i in range(4)
    ]


def test_smoke_keeps_all_contexts_and_requires_exact_executor_telemetry():
    data = rows()
    result = summarize(data)
    assert result["rows"] == result["completed"] == 32
    assert result["exact_executor_pairs"] == result["released"] == 16
    changed = copy.deepcopy(data)
    changed[0]["peak_holder_force_n"] += 0.001
    with pytest.raises(ValueError, match="mismatch"):
        summarize(changed)
    with pytest.raises(ValueError, match="every"):
        summarize(data[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(data[:-1] + [data[0]])
