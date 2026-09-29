import json

import pytest

from integrations.g1_dynamics import portable_maximum_effort as runner


@pytest.mark.parametrize("task,count", [("all", 22), ("batting", 14), ("bowling", 8)])
def test_complete_membership_and_original_effort_option(tmp_path, monkeypatch, task, count):
    bundle = tmp_path / "runtime"
    (bundle / "data").mkdir(parents=True)
    (bundle / "data/maximum_effort.json").write_text(json.dumps({"approach_shifts_m": {"right": 2, "left": 3}}))
    (bundle / "runtime_manifest.json").write_text("{}")
    monkeypatch.setattr(runner, "verify_bundle", lambda *args: {"amp_checkpoint": {}, "bowling_scenes": {}})
    monkeypatch.setattr(runner, "checked_file", lambda *args: None)
    jobs = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, inputs):
            jobs.extend(inputs)
            return [(dict(case=str(job[5])), {}) for job in inputs]

    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    result = runner.run(bundle, tmp_path, tmp_path, tmp_path, tmp_path / "output",
                        task=task, maximum=False)
    assert len(jobs) == count == len(result["rows"])
    assert len({job[5] for job in jobs}) == count
    assert all(job[6] is False for job in jobs)
    assert result["scope"] == "complete_cohort"
    assert result["promotion_allowed"] is False
    assert result["hardware_ceiling_claim"] is False
