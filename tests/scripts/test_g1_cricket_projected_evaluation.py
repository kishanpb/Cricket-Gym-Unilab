import json
from pathlib import Path

import pytest
from evaluate_g1_cricket_tracking import ROOT, evaluate


@pytest.mark.parametrize("failure", ["missing", "failed", "changed", "wrong_task"])
def test_invalid_projection_stops_before_environment_or_output(tmp_path, failure):
    run = tmp_path / "run"
    run.mkdir()
    original = ROOT / "g1_cricket_results/bimanual_batting_learning_v1/ppo_right"
    config = json.loads((original / "run_config.json").read_text())
    if failure == "wrong_task":
        config["config"]["training"]["task_name"] = "G1CricketBimanualTracking"
    (run / "run_config.json").write_text(json.dumps(config))
    (run / "run_summary.json").write_bytes((original / "run_summary.json").read_bytes())
    projection = tmp_path / "projection"
    projection.mkdir()
    if failure != "missing":
        (projection / "projection.json").write_text(
            json.dumps({"kinematic_pass": failure != "failed", "output_sha256": {"pose": "wrong"}})
        )
        (projection / "pose").write_bytes(b"pose")
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate(
            run,
            output=tmp_path / "evaluation",
            contact_dt=0.00003125,
            bounced_delivery=True,
            reference_directory=Path(projection),
        )
    assert not (tmp_path / "evaluation").exists()
