from pathlib import Path

import evaluate_g1_cricket_approach_learning as evaluation
import numpy as np
import pytest
import report_g1_cricket_approach_learning as reporting
from hydra import compose, initialize_config_dir
from PIL import Image

from unilab.base.config_adapter import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[2]


def rows():
    return [
        dict(
            hand=hand,
            controller=controller,
            resolution=resolution,
            seed=1,
            passed=controller == "ppo",
            checks={"gate": controller == "ppo"},
            final_root_position_m=[0, 0, 1],
            peak_holder_force_n=10,
            maximum_holder_error_m=0,
            foot_landings={"left": [1, 2], "right": [1, 2]},
        )
        for hand in evaluation.HANDS
        for controller in evaluation.CONTROLLERS
        for resolution, _ in evaluation.RESOLUTIONS
    ]


def test_candidate_qualification_is_separate_from_reference():
    result = evaluation.qualification(rows())
    assert result["ppo_approach_qualified"]
    assert not result["reference_approach_qualified"]
    assert not result["all_controller_checks_pass"]
    assert len(result["resolution_comparisons"]) == 4


@pytest.mark.parametrize("defect", ["missing", "duplicate", "error"])
def test_incomplete_or_invalid_pool_cannot_qualify(defect):
    pool = rows()
    index = next(i for i, row in enumerate(pool) if row["controller"] == "ppo")
    if defect == "missing":
        pool.pop(index)
    elif defect == "duplicate":
        pool.append(pool[index].copy())
    else:
        pool[index]["evaluation_error"] = {"message": "replay mismatch"}
    result = evaluation.qualification(pool)
    assert not result["ppo_approach_qualified"]
    assert not result["all_controller_checks_pass"]


@pytest.mark.parametrize("hand", evaluation.HANDS)
@pytest.mark.parametrize("failure", [None, "replay", "nonfinite"])
def test_partial_outcome_and_error_trace_are_retained(hand, failure, tmp_path, monkeypatch):
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose(
            "config",
            overrides=[
                "task=g1_cricket_approach_learning/mjbatch",
                f"env.handedness={hand}",
                "env.commands.motion.params.sampling_mode=start",
            ],
        )
    override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
    override["auto_reset"] = False
    env = create_env(owner, num_envs=1, env_cfg_override=override)
    original = evaluation.DeliveryReplay.step
    rendered = []

    def one_interval(self, *args, **kwargs):
        state = original(self, *args, **kwargs)
        if failure == "replay":
            raise AssertionError("test replay mismatch")
        if failure == "nonfinite":
            state.reward[0] = np.nan
        state.terminated[0] = True
        return state

    monkeypatch.setattr(evaluation.DeliveryReplay, "step", one_interval)
    monkeypatch.setattr(evaluation, "render_poses", lambda *args, **kwargs: rendered.append(args))
    try:
        for controller in evaluation.CONTROLLERS:
            for resolution, _ in evaluation.RESOLUTIONS:
                result = evaluation.evaluate_case(
                    env,
                    lambda: np.zeros((1, 29), np.float32),
                    tmp_path,
                    controller,
                    resolution,
                    True,
                )
                assert not result["passed"] and not result["checks"]["complete"]
                with np.load(tmp_path / result["trace"]) as trace:
                    assert len(trace["states"]) == 2
                    assert len(trace["controls"]) == 1
                    assert len(trace["steps"]) == 320
                assert result["exact_endpoint_and_sensor_replay"] == (failure is None)
                if failure:
                    assert result["evaluation_error"]["type"] == "AssertionError"
                else:
                    assert {"initial_settle", "final_settle", "cruise_speed"} <= set(
                        result["failures"]
                    )
        assert len(rendered) == 1
        assert rendered[0][3] == tmp_path / f"{hand}_ppo_approach.mp4"
        assert len(list(tmp_path.glob("*.npz"))) == 4
    finally:
        env.close()


@pytest.mark.parametrize("defect", [None, "blank", "truncated", "fps"])
def test_media_decode_and_early_termination_sheet(defect, tmp_path, monkeypatch):
    np.savez_compressed(tmp_path / "trace.npz", states=np.zeros((3, 10)))
    frame = np.zeros((540, 960, 3), np.uint8)
    if defect != "blank":
        frame[:270] = 255

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_meta_data(self):
            return {"fps": 30 if defect == "fps" else 25}

        def __iter__(self):
            return iter([frame] * (2 if defect == "truncated" else 3))

    monkeypatch.setattr(reporting.imageio, "get_reader", lambda path: Reader())
    row = dict(hand="right", trace="trace.npz")
    if defect:
        with pytest.raises(AssertionError):
            reporting.validate_media(tmp_path, row)
    else:
        result = reporting.validate_media(tmp_path, row)
        assert result["frames"] == 3 and result["sheet_physics_times_s"] == [0, 0.04]
        with Image.open(tmp_path / "right_ppo_approach_contact_sheet.png") as sheet:
            assert sheet.size == (768, 216)
