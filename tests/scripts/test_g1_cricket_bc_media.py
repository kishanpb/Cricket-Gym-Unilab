"""Fine-step media timing and interval tactile reporting, not success selection."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import render_g1_cricket_bc as media


@pytest.mark.parametrize("dt,steps,stride", [(0.0000625, 320, 160), (0.00003125, 640, 320)])
def test_frame_windows_cover_all_substeps_at_true_slow_motion(dt, steps, stride):
    windows = media.frame_windows(steps, dt)
    assert windows == ((0, stride), (stride, 2 * stride))
    assert [i for start, stop in windows for i in range(start, stop)] == list(range(steps))
    times = [(tick * steps + stop) * dt for tick in range(100) for _, stop in windows]
    np.testing.assert_allclose(times, np.arange(1, 201) * 0.01, atol=1e-12)
    assert len(times) / media.SELECTION["video_fps"] == 4.0


def test_invalid_frame_intervals_rejected():
    with pytest.raises(ValueError, match="divide"):
        media.frame_windows(200, 0.00003)
    with pytest.raises(ValueError, match="two 10 ms"):
        media.frame_windows(10, 0.00003125)


def contact(force=(3, 4, 0), normal=(1, 0, 0), load=3):
    return dict(
        force_on_ball_world_n=list(force), normal_on_ball_world=list(normal), normal_force_n=load
    )


def test_brief_contact_is_visible_between_frame_endpoints():
    steps = [{"contacts": []} for _ in range(320)]
    steps[5]["contacts"] = [contact()]
    steps[6]["contacts"] = [contact(force=(0, 0, 0), load=0)]
    fixture = np.zeros((320, 6))
    fixture[8] = [6, 8, 0, 0, 0, 2]
    result = media.tactile_bin(steps, fixture, 0.00003125)
    assert result["interval_seconds"] == 0.01
    assert result["blade_occupied_fraction"] == 2 / 320
    assert result["blade_loaded_fraction"] == 1 / 320
    assert result["peak_sum_normal_force_n"] == 3
    assert result["peak_sum_shear_magnitude_n"] == 4
    np.testing.assert_allclose(result["impulse_on_ball_world_ns"], np.array([3, 4, 0]) * 0.00003125)
    assert result["fixture_peak_force_n"] == 10
    assert result["fixture_peak_torque_nm"] == 2


def test_signed_impulse_not_confused_with_scalar_peak_sums():
    steps = [{"contacts": [contact(), contact(force=(-3, -4, 0), normal=(-1, 0, 0))]}]
    result = media.tactile_bin(steps, np.zeros((1, 6)), 0.00003125)
    assert result["peak_sum_normal_force_n"] == 6
    assert result["peak_sum_shear_magnitude_n"] == 8
    assert result["impulse_on_ball_world_ns"] == [0, 0, 0]


def test_no_contact_remains_zero_without_fabricated_touch():
    result = media.tactile_bin([{"contacts": []}] * 320, np.zeros((320, 6)), 0.00003125)
    assert result["blade_occupied_fraction"] == result["blade_loaded_fraction"] == 0
    assert result["peak_sum_normal_force_n"] == result["peak_sum_shear_magnitude_n"] == 0


def test_episode_peak_persists_without_fabricating_current_contact():
    hit = media.tactile_bin([{"contacts": [contact()]}], np.zeros((1, 6)), 0.01)
    empty = [{"contacts": []}]
    after = media.tactile_bin(empty, np.zeros((1, 6)), 0.01, hit["episode_peak_sum_normal_force_n"])
    assert after["episode_peak_sum_normal_force_n"] == 3
    assert after["peak_sum_normal_force_n"] == after["blade_loaded_fraction"] == 0
    assert media.tactile_bin(empty, np.zeros((1, 6)), 0.01)["episode_peak_sum_normal_force_n"] == 0


def test_all_fixed_clips_including_failures_are_selected(monkeypatch):
    rows, comparisons = [], []
    for hand in media.SELECTION["hands"]:
        for offset in media.SELECTION["offsets_m"]:
            identity = dict(hand=hand, controller="ppo", offset_m=offset, seed=4301)
            rows.append(
                dict(
                    engine="mjbatch",
                    sim_dt=0.00003125,
                    outcome=dict(**identity, passed=False, failures=["speed"]),
                )
            )
            comparisons.append(dict(**identity, qualified=False))
    report = dict(rows=rows, comparisons=comparisons)
    monkeypatch.setattr(media, "summarize", lambda _: comparisons)
    selected = media.select_rows(report)
    assert len(selected) == 6 and all(not row["outcome"]["passed"] for row, _ in selected)
    corrupt = copy.deepcopy(report)
    corrupt["rows"][0]["outcome"]["seed"] = 4302
    with pytest.raises(ValueError, match="missing or duplicated"):
        media.select_rows(corrupt)


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("exit_velocity", [None, 0.7])
def test_overlay_fits_and_distinguishes_training_and_transfer(hand, exit_velocity):
    tactile = media.tactile_bin([{"contacts": [contact()]}], np.ones((1, 6)), 0.01)
    lines = media.labels(
        hand,
        -0.12,
        2.0,
        1.12,
        tactile,
        {"passed": False, "first_separation_ball_vx_m_s": exit_velocity},
        False,
    )
    image = media.annotate(np.zeros((540, 960, 3), dtype=np.uint8), lines)
    assert image.shape == (800, 960, 3)
    assert "untrained left" in lines[0] if hand == "left" else "right-trained" in lines[0]
    assert "not learned bowling" in lines[-1]
    assert "paired qualification: FAIL" in lines[-2]
    expected = "none" if exit_velocity is None else "+0.700"
    assert "current vx=+1.12" in lines[2] and f"first-exit vx={expected}" in lines[2]
    assert "Prior 10 ms: touch=" in lines[3]
    assert "impulse magnitude " in lines[4]
