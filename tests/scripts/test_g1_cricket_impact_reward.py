"""Solved-phase kinematics, impulse signs and actual reward sampling."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_g1_cricket_impact_reward import (
    collect_episode,
    point_velocity,
    reward_readout,
    solved_impact,
)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("ball_type", ["sphere", "box"])
def test_solved_impulse_sign_and_pre_velocity(axis, ball_type):
    other = "box" if ball_type == "sphere" else "sphere"
    position = np.zeros(3)
    position[axis] = 0.055
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><option timestep="0.0005" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <geom name="bat_blade" type="{other}" size=".02 .02 .02" friction="0 0 0"/>
        <body pos="{" ".join(map(str, position))}"><freejoint name="ball_free"/>
          <geom name="ball_geom" type="{ball_type}" size=".036 .036 .036"
                mass=".156" friction="0 0 0"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[axis] = -2
    initial = data.qvel[:3].copy()
    impulse = np.zeros(3)
    closing_seen = accelerated_seen = False
    for step in range(40):
        pre_qvel = data.qvel.copy()
        pre = data.qvel[:3].copy()
        mujoco.mj_step(model, data)
        state = data.qpos.copy(), data.qvel.copy(), data.sensordata.copy()
        sample = solved_impact(
            model, data, model.geom("ball_geom").id, model.geom("bat_blade").id, pre, step
        )
        np.testing.assert_array_equal(sample["pre_ball_velocity_world_m_s"], pre)
        np.testing.assert_array_equal(sample["post_ball_velocity_world_m_s"], data.qvel[:3])
        for contact in sample["contacts"]:
            impulse += np.array(contact["force_on_ball_world_n"]) * model.opt.timestep
            jac = np.empty((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, np.array(contact["position_world_m"]), 1)
            np.testing.assert_allclose(
                contact["ball_point_velocity_world_m_s"], jac @ pre_qvel, atol=1e-12
            )
            closing_seen |= contact["relative_normal_velocity_m_s"] < 0
            accelerated_seen |= not np.array_equal(pre, data.qvel[:3])
        for actual, expected in zip((data.qpos, data.qvel, data.sensordata), state, strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert closing_seen and accelerated_seen
    np.testing.assert_allclose(impulse, 0.156 * (data.qvel[:3] - initial), atol=1e-10, rtol=1e-10)


def test_contact_point_velocity_uses_geom_origin_and_pre_solve_motion():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><option timestep=".001" gravity="0 0 -9.81" integrator="implicitfast"/>
      <worldbody><body pos="0 0 2"><freejoint/>
        <inertial pos=".2 .1 .05" mass="1" diaginertia=".1 .2 .3"/>
        <geom name="bat" type="box" pos=".3 -.1 .2" size=".02 .05 .2"/>
      </body></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    data.qvel[:] = [0.2, -0.3, 0.4, 0.5, 0.6, -0.7]
    pre = data.qvel.copy()
    mujoco.mj_step(model, data)
    geom = model.geom("bat").id
    point = data.geom_xpos[geom] + [0.02, 0.03, -0.04]
    expected = pre[:3] + np.cross(pre[3:], point - data.xpos[1])
    actual = point_velocity(model, data, geom, point)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    jac = np.empty((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, point, 1)
    np.testing.assert_allclose(actual, jac @ pre, atol=1e-12)
    assert np.linalg.norm(actual - jac @ data.qvel) > 0.001


@pytest.mark.parametrize("scale_by_dt", [False, True])
def test_reward_readout_does_not_invoke_reward_again(scale_by_dt):
    contact = np.zeros((1, 4, 17))
    term = SimpleNamespace(
        ball=SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_w=np.array([[2.0, 0, 0]]), root_link_pos_w=np.zeros((1, 3))
            )
        ),
        bat=SimpleNamespace(read=lambda: np.zeros((1, 3))),
        contact=SimpleNamespace(read=lambda: contact),
        hit_seen=np.array([True]),
        scored=np.array([True]),
    )
    rate = 5 * (1 + np.tanh(1)) / 0.02
    manager = SimpleNamespace(
        get_term_cfg=lambda _: SimpleNamespace(func=term, weight=1.0),
        get_active_iterable_terms=lambda _: [("batting", [rate])],
    )
    env = SimpleNamespace(
        reward_manager=manager, step_dt=0.02, cfg=SimpleNamespace(scale_rewards_by_dt=scale_by_dt)
    )
    result = reward_readout(env, (True, False))
    assert result["separation_reward_event"] and not result["touching"]
    assert result["separation_step_reward"] == rate * (0.02 if scale_by_dt else 1)
    assert term.scored[0] and term.hit_seen[0]


def test_episode_collection_keeps_all_ends_and_zero_load_occupancy():
    episodes = []
    contact = dict(
        normal_force_n=0.0, force_norm_n=0.0, force_on_ball_world_n=[0.0, 0.0, 0.0], distance_m=0.0
    )
    for step, records in enumerate(([contact], [], [contact], [])):
        collect_episode(episodes, {"solve_substep": step, "contacts": records}, 0.001)
    assert len(episodes) == 2
    assert [e["end"]["solve_substep"] for e in episodes] == [1, 3]
    assert all(e["contact_steps"] == 1 and e["loaded_steps"] == 0 for e in episodes)


def test_complete_impact_reward_evidence():
    from evaluate_g1_cricket_residual import sha256

    directory = ROOT / "g1_cricket_results/impact_v1"
    report = json.loads((directory / "impact_reward_audit.json").read_text())
    parent_path = directory / "trained_evaluation.json"
    parent = json.loads(parent_path.read_text())["reports"][1]
    assert report["parent_report_sha256"] == sha256(parent_path)
    assert report["checkpoint"] == parent["checkpoint"]
    assert report["checkpoint"]["sha256"] == sha256(ROOT / report["checkpoint"]["path"])
    assert report["versions"] == parent["versions"]
    for name in ("run_config", "run_summary"):
        assert report[f"{name}_sha256"] == sha256(directory / "right" / f"{name}.json")
    for name, expected in report["source_sha256"].items():
        assert sha256(ROOT / name) == expected
    assert len(report["rows"]) == 96
    assert [(r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in report["rows"]] == [
        (r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in parent["rows"]
    ]
    for row, old in zip(report["rows"], parent["rows"], strict=True):
        assert row["parent_row_exactly_reproduced"]
        samples = [
            dict(zip(row["control_sample_fields"], s, strict=True)) for s in row["control_samples"]
        ]
        assert len(samples) == 100
        assert [s["control_tick"] for s in samples] == list(range(100))
        assert [s["sensor_solve_substep"] for s in samples] == [
            160 * (i + 1) - 1 for i in range(100)
        ]
        before_hit = before_scored = False
        for sample in samples:
            assert sample["hit_seen_before"] == before_hit
            assert sample["scored_before"] == before_scored
            assert sample["hit_seen_after"] == (before_hit or sample["touching"])
            assert sample["scored_after"] == (
                before_scored or (sample["hit_seen_after"] and not sample["touching"])
            )
            assert sample["separation_reward_event"] == (
                not before_scored and sample["scored_after"]
            )
            before_hit, before_scored = sample["hit_seen_after"], sample["scored_after"]
            np.testing.assert_allclose(
                sample["native_step_reward"],
                sum(v for k, v in sample.items() if k.startswith("rate:")) * 0.02,
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                sample["approach_step_reward"] + sample["separation_step_reward"],
                sample["rate:batting"] * 0.02,
                rtol=1e-6,
                atol=1e-6,
            )
            expected_bonus = (
                sample["separation_reward_event"]
                * 5
                * (1 + np.tanh(sample["sampled_ball_vx_m_s"] - 1))
            )
            np.testing.assert_allclose(
                sample["separation_step_reward"], expected_bonus, rtol=1e-5, atol=1e-6
            )
        assert sum(s["native_step_reward"] for s in samples) == old["return"]
        assert row["paid_separation_events"] == sum(s["separation_reward_event"] for s in samples)
        assert row["loaded_episodes"] == sum(e["loaded_steps"] > 0 for e in row["episodes"])
        assert row["unobserved_loaded_episodes"] == sum(
            e["loaded_steps"] > 0 and not e["observed_control_ticks"] for e in row["episodes"]
        )
        for episode in row["episodes"]:
            assert episode["end"] is not None
            assert episode["duration_seconds"] == episode["contact_steps"] * 0.000125
            start, end = episode["start"]["solve_substep"], episode["end"]["solve_substep"]
            assert end - start == episode["contact_steps"]
            assert episode["last_contact"]["solve_substep"] == end - 1
            assert 0 <= episode["loaded_steps"] <= episode["contact_steps"]
            assert episode["observed_control_ticks"] == [
                s["control_tick"] for s in samples if start <= s["sensor_solve_substep"] < end
            ]
            for event in (episode["start"], episode["last_contact"], episode["end"]):
                assert event["solve_seconds"] == event["solve_substep"] * 0.000125
                assert event["integrated_seconds"] == (event["solve_substep"] + 1) * 0.000125
                for c in event["contacts"]:
                    normal = np.array(c["normal_on_ball_world"])
                    np.testing.assert_allclose(np.linalg.norm(normal), 1, atol=1e-12)
                    relative = np.array(c["ball_point_velocity_world_m_s"]) - np.array(
                        c["bat_point_velocity_world_m_s"]
                    )
                    np.testing.assert_allclose(c["relative_normal_velocity_m_s"], relative @ normal)
            assert (
                np.linalg.norm(episode["impulse_on_ball_world_ns"])
                <= episode["force_norm_integral_ns"] + 1e-12
            )
        assert sum(e["contact_steps"] for e in row["episodes"]) == old[
            "ball_contact_physics_step_counts"
        ].get("bat_blade", 0)
        paid = [s for s in samples if s["separation_reward_event"]]
        assert row["first_paid_ball_vx_m_s"] == (paid[0]["sampled_ball_vx_m_s"] if paid else None)
        assert row["physical_first_separation_vx_m_s"] == old["first_separation_ball_vx_m_s"]
    contacts = [r for r in report["rows"] if r["episodes"]]
    paid = [r for r in contacts if r["paid_separation_events"]]
    unpaid = [r for r in contacts if not r["paid_separation_events"]]
    assert len(contacts) == 40 and len(paid) == 31
    expected_unpaid = {("right", "ppo", -0.1, seed) for seed in range(4301, 4309)}
    expected_unpaid.add(("left", "ppo", 0.0, 4305))
    assert {
        (r["hand"], r["controller"], r["offset_m"], r["seed"]) for r in unpaid
    } == expected_unpaid
    assert all(r["episodes"][0]["loaded_steps"] for r in contacts)
    assert all(not r["episodes"][0]["observed_control_ticks"] for r in unpaid)
    assert sum(r["loaded_episodes"] for r in contacts) == 240
    assert sum(r["unobserved_loaded_episodes"] for r in contacts) == 156
    assert (
        max(abs(r["first_paid_ball_vx_m_s"] - r["physical_first_separation_vx_m_s"]) for r in paid)
        < 1e-8
    )
    assert report["physical_calibration_validated"] is report["policy_promoted"] is False
