from pathlib import Path

from g1_cricket_cpu.dynamics.unilab_tasks import batting


def test_registered_factory_preserves_retained_recovery_fade(monkeypatch):
    monkeypatch.setattr(
        batting,
        "verify_bundle",
        lambda *args: {
            "environment_options": {"whole_body": True, "swing_power": 0.65},
            "references": ["data/right.npz", "data/left.npz"],
        },
    )
    calls = []
    adapter = object()

    def factory(*args, **options):
        calls.append((args, options))
        return lambda num_envs, overrides: adapter

    monkeypatch.setattr(batting, "batting_env_factory", factory)
    env = batting.make_cricket_env(
        batting.G1CricketResidualCfg(
            runtime_bundle="runtime", groot_checkout="groot", scene_bundle="scenes"
        )
    )
    assert env.adapter is adapter
    assert calls == [
        (
            (Path("groot"), Path("runtime/data"), Path("scenes/batting.xml")),
            {
                "seed": 29,
                "whole_body": True,
                "swing_power": 0.65,
                "physics_backend": "mujoco",
                "recovery_fade": True,
                "reference_files": [Path("runtime/data/right.npz"), Path("runtime/data/left.npz")],
            },
        )
    ]
