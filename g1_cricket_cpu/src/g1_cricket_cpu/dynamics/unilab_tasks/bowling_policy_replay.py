"""Replay a packaged SKRL bowling policy through UniLab, retaining all cases."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from unilab.base.env_factory import registry_env_factory
from ..amp_bowling_env import CASES
from ..groot_batting_ppo import Policy
from ..twist2_cpu_probe import fingerprint

def replay(env, policy, output):
    native = env.adapter.envs[0]
    rows, observations, means, rewards = ([], [], [], [])
    reset = native.reset

    def retain_reset(*args, **kwargs):
        if native.result is not None:
            path = output / f"{native.result['case']}.npz"
            records = dict(native.records, residual_observations=np.asarray(observations), policy_mean_rows=np.asarray(means))
            assert all((np.isfinite(value).all() for value in records.values()))
            np.savez_compressed(path, **records)
            rows.append(dict(result=native.result, trace=fingerprint(path), episode_return=native.episode_return, physics_samples=len(records['metrics'])))
            observations.clear()
            means.clear()
        return reset(*args, **kwargs)
    native.reset = retain_reset
    try:
        state = env.init_state()
        for step in range(400 * len(CASES)):
            observations.append(state.obs['obs'][0].copy())
            with torch.no_grad():
                action = policy.compute({'observations': torch.from_numpy(state.obs['obs'])})[0].numpy()
            means.append(action[0].copy())
            state = env.step(action)
            rewards.append(float(state.reward[0]))
            assert bool(state.terminated[0]) == (step % 400 == 399)
            assert not state.truncated.any() and np.isfinite(state.obs['obs']).all()
            if state.terminated[0]:
                row = rows[-1]
                with np.load(output / f"{row['result']['case']}.npz") as trace:
                    np.testing.assert_array_equal(rewards, trace['reward_rows'][:, 1])
                rewards.clear()
                assert state.info['case_info'][0]['result'] == row['result']
                print(f"{row['result']['case']}: qualified={row['result']['passed']}", flush=True)
        assert len(rows) == len(CASES)
        assert {(r['result']['hand'], r['result']['timestep']) for r in rows} == set(CASES)
        return rows
    finally:
        native.reset = reset
        env.close()

def run(runtime_bundle, groot_checkout, amp_checkout, scene_bundle, policy_bundle, output):
    torch.set_num_threads(1)
    started = time.monotonic()
    paths = dict(runtime_bundle=runtime_bundle, groot_checkout=groot_checkout, amp_checkout=amp_checkout, scene_bundle=scene_bundle, policy_bundle=policy_bundle)
    identity = fingerprint(policy_bundle / 'policy_manifest.json')
    factory = registry_env_factory('G1CricketBowlingPolicyCpu', 'mujoco')
    output.mkdir(parents=True, exist_ok=False)
    env = factory(1, {key: str(path.resolve()) for key, path in paths.items()})
    policy = Policy(env.observation_space, env.action_space)
    policy.load_state_dict(torch.load(policy_bundle / 'final.pt', weights_only=True)['policy'])
    policy.eval()
    rows = replay(env, policy, output)
    assert fingerprint(policy_bundle / 'policy_manifest.json') == identity
    report = dict(task='G1CricketBowlingPolicyCpu', rows=rows, profile=identity, checkpoint=fingerprint(policy_bundle / 'final.pt'), control_steps=1600, qualified_deliveries=sum((r['result']['passed'] for r in rows)), elapsed_s=time.monotonic() - started, promotion_allowed=False, versions={name: importlib.metadata.version(name) for name in ('unilab', 'mujoco', 'torch', 'skrl')}, scope='Retained SKRL policy replay through UniLab; no new training or hardware-limit claim')
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime_bundle', 'groot_checkout', 'amp_checkout', 'scene_bundle', 'policy_bundle', 'output'):
        parser.add_argument(name, type=Path)
    run(**vars(parser.parse_args()))
