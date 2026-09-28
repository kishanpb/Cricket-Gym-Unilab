"""Fixed initial-velocity screen for a saved guard actor; no training or shots."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from g1_cricket_cpu.scripts.initialize_g1_guard_actor import ROOT, digest, rollout
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.training import algo_config_dict
CASES = (('nominal', (0.0, 0.0)), ('positive_x', (0.02, 0.0)), ('negative_x', (-0.02, 0.0)), ('positive_y', (0.0, 0.02)), ('negative_y', (0.0, -0.02)))

def make_owner(hand, velocity):
    with initialize_config_dir(config_dir=str(ROOT / 'src/unilab/conf/ppo'), version_base='1.3'):
        owner = compose('config', overrides=['task=g1_cricket_loaded_balance/mujoco', f'env.handedness={hand}'])
    OmegaConf.update(owner, 'env.events.guard_perturb', {'func': 'unilab.envs.mdp.reset_root_state_uniform', 'mode': 'reset', 'params': {'pose_range': {}, 'velocity_range': {'x': [velocity[0], velocity[0]], 'y': [velocity[1], velocity[1]]}}}, force_add=True)
    return owner

def evaluate(output, parent):
    prior = json.loads((parent / 'summary.json').read_text())
    fingerprints = dict(prior['source_input_sha256'])
    fingerprints[str((parent / 'summary.json').relative_to(ROOT))] = digest(parent / 'summary.json')
    fingerprints[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    for name, expected in prior['artifact_sha256'].items():
        fingerprints[str((parent / name).relative_to(ROOT))] = expected
    for path, expected in fingerprints.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'source/input changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    for hand in ('right', 'left'):
        for label, velocity in CASES:
            owner = make_owner(hand, velocity)
            override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
            override['auto_reset'] = False
            env = create_env(owner, num_envs=1, env_cfg_override=override)
            try:
                torch.manual_seed(1)
                wrapper = RslRlVecEnvWrapper(env, device='cpu')
                config = normalize_ppo_train_cfg(algo_config_dict(owner))
                config['logger'] = 'none'
                runner = OnPolicyRunner(wrapper, config, log_dir=None, device='cpu')
                actor = runner.alg.actor
                actor.load_state_dict(torch.load(parent / f'{hand}_actor.pt', weights_only=True))
                result, trace = rollout(env, wrapper, actor)
                path = output / f'{hand}_{label}.npz'
                np.savez_compressed(path, **trace)
                row = dict(hand=hand, case=label, initial_world_velocity_m_s=velocity, result=result, trace=str(path.relative_to(ROOT)))
                rows.append(row)
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    for path, expected in fingerprints.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'source/input changed during evaluation: {path}')
    report = {'scope': 'fixed initial-velocity development screen, not full grip or batting qualification', 'protocol': 'both hands, nominal and +/-0.02m/s world x/y reset velocity, eight seconds, final actor weights, fresh memory per case', 'selection': 'all ten predeclared cases retained; no training or threshold changes', 'all_cases_completed': all((row['result']['completed_without_termination'] for row in rows)), 'full_substep_grip_audit_required': True, 'source_input_sha256': fingerprints, 'rows': rows, 'artifact_sha256': {path.name: digest(path) for path in output.iterdir() if path.is_file()}}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parent', type=Path, default=ROOT / 'g1_cricket_results/guard_aggregation_v1')
    args = parser.parse_args()
    evaluate(args.output.resolve(), args.parent.resolve())
