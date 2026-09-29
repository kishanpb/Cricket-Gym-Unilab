"""Source-bound zero-residual parity and bounded SKRL PPO training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import skrl
import torch
from torch import nn
from skrl.agents.torch.ppo import PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import TrainerCfg
from skrl.utils import set_seed
from g1_cricket_cpu.dynamics.groot_batting_env import ACTION_DIM, EPISODE_STEPS, GrootBattingEnv
from g1_cricket_cpu.dynamics.twist2_cpu_probe import fingerprint
from g1_cricket_cpu.dynamics.groot_batting_vector import make_vector

class Policy(GaussianMixin, Model):

    def __init__(self, observation_space, action_space, *, initial_mean=None, log_std=-3.0):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device='cpu')
        GaussianMixin.__init__(self, clip_actions=True)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, self.num_actions), nn.Tanh())
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)
        if initial_mean is not None:
            mean = torch.as_tensor(initial_mean, dtype=torch.float32)
            if mean.shape != (self.num_actions,) or not torch.all(torch.isfinite(mean) & (mean.abs() < 1)):
                raise ValueError('Initial mean must be finite and strictly inside the action bounds')
            with torch.no_grad():
                self.net[-2].bias.copy_(torch.atanh(mean))
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(log_std)))

    def compute(self, inputs, role=''):
        return (self.net(inputs['observations']), {'log_std': self.log_std})

class Value(DeterministicMixin, Model):

    def __init__(self, observation_space, action_space):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device='cpu')
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(nn.Linear(self.num_observations, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def compute(self, inputs, role=''):
        return (self.net(inputs['observations']), {})

def make_agent(env, steps, *, gae_lambda=0.95, rollouts=EPISODE_STEPS, initial_mean=None, log_std=-3.0, policy_class=Policy):
    cfg = dict(rollouts=rollouts, learning_epochs=4, mini_batches=4, learning_rate=0.0003, gae_lambda=gae_lambda, discount_factor=0.995, random_timesteps=0, learning_starts=0, time_limit_bootstrap=False, experiment=dict(write_interval=0, checkpoint_interval=0, wandb=False))
    agent = PPO(models={'policy': policy_class(env.observation_space, env.action_space, initial_mean=initial_mean, log_std=log_std), 'value': Value(env.observation_space, env.action_space)}, memory=RandomMemory(memory_size=rollouts, num_envs=getattr(env, 'num_envs', 1), device='cpu'), observation_space=env.observation_space, action_space=env.action_space, device='cpu', cfg=cfg)
    agent.init(trainer_cfg=TrainerCfg(timesteps=steps))
    return (agent, cfg)

def source_snapshot(baseline, output):
    parent = json.loads((baseline / 'summary.json').read_text())
    extra = [Path(__file__), Path('integrations/g1_dynamics/groot_batting_env.py'), Path('tests/test_groot_batting_env.py'), Path('tests/test_groot_batting_ppo.py'), Path('integrations/g1_dynamics/groot_batting_vector.py'), Path('tests/test_groot_batting_vector.py'), Path('tests/test_groot_whole_body.py'), baseline / 'summary.json']
    files = set(parent['source_input_fingerprints']) | {str(p) for p in extra}
    inputs = {p: fingerprint(Path(p)) for p in sorted(files)}
    with zipfile.ZipFile(baseline / 'source_bundle.zip') as archive:
        sources = set(archive.namelist())
    sources.update((str(p.resolve().relative_to(Path.cwd())) for p in extra))
    with zipfile.ZipFile(output / 'source_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.writestr(path, Path(path).read_bytes())
    return inputs

def verify_inputs(inputs):
    for name, expected in inputs.items():
        if fingerprint(Path(name)) != expected:
            raise ValueError(f'Source/input changed: {name}')

def parity_case(job):
    upstream, checkout, baseline, output, hand, recovery_fade, env_options = job
    started = time.monotonic()
    env = GrootBattingEnv(upstream, checkout, baseline / 'scene.xml', recovery_fade=recovery_fade, **env_options)
    env.reset(seed=0, options={'hand': hand, 'feed_index': 0})
    for _ in range(EPISODE_STEPS):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        assert not truncated
    assert terminated
    result = parity_result(env, baseline, output, started)
    env.close()
    return result

def parity_result(env, baseline, output, started):
    case = f'{env.hand}_feed{env.feed}'
    arrays = dict(states=env.states, controls=env.controls, policy=np.asarray(env.controller.policy_rows), ball_metrics=np.asarray(env.audit.metrics), invalid_ball_contacts=np.asarray(env.audit.invalid_metrics), ball_time_s=np.asarray(env.audit.sample_times), integrator_code=np.asarray(env.audit.integrator_codes, dtype=np.uint8))
    with np.load(baseline / f'{case}.npz') as old:
        assert set(arrays) == set(old.files)
        for name, array in arrays.items():
            np.testing.assert_array_equal(array, old[name], err_msg=f'{case}: {name}')
    old_rows = json.loads((baseline / 'summary.json').read_text())['rows']
    old_row = next((r for r in old_rows if r['case'] == case))
    assert all((env.result[k] == old_row[k] for k in env.result))
    assert env.tactile == json.loads((baseline / f'{case}_contacts.json').read_text())
    trace = output / f'{case}.npz'
    np.savez_compressed(trace, **arrays)
    result = dict(case=case, steps=EPISODE_STEPS, array_exact=True, physical_and_tactile_exact=True, wall_seconds=time.monotonic() - started, trace={'path': str(trace), **fingerprint(trace)})
    return result

def whole_body_options(baseline):
    parent = json.loads((baseline / 'summary.json').read_text())
    return {'whole_body': True, 'reference_files': [parent['reference_files'][hand] for hand in ('right', 'left')], 'condition_timing': parent['timing_mode'] == 'launch_velocity', 'swing_power': parent['swing_power']}

def parity(upstream, checkout, baseline, output, *, recovery_fade=False, workers=1, whole_body=False):
    if workers not in (1, 2, 4):
        raise ValueError('Expected one, two or four workers')
    output.mkdir(parents=True, exist_ok=False)
    inputs = source_snapshot(baseline, output)
    env_options = {}
    if whole_body:
        env_options = whole_body_options(baseline)
    started = time.monotonic()
    if workers == 1:
        jobs = [(upstream, checkout, baseline, output, hand, recovery_fade, env_options) for hand in ('right', 'left')]
        with ProcessPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(parity_case, jobs))
    else:
        raw, env = make_vector(upstream, checkout, baseline / 'scene.xml', workers=workers, seed=0, recovery_fade=recovery_fade, env_options=env_options)
        try:
            cases = [{'hand': hand, 'feed_index': feed} for feed in range(workers // 2) for hand in ('right', 'left')]
            env.reset(options={'cases': cases})
            for step in range(EPISODE_STEPS):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros((workers, *env.action_space.shape)))
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                assert bool(terminated.all()) == (step == EPISODE_STEPS - 1)
                assert not truncated.any()
            rows = list(raw.call('parity_record', baseline, output))
        finally:
            env.close()
    verify_inputs(inputs)
    report = dict(scope='Predeclared two-hand feeds, complete eight seconds; adapter parity only, not policy improvement', passed=True, recovery_fade=recovery_fade, workers=workers, environment_options=env_options, baseline=str(baseline.resolve()), action_dim=29 if whole_body else ACTION_DIM, wall_seconds=time.monotonic() - started, rows=rows, inputs=inputs)
    (output / 'parity.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'parity_passed': True, 'cases': len(rows)}), flush=True)

def train(upstream, checkout, baseline, parity_path, output, steps, seed, *, gae_lambda=0.95):
    if steps <= 0 or steps % EPISODE_STEPS:
        raise ValueError('Training budget must contain complete 400-step episodes')
    proof = json.loads(parity_path.read_text())
    workers = proof.get('workers', 1)
    if workers not in (1, 2, 4) or steps % (EPISODE_STEPS * workers):
        raise ValueError('Budget must contain complete synchronized worker episodes')
    expected = {f'{hand}_feed{feed}' for feed in range(max(1, workers // 2)) for hand in ('right', 'left')}
    if not proof['passed'] or {r['case'] for r in proof['rows']} != expected or len(proof['rows']) != len(expected):
        raise ValueError('Complete two-hand parity cases for every worker are required')
    env_options = proof.get('environment_options', {})
    if env_options or proof.get('action_dim', ACTION_DIM) != ACTION_DIM:
        if proof.get('action_dim') != 29 or env_options != whole_body_options(baseline) or proof.get('baseline') != str(baseline.resolve()):
            raise ValueError('Whole-body training must match its source-bound parity configuration')
    verify_inputs(proof['inputs'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    set_seed(seed)
    recovery_fade = proof.get('recovery_fade', False)
    if workers == 1:
        raw = GrootBattingEnv(upstream, checkout, baseline / 'scene.xml', recovery_fade=recovery_fade, **env_options)
        env = wrap_env(raw, wrapper='gymnasium', verbose=False)
    else:
        raw, env = make_vector(upstream, checkout, baseline / 'scene.xml', workers=workers, seed=seed, recovery_fade=recovery_fade, env_options=env_options)
    try:
        vector_steps = steps // workers
        agent, cfg = make_agent(env, vector_steps, gae_lambda=gae_lambda)
        before = {name: torch.cat([p.detach().flatten().clone() for p in model.parameters()]) for name, model in agent.models.items()}
        optimizer_steps = []
        hook = agent.optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(1))
        agent.enable_training_mode(True)
        obs, _ = env.reset()
        probes = [obs.clone()]
        updates = []
        for timestep in range(vector_steps):
            agent.pre_interaction(timestep=timestep, timesteps=vector_steps)
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=timestep, timesteps=vector_steps)
                next_obs, reward, terminated, truncated, info = env.step(actions)
                assert not truncated.any()
                assert bool(terminated.all()) == ((timestep + 1) % EPISODE_STEPS == 0)
                assert bool(terminated.any()) == bool(terminated.all())
                agent.record_transition(observations=obs, states=None, actions=actions, rewards=reward, next_observations=next_obs, next_states=None, terminated=terminated, truncated=truncated, infos=info, timestep=timestep, timesteps=vector_steps)
            agent.post_interaction(timestep=timestep, timesteps=vector_steps)
            if (timestep + 1) % EPISODE_STEPS == 0:
                episodes = [raw.episodes[-1]] if workers == 1 else list(raw.call('episode_record'))
                updates.append({'steps': (timestep + 1) * workers, 'optimizer_steps': len(optimizer_steps), **({'episode': episodes[0]} if workers == 1 else {'episodes': episodes}), 'tracking': {k: float(np.mean(v)) for k, v in agent.tracking_data.items() if len(v)}})
                agent.tracking_data.clear()
                probes.append(next_obs.clone())
                print(json.dumps({'steps': (timestep + 1) * workers, 'optimizer_steps': len(optimizer_steps), 'qualified_hits': sum((e['qualified_hit'] for e in episodes))}), flush=True)
            obs = next_obs
            if bool(terminated.all()) and timestep + 1 < vector_steps:
                obs, _ = env.reset()
        hook.remove()
        changes = {}
        for name, model in agent.models.items():
            vector = torch.cat([p.detach().flatten() for p in model.parameters()])
            assert torch.isfinite(vector).all()
            changes[name] = float(torch.linalg.vector_norm(vector - before[name]))
            assert changes[name] > 0
        assert len(optimizer_steps) == vector_steps // EPISODE_STEPS * 16
        agent.save(str(output / 'final.pt'))
        restored, _ = make_agent(env, vector_steps, gae_lambda=gae_lambda)
        restored.load(str(output / 'final.pt'))
        with torch.no_grad():
            probe = torch.cat(probes)
            original = agent.policy.compute({'observations': probe})[0]
            reloaded = restored.policy.compute({'observations': probe})[0]
            torch.testing.assert_close(original, reloaded, rtol=0, atol=0)
        verify_inputs(proof['inputs'])
        result = dict(scope='Bounded PPO optimization smoke, not a trained-policy performance claim; full-cohort deterministic evaluation pending', seed=seed, recovery_fade=recovery_fade, environment_options=env_options, action_dim=int(env.action_space.shape[0]), num_envs=workers, steps=steps, vector_steps=vector_steps, cfg=cfg, skrl_version=skrl.__version__, torch_version=torch.__version__, optimizer_steps=len(optimizer_steps), parameter_change_l2=changes, checkpoint_action_roundtrip_exact=True, checkpoint=fingerprint(output / 'final.pt'), updates=updates, parity={'path': str(parity_path), **fingerprint(parity_path)}, wall_seconds=time.monotonic() - started)
        (output / 'training.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    finally:
        env.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('parity', 'train'))
    parser.add_argument('upstream', type=Path)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parity', type=Path)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--recovery-fade', action='store_true')
    parser.add_argument('--whole-body', action='store_true')
    parser.add_argument('--gae-lambda', type=float, choices=(0.95, 1.0), default=0.95)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    if args.mode == 'parity':
        if args.gae_lambda != 0.95:
            parser.error('Set --gae-lambda on train; parity does not optimize a policy')
        parity(args.upstream, args.checkout, args.baseline, args.output, recovery_fade=args.recovery_fade, workers=args.workers, whole_body=args.whole_body)
    else:
        if args.workers != 1:
            parser.error('Training inherits worker count from --parity')
        if args.recovery_fade or args.whole_body:
            parser.error('Training inherits controller options from --parity')
        if args.parity is None:
            parser.error('Training requires --parity')
        train(args.upstream, args.checkout, args.baseline, args.parity, args.output, args.steps, args.seed, gae_lambda=args.gae_lambda)
