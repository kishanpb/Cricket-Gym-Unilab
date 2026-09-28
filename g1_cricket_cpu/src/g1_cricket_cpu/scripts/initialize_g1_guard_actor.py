"""Behavior-clone a qualified static guard into the native recurrent PPO actor."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from rsl_rl.runners import OnPolicyRunner
from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg
from unilab.base.config_adapter import BackendAdapter, create_env
from unilab.training import algo_config_dict
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def clone_guard(actor, observations, targets, epochs=1000):
    sequence = torch.as_tensor(np.asarray(observations), dtype=torch.float32).unsqueeze(1)
    desired = torch.as_tensor(np.asarray(targets), dtype=torch.float32).unsqueeze(1)
    parameters = [*actor.rnn.rnn.parameters(), *actor.mlp.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        hidden, _ = actor.rnn.rnn(sequence)
        predicted = actor.mlp(hidden)
        loss = (predicted - desired).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    actor.reset()
    return losses

def rollout(env, wrapper, actor=None):
    env.reset(seed=17)
    if actor is not None:
        actor.reset()
    model = env.get_playback_model()
    data = mujoco.MjData(model)
    action = env.action_manager.get_term('batting')
    observations, actions, rewards = ([], [], [])
    states = [env.get_physics_state_snapshot()[0]]
    maximum_grip = 0.0
    grip = env.termination_manager.get_term_cfg('grip_lost').func
    for step in range(round(env.cfg.max_episode_seconds / env.step_dt)):
        obs = wrapper.get_observations()
        if actor is None:
            mujoco.mj_setState(model, data, states[-1], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            action.controller.apply(data)
            command = (data.ctrl[action.controller.actuators[:12]] - action.legs.default) / action.legs.config['action_scale']
            command = command.astype(np.float32)[None, :]
        else:
            with torch.inference_mode():
                command = actor(obs).numpy()
        observations.append(obs['policy'][0].numpy().copy())
        state = env.step(command)
        actions.append(command[0].copy())
        rewards.append(float(state.reward[0]))
        states.append(env.get_physics_state_snapshot()[0])
        maximum_grip = max(maximum_grip, float(np.linalg.norm(grip.errors(), axis=-1).max()))
        if state.terminated[0] or state.truncated[0]:
            break
    return ({'seconds': (step + 1) * env.step_dt, 'completed_without_termination': bool(state.truncated[0]) and (not bool(state.terminated[0])), 'termination_flags': {name: bool(env.termination_manager.get_term(name)[0]) for name in env.termination_manager.active_terms}, 'maximum_grip_error_m': maximum_grip}, dict(observations=observations, actions=actions, states=states, rewards=rewards))

def initialize(output, teacher):
    audit = json.loads(teacher.read_text())
    if not audit['qualified_teacher']:
        raise ValueError('static guard teacher did not pass physical qualification')
    for path, expected in audit['source_input_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'teacher source/input changed: {path}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fingerprints = {**audit['source_input_sha256'], **{str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), teacher, ROOT / 'src/unilab/training/g1_locomotion.py', *sorted((ROOT / 'src/unilab/conf/ppo/task/g1_cricket_loaded_balance').glob('*.yaml')), ROOT / 'src/unilab/conf/ppo/task/g1_cricket_articulated_batting/mujoco.yaml', ROOT / 'g1_cricket_results/locomotion_assets/motion.pt', ROOT / 'g1_cricket_results/locomotion_assets/g1.yaml']}}
    rows = []
    for hand in ('right', 'left'):
        with initialize_config_dir(config_dir=str(ROOT / 'src/unilab/conf/ppo'), version_base='1.3'):
            owner = compose('config', overrides=['task=g1_cricket_loaded_balance/mujoco', f'env.handedness={hand}'])
        override = BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()
        override['auto_reset'] = False
        env = create_env(owner, num_envs=1, env_cfg_override=override)
        try:
            torch.manual_seed(1)
            wrapper = RslRlVecEnvWrapper(env, device='cpu')
            config = normalize_ppo_train_cfg(algo_config_dict(owner))
            config['logger'] = 'none'
            runner = OnPolicyRunner(wrapper, config, log_dir=None, device='cpu')
            teacher_result, demonstrations = rollout(env, wrapper)
            np.savez_compressed(output / f'{hand}_teacher.npz', **demonstrations)
            if not teacher_result['completed_without_termination']:
                raise ValueError(f'native {hand} teacher failed: {teacher_result}')
            actor = runner.alg.actor
            losses = clone_guard(actor, demonstrations['observations'], demonstrations['actions'])
            result, trace = rollout(env, wrapper, actor)
            np.savez_compressed(output / f'{hand}_student.npz', **trace)
            checkpoint = output / f'{hand}_actor.pt'
            torch.save(actor.state_dict(), checkpoint)
            row = dict(hand=hand, teacher=teacher_result, student=result, epochs=1000, losses=losses, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'losses'}), flush=True)
        finally:
            env.close()
    for path, expected in fingerprints.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'source/input changed: {path}')
    report = {'scope': 'behavior-cloned static guard initialization, not PPO training or cricket', 'protocol': 'one 8s teacher trajectory per hand, 1000 fixed epochs, seed1, final weights only', 'generalization': 'same-start closed-loop development check only; no perturbation or batting qualification', 'ppo_resume': 'load actor weights only; critic and PPO optimizer remain fresh', 'both_hands_completed': all((row['student']['completed_without_termination'] for row in rows)), 'full_substep_grip_audit_required': True, 'source_input_sha256': fingerprints, 'rows': rows, 'artifact_sha256': {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--teacher', type=Path, default=ROOT / 'g1_cricket_results/guard_teacher_v1/summary.json')
    args = parser.parse_args()
    initialize(args.output.resolve(), args.teacher.resolve())
