"""Screen held-ball release opportunities; projections are not delivery results."""
import argparse
import inspect
import json
from pathlib import Path
import mujoco
import numpy as np
from g1_cricket_cpu.scripts.g1_cricket_delivery_trial import RETURN_Y, TARGET_POPPING_X
from .amp_bowling_probe import run
from .portable_scenes import compare_models
from .twist2_cpu_probe import fingerprint

def first_bounce(position, velocity, gravity, contact_height):
    position, velocity = (np.asarray(position), np.asarray(velocity))
    height = position[2] - contact_height
    if height <= 0 or gravity <= 0:
        raise ValueError('Projection requires a ball above the plane and downward gravity')
    duration = (velocity[2] + np.sqrt(velocity[2] ** 2 + 2 * gravity * height)) / gravity
    point = position + duration * velocity
    point[2] -= 0.5 * gravity * duration ** 2
    return (float(duration), point)

def projection_model(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ball, pitch, joint = (model.geom('ball_geom'), model.geom('pitch'), model.joint('ball_free'))
    dof = int(joint.dofadr[0])
    if pitch.type[0] != mujoco.mjtGeom.mjGEOM_PLANE or pitch.bodyid[0] != 0 or (not np.array_equal(data.geom_xmat[pitch.id].reshape(3, 3)[:, 2], [0, 0, 1])) or (ball.type[0] != mujoco.mjtGeom.mjGEOM_SPHERE) or np.any(ball.pos) or np.any(model.opt.gravity[:2]) or (model.opt.gravity[2] >= 0) or (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY) or model.opt.density or model.opt.viscosity or np.any(model.dof_damping[dof:dof + 6]) or model.body_gravcomp[ball.bodyid[0]]:
        raise ValueError('Model does not satisfy contact-free gravity-only projection assumptions')
    return (-float(model.opt.gravity[2]), float(data.geom_xpos[pitch.id, 2] + ball.size[0]))

def screen(folder, parent):
    summary = json.loads((folder / 'summary.json').read_text())
    previous = json.loads((parent / 'summary.json').read_text())
    parameters = inspect.signature(run).parameters
    changed = [name for name, param in parameters.items() if param.kind == inspect.Parameter.KEYWORD_ONLY and summary.get(name, param.default) != previous.get(name, param.default)]
    if changed != ['minimum_release_speed'] or summary['minimum_release_speed'] != 100:
        raise ValueError('Expected only the predeclared held-ball trigger change')
    expected = {(h, dt) for h in ('right', 'left') for dt in (6.25e-05, 3.125e-05)}
    for report in (summary, previous):
        if len(report['rows']) != 4 or {(r['hand'], r['timestep']) for r in report['rows']} != expected:
            raise ValueError('Complete bilateral resolution cohort required')
    rows = []
    for result in summary['rows']:
        hand, name = (result['hand'], result['case'])
        if result['release'] is not None:
            raise ValueError('Held-ball diagnostic unexpectedly released')
        model = mujoco.MjModel.from_xml_path(str(folder / f'{hand}.xml'))
        model_match = compare_models(model, mujoco.MjModel.from_xml_path(str(parent / f'{hand}.xml')))
        gravity, height = projection_model(model)
        data = mujoco.MjData(model)
        joint = model.joint('ball_free')
        q, v = (int(joint.qposadr[0]), int(joint.dofadr[0]))
        arm = [model.body(f'{hand}_{part}_link').id for part in ('shoulder_roll', 'elbow')]
        opportunities = []
        with np.load(folder / f'{name}.npz') as trace, np.load(parent / f'{name}.npz') as old:
            assert fingerprint(folder / f'{name}.npz') == result['trace']
            parent_row = next((r for r in previous['rows'] if r['case'] == name))
            assert fingerprint(parent / f'{name}.npz') == parent_row['trace']
            count = int(np.flatnonzero(old['released'])[0])
            np.testing.assert_array_equal(trace['states'][:count + 1], old['states'][:count + 1])
            for key in ('observations', 'actions', 'targets', 'balance_rows'):
                np.testing.assert_array_equal(trace[key][:count], old[key][:count])
            steps = int(np.searchsorted(old['metrics'][:, 0], parent_row['release']['time'], side='right'))
            np.testing.assert_array_equal(trace['metrics'][:steps], old['metrics'][:steps])
            assert not trace['released'].any()
            for index, state in enumerate(trace['states'][:-1]):
                if not 1.7 <= trace['swing_rows'][index, 0] <= 2.7:
                    continue
                mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_kinematics(model, data)
                position, velocity = (data.qpos[q:q + 3], data.qvel[v:v + 3])
                shoulder, elbow = data.xpos[arm]
                eligible = position[2] > shoulder[2] + 0.12 and elbow[2] > shoulder[2] and (velocity[0] > 6) and (abs(velocity[1]) < 2)
                if not eligible:
                    continue
                duration, point = first_bounce(position, velocity, gravity, height)
                opportunities.append(dict(time_s=float(state[0]), position=position.tolist(), velocity=velocity.tolist(), original_trigger=bool(velocity[2] < 0), flight_time_s=duration, projected_first_bounce=point.tolist(), projected_bounce_in_zone=bool(4 < point[0] < TARGET_POPPING_X and abs(point[1]) < RETURN_Y)))
        rows.append(dict(case=name, retained_failures=result['failures'], model_match=model_match, prefix_controls=count, prefix_physics_samples=steps, original_trigger_count=sum((r['original_trigger'] for r in opportunities)), optimistic_trigger_count=len(opportunities), projected_zone_count=sum((r['projected_bounce_in_zone'] for r in opportunities)), opportunities=opportunities))
    report = dict(scope='Necessary contact-free first-bounce screen of every held-ball release-decision tick; not actual released flight, full qualification or a hardware limit', summary=fingerprint(folder / 'summary.json'), parent_summary=fingerprint(parent / 'summary.json'), analyzer=fingerprint(Path(__file__)), rows=rows, decision='release_window_needs_live_candidate' if any((r['projected_zone_count'] for r in rows)) else 'no_sampled_timing_only_first_bounce_solution', promotion_allowed=False)
    (folder / 'release_window.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('parent', type=Path)
    args = parser.parse_args()
    result = screen(args.folder, args.parent)
    print(json.dumps(dict(decision=result['decision'], cases=len(result['rows']))))
