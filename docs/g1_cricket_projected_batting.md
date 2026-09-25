# Two-Hand Batting Reference Closure

This offline correction preserves all 151 bat positions, orientations and times
from the retained supported swing. It projects robot hinges and root translation
onto an exact lower-hand grip and fixed initial foot transforms, retaining the
original joint limits, root orientation and ball state. Root translation is
bounded within 5 cm per axis; nearest-pose least squares weights translation
four times joint displacement. It never writes simulated states during rollout.

Kinematic closure is not physical feasibility. Initial foot compression is
preserved, not assumed to be dynamically equilibrated. Native inverse dynamics
is decomposed at the same pose into static, velocity-only and accelerating
conditions, with independent parked-ball DOFs excluded. Both 20 and 40 ms
derivative half-stencils and both physics resolutions are retained. Native
soft-contact correction is not evidence that an ideal load distribution exists
in the actual forward simulation; see the [MuJoCo computation reference](https://mujoco.readthedocs.io/en/stable/computation/index.html).

## Declared Experiment

The only changed axis is the whole-body reference projection. Freeze both final
ball/contact-observed PPO actors from `bimanual_batting_learning_v1`, gains,
reward, delivery, model, limits and zero motor lead. Evaluate all **16 outcomes**:
original/projected reference x right/left x reference-only/PPO x
31.25/15.625-microsecond physics. Every episode starts at time zero and lasts
three seconds unless terminated. No retraining, checkpoint selection, external
root force, impact-state write, relaxed gate or selected-row qualification.
The evaluator measures bat error against the **original** reference FK, not a
new easier trajectory. Preserve all failures and resolution comparisons.

```sh
PYTHONPATH=src:scripts uv run python scripts/project_g1_cricket_batting.py \
  g1_cricket_results/bimanual_projected_v1
PYTHONPATH=src:scripts uv run python scripts/audit_g1_cricket_batting_dynamics.py \
  g1_cricket_results/bimanual_projected_v1/original_inverse_dynamics.json.gz
PYTHONPATH=src:scripts uv run python scripts/audit_g1_cricket_batting_dynamics.py \
  g1_cricket_results/bimanual_projected_v1/projected_inverse_dynamics.json.gz \
  --right-reference g1_cricket_results/bimanual_projected_v1/right_reference.npz \
  --left-reference g1_cricket_results/bimanual_projected_v1/left_reference.npz
PYTHONPATH=src:scripts OMP_NUM_THREADS=2 uv run python scripts/evaluate_g1_cricket_tracking.py \
  g1_cricket_results/bimanual_batting_learning_v1/ppo_right \
  --output g1_cricket_results/bimanual_projected_v1/projected_right_fine \
  --reference-directory g1_cricket_results/bimanual_projected_v1 \
  --contact-dt 0.00003125 --bounced-delivery --compact-substeps
```

Repeat for both hands and resolutions; omit `--reference-directory` for each
`baseline_*` case. Generate `summary.json` using
`scripts/report_g1_cricket_projected_batting.py` and the candidate directory.
These are nominal-feed frozen-actor diagnostics, not robust learned interception
or running bowling. No new showcase video is qualified by projection alone.

## Reference and Inverse-Dynamics Results

All 302 handed reference frames pass kinematic closure. Maximum absolute
constraint residual is `1.9923e-13`; maximum joint change is 0.006301 rad and
maximum root translation is 0.000680 m. No unintended overlap is introduced.
The bat's full pose and timing are unchanged, including follow-through.

Each compressed inverse audit retains 1,208 rows, with three separate dynamic
conditions per row. At the 20 ms derivative half-stencil:

| Hand | Reference | Peak static motor-cap excess (Nm) | Peak dynamic motor-cap excess (Nm) | Peak dynamic root-force residual (N) |
| --- | --- | ---: | ---: | ---: |
| Right | Original | 117.290 | 127.576 | 707.920 |
| Right | Projected | 0 | 5.785 | 183.920 |
| Left | Original | 117.544 | 129.036 | 708.986 |
| Left | Projected | 0 | 6.072 | 184.050 |

These are required inverse loads, not measured rollout loads. Both physics
resolutions give the same inverse summaries. The projected reference still has
about 51 N static root-force residual, so zero static motor excess does not
establish supported equilibrium. Dynamic command-reachable force excess remains
15.684/15.597 Nm (right/left), distinct from motor-cap excess. At the wider 40 ms
derivative half-stencil, projected dynamic motor excess rises to 61.797/61.548 Nm.
This sensitivity is retained, not selected away: the result justifies separating
constraint correction from acceleration demand, not injecting these raw inverse
torques into the controller or declaring the motion physically qualified.

Full evidence: [projection](../g1_cricket_results/bimanual_projected_v1/projection.json),
[original inverse audit](../g1_cricket_results/bimanual_projected_v1/original_inverse_dynamics.json.gz),
[projected inverse audit](../g1_cricket_results/bimanual_projected_v1/projected_inverse_dynamics.json.gz).
