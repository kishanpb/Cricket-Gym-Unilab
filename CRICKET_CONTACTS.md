# Cricket Contact Diagnostics

Opt-in force and simulated-touch reporting for the existing external Gymnasium
cricket adapter. This is not a native Unitree task or a new learned-policy result.
The existing videos retain their original scripted-motion and PPO/A2C provenance.

## Reproduce

From this fork, use Python 3.13 and the pinned shared package:

```sh
uv sync --python 3.13 --extra mujoco
uv pip install 'cricket-gym[telemetry] @ git+https://github.com/kishanpb/gym-cricket.git@2a6641ddc030407b10e2320f07d6b88a92e23072'
uv run --no-sync python scripts/cricket_contact_report.py \
  --output cricket_contact_results/reference_diagnostics.json
```

The [retained report](cricket_contact_results/reference_diagnostics.json) contains
eight complete fixed-control episodes: batting and bowling, both hands, two seeds
each. Actions, seeds, runtime versions and source hashes are retained. This checks
instrumentation and reset behavior; it is neither an RL benchmark nor an estimate
of policy success. The separate mjbatch fork reports the full retained PPO/A2C pool.

The right-handed batting reference at seed 17000 reports a 54,712.67 N bat-blade
normal-force peak at a 2.5 microsecond adaptive step. It is retained as a model
diagnostic, not a credible real-world impact or safety measurement. All four bowling
reference episodes have zero load-bearing integrated samples; the two aimed wicket
contacts appear only in the separately labelled terminal forward solve. Neither
case is hidden or converted into an impulse estimate.

For native UniLab PPO runs, append `env.contact_telemetry=true` to either cricket
task command. Defaults remain off. Policy observations and reward functions are
unchanged; the telemetry is diagnostic information, not a policy input.

## Interpretation

Ball-versus-bat/pitch/outfield/stump channels report load-bearing touch, contact
count, normal/shear force and peaks (N). Active-substep traces include contact-frame
force (N), torque (N m), world position/frame vectors, distance (m), and timestep (s).
The adapter preserves each terminal trace across automatic reset; the next episode's
observation and the previous episode's telemetry must not be confused.

These are geometry-level simulated tactile states, not hardware taxels or calibrated
pressure. Grip load and dynamic foot support are unavailable in the scripted model.
RK4 sensor samples are not integrated impulses. Bowling can stop at a terminal
pose/forward contact solve before the next integration step: that solve appears only
under `terminal_forward`, not in integrated peaks or active counts.

The [shared signal contract](https://github.com/kishanpb/gym-cricket/blob/2a6641ddc030407b10e2320f07d6b88a92e23072/docs/contact_telemetry.md)
defines frames, reset semantics and limitations. Validation includes a 1 kg sphere
support-load fixture (9.81 N), sensor-on/off dynamics equality in both tasks/hands,
and ten UniLab adapter tests, including terminal telemetry and native PPO smoke.
The fixtures are CPU/macOS checks, not robot-safety or sim-to-real validation.

Run this fork's optional reporting and telemetry-enabled native PPO smoke tests
with `uv run --no-sync pytest tests/test_cricket_contacts.py -q` after installing
the package above and `pytest`. The PPO test checks two learner updates, not convergence.

## Next: Unitree Cricket

The planned extension uses the repository's Unitree G1 model through UniLab's native
task/backend sensor contracts. It will add bat and ball objects, physical support
and grip assumptions, joint-limit/stability tests, then bounded PPO training and
full fixed-seed evaluation before producing a showcase. Bowling must distinguish
learned release and locomotion from any temporary scripted curriculum assistance.
No trained G1 cricket result or new robot video is claimed yet.
