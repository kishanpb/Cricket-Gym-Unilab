# Next diagnostic: impact-to-reward timing

Status: design only; this audit has not run. Do not change the contact model,
reward, action space or checkpoint while collecting this evidence.

The corrected-contact PPO experiment does not establish successful batting.
The retained isolated probe also shows a dissipative response: at 0.0625 ms,
the fixed blade returns a 2.5 m/s ball at 0.335676 m/s (ratio 0.13427), with
15.9375 ms between first contact and separation. This is not a measured cricket
material model, nor proof that an actively moving bat cannot hit successfully.

## Declared comparison

Use the final impact-v1 checkpoint and zero residual, both hands, all three
lanes and seeds 4301-4308: the same 96 development rows, at 0.125 ms physics
and 20 ms control. Preserve complete two-second episodes and every failure.
Require exact native/serial state and named-sensor equality, and reproduce
the authoritative evaluation's returns, contact events and shot decisions.
Left-hand PPO remains untrained transfer.

For every blade-contact start and end, retain the pre/post ball velocity, blade
contact-point velocity, contact normal, signed world impulse on the ball,
penetration and contact duration. Separately retain each control sample's
actual `touching`, before/after `hit_seen` and `scored`, separation event and
batting reward, with reset boundaries explicit. Read the reward term's state
around its single native call; do not invoke it again or substitute a new reward
implementation. Identify the paid event from `scored` changing false to true.
Retain the RewardManager's cached contribution, weight and `scale_by_dt`:
weighted pre-dt rates are not the returned per-step reward. Reconcile the native
total and separate approach shaping from the one-shot separation bonus.

Contact forces belong to the pre-integration solve, while the integrated state
is one physics step later. Record both times rather than pairing a solved
contact with a post-step contact-point pose. Keep net signed impulse distinct
from the integral of force norm. Do not infer contact-point velocity solely
from the wrist or bat-center velocity.

Count all contact intervals and all reward events, including contacts wholly
between control samples. Distinguish geometric occupancy from positive-load
contacts and impulse, so zero-force chatter is not a missed physical hit.
Do not use the existing first-separation-only event list to reconstruct later
contact endings. Classify whether the reward missed a physical hit,
sampled a different separation velocity, or correctly observed a strike that
failed the shot gate. Report denominators by hand, controller and lane; do not
select favorable impacts. Close the audit on a reproduced explanation, not a
new reward curve or a successful-looking clip.

## Physical interpretation

MuJoCo's positive `solref` parameters specify time constant and damping ratio;
critical damping is not a measured ball coefficient of restitution. Its direct
negative format permits independent stiffness/damping identification, with
constraint-space scaling that must not be confused with force-space material
constants. [MuJoCo solver documentation](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters).

Rod Cross describes roughly 1 ms cricket impacts and an illustrative ball/steel
drop-test restitution near 0.58 at 6.26 m/s. These are contextual checks, not
calibration data for our slower toss, wooden bat or rigid robot fixture. Ball/steel
restitution is distinct from the apparent rebound ratio of a moving or compliant
bat. [Cross, Physics of Cricket, sections 3-6](https://physics.usyd.edu.au/~cross/cricket.html).

Allen et al. validated a ball model against fixed-surface experiments, then tested
freely suspended bats at 30 m/s; their rigid-body model did not match every blade
impact location. This supports keeping our rigid-blade simplification explicit,
not transferring one rebound coefficient to every contact. [Allen et al., 2014](https://shura.shu.ac.uk/8205/).

Any later material-response revision needs its own bounded impact validation
before robot retraining. Neither literature agreement nor this timing audit
constitutes physical force calibration, policy promotion or a showcase result.
