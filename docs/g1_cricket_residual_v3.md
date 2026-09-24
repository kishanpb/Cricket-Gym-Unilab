# Residual v3: Remove The Miss Incentive

V2's signed separation reward produced zero blade contacts in all 48 PPO trials.
This reward-only follow-up replaces `10 * tanh(vx - 1)` with
`5 * (1 + tanh(vx - 1))` at the same first sampled blade separation. A miss earns
zero event reward; a backward or weak exit earns little; faster forward exits
earn monotonically more, bounded by 10. At vx=1 the shaping event pays 5, but
the evaluation still requires **strictly greater than 1 m/s** and all full gates.
Reward is not the success criterion. Original approach shaping is unchanged.

Preserve v1/v2 code and reports. Reuse v2 reset/one-shot state, but use a separate
reward owner so historical source hashes remain valid. Change no robot geometry,
collision, gain, action cap, observation, delivery, horizon, termination or
training hyperparameter. Train fresh right-hand seed-1 CPU PPO for exactly
199,680 transitions (4 x 24 x 2,080); retain the final checkpoint, not the best.

Re-evaluate all 96 development rows: zero residual/PPO x right/left x offsets
[-.12, -.10, 0] x seeds 4301-4308. These are reused development conditions, not
independent generalization evidence; left hand is untrained transfer. Require
exact comparator physics and unchanged v1 blade-first, forward-separation,
body/wicket/ground, stability, joint and actuator gates at every replayed 2 ms
step. A native/serial endpoint or named-sensor mismatch invalidates the audit.

If the reward again only buys contact, misses, or instability, retain all failures
and do not advertise selected frames. A successful development result still needs
new held-out deliveries, both-hand training, impact sensitivity, bowling,
the mjbatch integration and validated learned-control videos for the full goal.
