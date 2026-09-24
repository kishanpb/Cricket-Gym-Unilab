# Residual v2: Forward Separation Reward

Reward-only follow-up to v1, frozen at commit
`5df377fdb8a0ada87c59e82b01452bc3fd27d36e`. V1 touched the blade in all 24
right-hand trials but passed zero shot gates. Mere contact is not the target.

The only owner change is the batting reward function. Keep the original approach
shaping, but replace the first-contact bonus with one event at the first sampled
separation after blade contact: `10 * tanh(ball_vx - 1)` total reward. Thus weak or
backward exits are penalized, velocity exactly 1 m/s gets no event reward, and
forward exits above 1 m/s earn positive reward. Divide by the control interval
inside the reward term to compensate native reward-time scaling. Recontacts do
not earn additional events. Partial resets clear only the corresponding rows.

Training still samples sensors every 20 ms and can miss short contacts; this
reward is not the 2 ms evaluation gate. No collision, reset, pose, controller,
observation, action cap, gain, geometry, opponent or success threshold changes.

Start a fresh right-hand CPU PPO run from seed 1, not v1's checkpoint. Keep
4 environments x 24 steps x 2,080 updates = 199,680 transitions and every v1
hyperparameter. Retain only the final checkpoint, full scalar trace, config,
summary and complete evaluation. Do not choose the best checkpoint.

Use the same 96 development rows: zero residual/PPO x right/left x three fixed
lanes [-.12, -.10, 0] x seeds 4301-4308. These lanes/seeds are reused for method
development, not a new independent generalization claim. Left is untrained
transfer. Require the unchanged blade-first, strict outgoing vx > 1 m/s, no
ball/body/handle/wicket or pre-hit ground contact, full two-second stability,
guard, joint and actuator gates from the v1 contract. Audit every physics step
only after exact replay/native endpoint and named-sensor agreement.

Record all outcomes even if the new reward makes the robot avoid the ball.
A higher training reward or more contacts is not promotion evidence. New held-out
delivery conditions, seeds, impact convergence and both-hand training remain
required before a showcase claim, as do bowling and the mjbatch integration.
