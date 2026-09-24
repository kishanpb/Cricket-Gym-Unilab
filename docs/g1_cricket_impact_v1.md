# G1 bat/ball impact v1

The residual-v3 resolution audit found up to 28 mm penetration. Add a separate
G1CricketImpact owner, retaining the old default task and all historical reports.
Only the explicit ball/blade contact time constant changes from 20 to 4 ms;
the damping ratio, impedance, friction, margin, geometry and all other contacts
remain unchanged. The owner uses 0.5 ms physics and 20 ms control. This is a
numerical compliance correction, not experimentally calibrated cricket material.

Before training, evaluate the frozen final v3 PPO checkpoint and zero residual
on all 96 existing development rows at 0.5 and 0.25 ms. Keep all original shot,
stability, guarded-contact and joint/actuator gates; additionally require maximum
ball/blade penetration <=6 mm. Report all numerical changes, not just shots.
Re-run the original 96 v3 rows to prove that the shared factory's default path
has unchanged behavior. Historical source hashes refer to commit
d088364fe7caaf61f0a1922cd83a066cf7447495, not the corrected implementation.

No retraining or showcase promotion until both-hand frozen evaluations finish,
no new guard/early-termination failures arise, and every row passes the new
penetration bound. Shot failures remain failures. Compare contact-bearing paired
rows for penetration, peak load and first separation velocity using the previous
resolution tolerances (5%, floors 0.1 mm, 1 N, 0.05 m/s); no claim of numerical
impact convergence unless all compared rows pass. A failed convergence check
requires further resolution work before interpreting force peaks as stable.
