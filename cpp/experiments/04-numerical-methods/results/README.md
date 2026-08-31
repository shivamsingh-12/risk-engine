**Experiment 4: Characterizing Newton-Raphson Failure Modes**

**Objective:**  
Empirically trigger and distinguish Newton-Raphson’s three distinguishable failure modes using deliberately constructed test cases, demonstrate bisection and Brent’s methods do not fall the same way on those problems.

**Methodology:**

Three functions were chosen, each engineered specifically to trigger one specific failure mode in qre::optimization::newton\_raphson. (1) f(x) \= x^2 \-4, starting at x0 \= 0,0 where f’(0) exactly, a flat tangent point that is not itself a root, forcing an immediate derivative near zero check. (2) f(x) \= cbrt(x), started at x0 \= 1.0, a classic case where Newton’s tangent steps overshoot and diverge. (3) f(x) \= cos(x)-x, started at x0 \= 0,0 with max\_iterations artificially capped at 2, despite the function being well behaved with a real root near x=0.739. Each case was also run through a bisection and brent with a valid bracket around the true root as comparisons, showing the same problems do not break bracket-based methods. This was only implemented in C++, this experiment exercises specific behavior of solvers written for this project, cross language validation would not add correctness confidence.

**Results:** 

| Function | Solver | Converged | Iterations | Failure Reason |
| :---- | :---- | :---- | :---- | :---- |
| x²−4 | newton | false | 0 | DerivativeNearZero |
| x²−4 | bisection | true | 29 | None |
| x²−4 | brent | true | 7 | None |
| cbrt(x) | newton | false | 33 | Diverged |
| cbrt(x) | bisection | true | 1 | None |
| cbrt(x) | brent | true | 1 | None |
| cos(x)−x | newton | false | 2 | MaxIterations |
| cos(x)−x | bisection | true | 24 | None |
| cos(x)−x | brent | true | 4 | None |

**Discussion:**

**All three failure modes were triggered as designed, each cleanly attributable to a single cause.** Starting Newton at a flat tangent point produced instant DerivativeNearZero direction at iteration 0\. The cbrt(x) case diverged after 33 iterations, consistent with the well-known textbook result that Newton’s tangent steps overshoot near a cube-root singularity. Capped iteration budget on cos(x)-x produced MaxIterations as expected.

**Bisection and brent succeeded on every case Newton failed.**  This confirms the claim motivating the experiment: none of Newton’s failure modes are inherent to the underlying problems themselves, only to Newton’s specific reliance on a well behaved local derivative/good starting guess.

**Brent consistently require far fewer iterations than bisection when both succeeded** (7 vs 29 and 4 vs 24\)  which demonstrates why Brent is generally preferred in practice. The one exception of cube root where both converged in one iteration is more attributable to the symmetric bracket.

**Conclusion:**

Newton-Raphson’s three failure modes were successfully isolated and independently triggered through carefully constructed test cases; bracket based methods are immune to all three, at the cost of requiring a valid bracket rather than a single starting guess. Brent’s interpolation reduces iteration count versus bisection’s whenever the function’s local behavior permits it, supporting use as the default choice for calibration problems in later phases.

**File Paths:**

/cpp/experiments/04-numerical-methods for newton\_failure\_modes.cpp and csv with results
