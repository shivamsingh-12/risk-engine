**Experiment 5: Implied Volatility Recovery under Newton-Raphson, Bisection and Brent**

**Objective**  
Black-Scholes has no closed-form inverse for volatility, so implied vol is a calibration problem: find sigma such that BS(sigma) equals the observed price. This experiment measures how the three Phase 4 solvers behave on that problem across moneyness, volatility level and maturity, and where each one fails.

**Methodology**  
For each point on a grid, a European call is priced with `qre::pricing::price` at a known true volatility. Each solver then recovers the volatility from that price alone, and the absolute error against the true value is recorded. Grid: spot 100, rate 5%, moneyness K/S in {0.7, 0.85, 1.0, 1.15, 1.3}, true vol in {0.05, 0.2, 0.5, 1.0, 1.5}, maturity in {0.1, 1.0} years. Of the 50 combinations, 5 were skipped because the call price is indistinguishable from intrinsic value, so no volatility can be extracted, leaving 45 inversions. All solvers use tolerance 1e-10. Newton-Raphson starts at 0.2 with a central finite-difference derivative (`calibrate_newton`); bisection and Brent use the bracket [1e-6, 5.0].

C++ only: this tests how the Phase 4 solvers behave on the Phase 5 pricer (iteration counts and failure modes), not a mathematical fact, so no independent library implementation would be a meaningful cross-check. Correctness of the pricer itself is covered by the Python reference and test suites.

**Results**

| Solver | Converged (of 45) | Failures | Mean iterations | Max iterations | Max abs. error |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Newton-Raphson | 33 | 12 | 3.5 | 13 | 8.9e-09 |
| Bisection | 45 | 0 | 34.9 | 40 | 1.1e-04 |
| Brent | 45 | 0 | 8.8 | 18 | 8.6e-05 |

Ten of Newton's converged runs took 0 iterations because the true vol of 0.2 equals its starting guess. Restricted to the 23 points where Newton converged from a guess that was not already correct:

| Solver | Mean iterations | Max iterations | Max abs. error |
| :---- | :---- | :---- | :---- |
| Newton-Raphson | 5.0 | 13 | 8.9e-09 |
| Bisection | 36.6 | 40 | 1.1e-08 |
| Brent | 8.1 | 18 | 9.8e-12 |

**Newton-Raphson failures (12 of 45)**

| Failure | Count | Where |
| :---- | :---- | :---- |
| DerivativeNearZero | 6 | Maturity 0.1, moneyness 0.7 or 1.3, vol 0.5 to 1.5 |
| Pricer rejected a non-positive vol (exception) | 6 | True vol 1.0 or 1.5, far from the 0.2 start |

**Discussion**  
**Brent is the best default.** It converged on every point, needed roughly a quarter of bisection's iterations, and is within a factor of two of Newton's cost on the shared subset while never failing. This matches the Phase 4 result and is why `qre::pricing::implied_vol` uses Brent.

**Newton fails in two different ways, both predicted by the Phase 4 failure modes.** Where vega is tiny (short maturity, deep in or out of the money), the price is nearly flat in sigma, the finite-difference slope falls below the derivative threshold, and the step is undefined (`DerivativeNearZero`). Where the true vol is far above the starting guess, the tangent step overshoots below zero and the pricer throws `InvalidInputError`, because a non-positive volatility is invalid input. The exception is the intended behavior: the pricer fails loudly instead of returning a meaningless price.

**The tolerance is on price residual, not on volatility.** All solvers stop when |BS(sigma) - target| < 1e-10, which bounds the price error, not the volatility error. The two are linked by vega: sigma error is roughly residual / vega. The worst case, K/S = 0.7, sigma = 0.2, T = 0.1, has vega near 8e-7, so a 1e-10 price residual still permits about 1e-4 of volatility error (bisection 1.1e-04, Brent 8.6e-05). Implied vol is ill-conditioned wherever vega is small, regardless of solver. This is a property of the problem, not an implementation defect, but it means a tight price tolerance does not guarantee a tight volatility estimate there.

**Limitations.** The grid is coarse (45 inversions), the starting guess of 0.2 favors mid-range vols, and there is a single starting point per run. Newton's failure counts depend on that choice; a better initial guess (for example a Brenner-Subrahmanyam approximation) would likely remove the overshoot failures but not the low-vega ones.

**Conclusion**  
Brent's method is the right solver for implied volatility in this engine: robust across the whole grid, with a small iteration count. Newton is fastest when it works but fails in 27% of the grid, so it should only be used with a good initial guess and a fallback. Callers should treat implied vol near zero vega as low-confidence regardless of how small the residual is.

**File Paths**  
/cpp/experiments/04-greeks/implied_vol_solvers.cpp and csv with results in /cpp/experiments/04-greeks/results/
