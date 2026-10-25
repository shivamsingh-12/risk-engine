**Experiment 6a: Monte Carlo Convergence of an Option Price under N, with Variance Reduction**

**Objective**  
Empirically confirm that the Monte Carlo price error shrinks like 1/sqrt(N) (log-log slope of -0.5), and that antithetic sampling and a control variate change the constant in front of that rate without changing the rate itself.

**Methodology**  
A European call (spot 100, strike 100, rate 5%, vol 20%, one year) is priced by `qre::montecarlo::mc_european_price` using the exact lognormal terminal step. For each N in {100, 1,000, 10,000, 100,000} and each of four methods (plain, antithetic, control variate, both), 100 independent replications are run, each with its own seed spawned from one master `Rng`. Error is the difference from the Black-Scholes closed form (Phase 5). The RMSE over replications estimates the standard error of the estimator, and a least-squares line through log(RMSE) against log(N) gives the empirical convergence rate. Replicating each N fixes the weakness noted in Experiment 1, where a single draw per N produced non-monotone error.

C++ only: Black-Scholes is the independent ground truth, so a second Monte Carlo implementation would add nothing. The Python reference (`gbm.py`) is tested for the same 1/sqrt(N) scaling in its pytest suite.

**Results**

RMSE of the price (exact price 10.4506):

| N | Plain | Antithetic | Control variate | Both |
| :---- | :---- | :---- | :---- | :---- |
| 100 | 1.4349 | 1.0331 | 0.5550 | 0.3099 |
| 1,000 | 0.4117 | 0.3086 | 0.1591 | 0.0766 |
| 10,000 | 0.1530 | 0.0993 | 0.0529 | 0.0298 |
| 100,000 | 0.0514 | 0.0305 | 0.0191 | 0.0097 |
| **Log-log slope** | **-0.477** | **-0.508** | **-0.487** | **-0.492** |

*Theoretical expectation: slope -0.5, i.e. a 3.16x error reduction per 10x increase in N.*

Mean standard error reported by the estimator versus the measured RMSE at N = 100,000: plain 0.0466 vs 0.0514, antithetic 0.0329 vs 0.0305, control variate 0.0177 vs 0.0191, both 0.0087 vs 0.0097.

**Discussion**  
**The 1/sqrt(N) rate holds for every method.** All four slopes sit within 0.025 of -0.5, and the per-decade error ratios cluster around 3.16 (range 2.6 to 4.0). Variance reduction moved each line down without tilting it, which is the expected behavior: it reduces the variance constant, not the order of convergence. Getting a 10x smaller error still costs 100x more paths, so variance reduction is the only way to buy accuracy without paying quadratically.

**Variance reduction is worth the equivalent of orders of magnitude in N.** At N = 100,000, plain MC reaches an RMSE of 0.051, while the combined method reaches 0.0097 with the same number of paths, about 5.3x smaller error (28x smaller variance). Plain MC would need roughly 2.8 million paths to match it.

**Small-sample caveat for the control variate.** At N = 100 the combined method reports a mean standard error of 0.256 against a measured RMSE of 0.310. The control-variate coefficient is estimated from the same 50 pair-averages it is applied to, which introduces a small bias and makes the reported standard error optimistic. The effect disappears by N = 1,000 (0.086 vs 0.077), but it is a reason not to trust control-variate intervals at very small N.

**Limitations.** One option, one strike, 100 replications per cell, so each RMSE has roughly 7% relative sampling error. That is enough to confirm the slope, not to rank the methods precisely; Experiment 6b does the cross-strike comparison.

**Conclusion**  
The Monte Carlo engine converges at the theoretical rate and its standard errors track the measured error. No evidence of an implementation defect was found.

**File Paths**  
/cpp/experiments/01-mc-convergence/mc_convergence.cpp and csv with results in /cpp/experiments/01-mc-convergence/results/
