**Experiment 6b: Monte Carlo vs Black-Scholes across Moneyness, and Accuracy per Unit of Compute**

**Objective**  
Check that Monte Carlo agrees with the Black-Scholes closed form across strikes, that its reported standard errors are honest, and measure which variance-reduction method buys the most accuracy per unit of compute.

**Methodology**  
A one-year European call (spot 100, rate 5%, vol 20%) is priced at moneyness K/S of 0.8, 0.9, 1.0, 1.1 and 1.2. For each strike and each method (plain, antithetic, control variate, both), 400 independent replications of N = 5,000 paths are run, with per-replication seeds spawned from a master `Rng`. For each cell the experiment records the bias and RMSE against the closed form, the mean reported standard error, the coverage of the 95% interval (price +/- 1.96 se), the variance reduction factor (plain MSE divided by method MSE), the wall time per run, and the efficiency relative to plain MC, defined as 1 / (MSE x time) so that cheaper and more accurate both count.

C++ only: Black-Scholes is the independent ground truth. Wall-clock times depend on the machine and compiler, so compare ratios rather than absolute times.

**Results**

Variance reduction factor (higher is better) and efficiency relative to plain MC:

| K/S | Antithetic VRF | Control variate VRF | Both VRF | Antithetic eff. | Control variate eff. | Both eff. |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 0.8 | 8.2 | 64.0 | 561 | 12.3 | 56.3 | 770 |
| 0.9 | 4.1 | 17.7 | 202 | 5.9 | 16.0 | 265 |
| 1.0 | 2.2 | 6.8 | 29.2 | 2.9 | 6.2 | 37.2 |
| 1.1 | 1.4 | 3.8 | 32.1 | 1.9 | 3.4 | 39.9 |
| 1.2 | 1.1 | 2.1 | 55.4 | 1.6 | 2.0 | 77.8 |

Calibration of the 95% interval (fraction of 400 replications with the exact price inside it): 0.93 to 0.97 across all 20 cells (plain 0.93 to 0.96, antithetic 0.94 to 0.96, control variate 0.94 to 0.97, both 0.945 to 0.965).

Plain MC RMSE falls from 0.269 at K/S = 0.8 to 0.120 at K/S = 1.2. Bias was within 2 standard errors of the replicated mean in 19 of 20 cells. The one exception was antithetic at K/S = 0.8 (-0.011, or 2.3 standard errors); the largest absolute bias, -0.014 (antithetic at K/S = 1.0), was 1.96 standard errors.

**Discussion**  
**Monte Carlo agrees with Black-Scholes everywhere, and the standard errors are trustworthy.** Coverage of the nominal 95% interval is 93% to 97% in every cell, so the reported standard errors mean what they claim. With 20 cells and 400 replications, one bias at 2.3 standard errors is about what chance produces, not evidence of a defect, though 400 replications cannot rule out a bias much smaller than 0.01.

**The combination dominates, and the gain is largest in the wings.** Using both techniques gave the best accuracy per unit compute at every strike: 770x at K/S = 0.8 down to 37x at the money. In-the-money, the call payoff is close to linear in the terminal price, which is exactly what the control variate removes, and antithetic pairs cancel the linear part's noise entirely. At the money the payoff has a kink, so neither technique removes as much. The surprising part is that the two techniques are far stronger together than either alone (VRF 29 versus 2.2 and 6.8 at the money). A plausible reason, which this experiment did not isolate, is that averaging each antithetic pair makes both the payoff and the control even functions of the shock, which raises their correlation and lets the control variate work much harder.

**Antithetic sampling is cheap as well as accurate.** It halves the number of normal draws, so each run was 25% to 33% faster than plain MC here, which is why its efficiency gain exceeds its variance reduction factor. The control variate costs slightly more per run (an extra pass for the coefficient) but pays for it many times over.

**Limitations.** A single option type and volatility, European payoffs only, and the same master seed stream for all methods (so the methods see correlated random numbers, which makes cross-method comparisons less noisy but cells not fully independent). The efficiency numbers rest on single-machine timings of sub-millisecond runs.

**Conclusion**  
The engine prices European options correctly across strikes with calibrated standard errors, and combining antithetic sampling with a control variate is the right default: it was never worse than plain MC and was between 37x and 770x more efficient.

**File Paths**  
/cpp/experiments/03-bs-vs-mc/mc_vs_bs.cpp and csv with results in /cpp/experiments/03-bs-vs-mc/results/
