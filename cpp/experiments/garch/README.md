**Experiment 11: GARCH(1,1) — Does Estimating the Volatility Dynamics Beat Fixing Them?**

**Objective**  
Experiment 10 showed that adaptive volatility (EWMA, and filtered historical simulation built on it) fixes the clustered breaches of the equal-weight models. EWMA fixes its decay at λ = 0.94 and has no mean reversion. GARCH(1,1) estimates both from the data, and EWMA is the special case where persistence is exactly 1. This experiment asks whether that extra flexibility improves out-of-sample risk forecasts, and how reliable the estimates are on the windows a daily risk system would use.

**Methodology**  
*Part 1, backtest.* Same setup as Experiments 9 and 10: real 6-stock equal-weight portfolio, hardened data, a 1-day forecast every trading day from 2021-12-31 to 2023-12-29 (502 days), each from the previous 250 days only. Two GARCH methods are added and refitted every day on the 250-day portfolio return window:
- **GARCH-normal:** a normal distribution with the GARCH volatility forecast.
- **FHS-GARCH:** filtered historical simulation using GARCH volatilities instead of EWMA ones.

Estimation is variance-targeted maximum likelihood on zero-mean returns: the long-run variance is fixed to the window's mean squared return, and α and β are estimated through a reparameterization that keeps every candidate valid (α, β ≥ 0, α + β < 1). The optimizer is the engine's new Nelder-Mead, run from three starting points with the best kept. Baselines are historical, parametric, EWMA and FHS (λ = 0.94).

*Part 2, fit diagnostics.* For the 252 forecast days from 2022-12-29 onwards, GARCH was fitted on both a 250-day and a 500-day window, recording α, β and persistence (α + β) each day. A single fit on the whole history (752 returns) serves as a reference.

C++ only: Nelder-Mead and GARCH are cross-validated in the test suites. The C++ fit reaches the same optimum as scipy's Nelder-Mead (log-likelihood within 1e-6), and the Python fit agrees with the independent `arch` package to within 0.02 on α and β.

**Results**

99% VaR (expected 5.0 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Cond. cov. p | Z2 (ES) | Mean VaR | Daily VaR change (sd) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 7 (7 / 0) | 0.40 | 0.66 | 0.64 | 0.51 | 2.49% | 0.027% |
| Parametric | 11 (11 / 0) | **0.020** | **0.018** | **0.004** | 1.34 | 2.46% | 0.013% |
| EWMA | 7 (4 / 3) | 0.40 | 0.66 | 0.64 | 0.43 | 2.36% | 0.108% |
| FHS | 5 (3 / 2) | 0.99 | 0.75 | 0.95 | -0.03 | 2.65% | 0.135% |
| GARCH | 6 (5 / 1) | 0.67 | 0.70 | 0.85 | 0.27 | 2.36% | 0.290% |
| FHS-GARCH | 5 (4 / 1) | 0.99 | 0.75 | 0.95 | 0.05 | 2.53% | 0.342% |

97.5% VaR (expected 12.6 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Cond. cov. p | Z2 (ES) | Mean VaR | Daily VaR change (sd) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 11 (11 / 0) | 0.65 | **0.018** | 0.056 | -0.03 | 2.18% | 0.025% |
| Parametric | 12 (12 / 0) | 0.87 | **0.027** | 0.087 | 0.14 | 2.08% | 0.011% |
| EWMA | 14 (8 / 6) | 0.68 | 0.37 | 0.62 | 0.15 | 1.99% | 0.091% |
| FHS | 10 (4 / 6) | 0.45 | 0.52 | 0.61 | -0.19 | 2.06% | 0.103% |
| GARCH | 14 (10 / 4) | 0.68 | 0.054 | 0.14 | 0.14 | 1.99% | 0.244% |
| FHS-GARCH | 13 (7 / 6) | 0.90 | 0.41 | 0.70 | 0.02 | 2.02% | 0.267% |

Bold: rejected at the 5% level.

GARCH parameter estimates, forecast days 2022-12-29 to 2023-12-29 (median, 10th-90th percentile):

| Estimation window | α | β | Persistence | Share of fits with persistence > 0.99 | Day-to-day change in persistence (sd) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 250 days | 0.053 (0.021-0.076) | 0.931 (0.872-0.966) | 0.975 | 17.9% | 0.0265 |
| 500 days | 0.081 (0.069-0.098) | 0.877 (0.859-0.910) | 0.960 | 0% | 0.0010 |
| Full history (752 days) | 0.077 | 0.890 | 0.967 | — | — |

All 504 diagnostic fits (252 at each window length) reported convergence.

**Discussion**  
**GARCH passes, but it does not beat EWMA.** At 99%, GARCH-normal (6 breaches, every test passed) and FHS-GARCH (5 breaches, Z2 = 0.05) are as well calibrated as EWMA and FHS. At 97.5%, GARCH-normal is close to failing the independence test (p = 0.054) where EWMA passes comfortably. On ES, FHS-GARCH has the Z2 closest to zero at 97.5% (0.02) and FHS has it at 99% (-0.03); with about 5 to 13 expected breaches, differences of this size are noise. Estimating the dynamics bought no measurable improvement in forecast quality on this data.

**What it did buy is a much jumpier number.** GARCH VaR changes 2.7 times more from day to day than EWMA VaR (0.290% vs 0.108%), and FHS-GARCH 2.5 times more than FHS. EWMA's day-to-day moves come only from new returns. GARCH's also come from re-estimating α and β every day, so parameter noise leaks straight into the risk number.

**250 days is too short to pin GARCH down.** With a 250-day window, persistence moved 26 times more from one day to the next than with a 500-day window (sd 0.0265 vs 0.0010). The 10th-90th percentile range of α is three times wider (0.021-0.076 vs 0.069-0.098). On 17.9% of days persistence exceeded 0.99, where GARCH is effectively EWMA with an estimated λ. The 500-day window never got there, and its estimates sit close to the full-history fit (α 0.081 vs 0.077, β 0.877 vs 0.890). This is a well-known difficulty with GARCH: the likelihood is nearly flat in the persistence direction unless the sample contains several volatility episodes, and one year usually does not.

**The full-history fit validates EWMA's design choice.** Persistence of 0.967 means volatility shocks decay with a half-life of about 21 trading days, and the estimated α (0.077) is close to EWMA's 1 - λ = 0.06. EWMA with λ = 0.94 is therefore a sensible fixed approximation of what GARCH estimates. Its one structural difference, no pull back to a long-run level, matters for multi-day horizons more than for the 1-day forecasts tested here.

**Limitations.** Univariate GARCH on the portfolio return series only; a multivariate model (for example DCC) would also adapt correlations. Gaussian likelihood only (Student-t innovations are the usual next step for fat tails). One portfolio, one two-year period, 1-day horizon. Fits use three starting points, which found the same optimum as scipy in the tests, but Nelder-Mead gives no global guarantee.

**Conclusion**  
GARCH(1,1) works: it passes every backtest at 99% and its full-history estimates are sensible. On a 250-day window, though, its estimates are too noisy to beat a fixed EWMA decay, and that noise makes the daily VaR 2.5 to 3 times jumpier for no gain in calibration. For this engine's daily 1-day risk, FHS with fixed λ remains the recommended default. GARCH is worth revisiting with longer estimation windows, multi-day horizons where mean reversion matters, or fat-tailed innovations. Separately, the Nelder-Mead optimizer built for this step is now available for later calibration work.

**File Paths**  
/cpp/experiments/garch/garch_backtest.cpp and csv with results (method_comparison, fit_diagnostics, daily_garch_params) in /cpp/experiments/garch/results/
