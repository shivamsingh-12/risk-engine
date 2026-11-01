**Experiment 10: Adaptive Volatility — Do EWMA and Filtered Historical Simulation Fix the Backtest?**

**Objective**  
The backtest in Experiment 9 found that the three equal-weight 250-day models breached in clusters: every 99% breach fell between April and September 2022, and there were none at all in 2023. The suspected cause was a window that reacts too slowly to volatility regimes. This experiment tests that diagnosis by adding two methods that adapt, measuring whether they fix it, and measuring what adapting costs.

**Methodology**  
Same setup as Experiment 9: real 6-stock equal-weight portfolio, hardened data, a 1-day forecast every trading day from 2021-12-31 to 2023-12-29 (502 days), each from the 250 returns before it (point-in-time). Two adaptive methods are added to the original three (historical, parametric, Monte Carlo):
- **EWMA** (RiskMetrics): a normal model whose covariance weights recent days more, with weight proportional to λ^age.
- **FHS** (filtered historical simulation, Hull-White): each past portfolio return is divided by the EWMA volatility forecast for its day, then rescaled by tomorrow's forecast before taking the empirical quantile. It keeps the historical distribution's shape, including fat tails, but at today's volatility.

λ = 0.94 (the RiskMetrics daily standard) for the main comparison, then a sweep over λ in {0.90, 0.94, 0.97, 0.99}. Alongside the Experiment 9 statistics (Kupiec, Christoffersen, conditional coverage, Z2 for ES), two cost measures are added: average VaR (how much capital the model asks for) and the standard deviation of VaR's day-to-day change (how much the number jumps).

C++ only: EWMA weights, covariance, the volatility path, FHS and EWMA-parametric VaR/ES are cross-validated against the Python reference in both test suites (agreement to 1e-14).

**Results**

99% VaR, λ = 0.94 (expected 5.0 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Cond. cov. p | Z2 (ES) | Mean VaR | Daily VaR change (sd) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 7 (7 / 0) | 0.40 | 0.66 | 0.64 | 0.51 | 2.49% | 0.027% |
| Parametric | 11 (11 / 0) | **0.020** | **0.018** | **0.004** | 1.34 | 2.46% | 0.013% |
| Monte Carlo | 11 (11 / 0) | **0.020** | **0.018** | **0.004** | 1.36 | 2.43% | 0.043% |
| EWMA | 7 (4 / 3) | 0.40 | 0.66 | 0.64 | 0.43 | 2.36% | 0.108% |
| FHS | 5 (3 / 2) | 0.99 | 0.75 | 0.95 | -0.03 | 2.65% | 0.135% |

97.5% VaR, λ = 0.94 (expected 12.6 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Cond. cov. p | Z2 (ES) | Mean VaR | Daily VaR change (sd) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 11 (11 / 0) | 0.65 | **0.018** | 0.056 | -0.03 | 2.18% | 0.025% |
| Parametric | 12 (12 / 0) | 0.87 | **0.027** | 0.087 | 0.14 | 2.08% | 0.011% |
| Monte Carlo | 12 (12 / 0) | 0.87 | **0.027** | 0.087 | 0.16 | 2.06% | 0.031% |
| EWMA | 14 (8 / 6) | 0.68 | 0.37 | 0.62 | 0.15 | 1.99% | 0.091% |
| FHS | 10 (4 / 6) | 0.45 | 0.52 | 0.61 | -0.19 | 2.06% | 0.103% |

Bold: rejected at the 5% level.

λ sweep at 99% (breaches 2022 / 2023, Christoffersen p, daily VaR change sd):

| λ | EWMA | FHS |
| :---- | :---- | :---- |
| 0.90 | 6 (3 / 3), p = 0.70, 0.175% | 5 (3 / 2), p = 0.75, 0.219% |
| 0.94 | 7 (4 / 3), p = 0.66, 0.108% | 5 (3 / 2), p = 0.75, 0.135% |
| 0.97 | 6 (5 / 1), p = 0.70, 0.056% | 4 (2 / 2), p = 0.80, 0.071% |
| 0.99 | 7 (7 / 0), p = 0.66, 0.022% | 5 (4 / 1), p = 0.75, 0.035% |

At 97.5%, EWMA with λ = 0.99 is again rejected for clustering (Christoffersen p = 0.039, 12 of 13 breaches in 2022); every other λ passes for both methods.

**Discussion**  
**The diagnosis was right: adaptivity removes the clustering.** With the same normal distribution and the same 250 days of data, EWMA takes parametric VaR from 11 clustered breaches (rejected on every test) to 7 breaches spread across both years (Kupiec p = 0.40, Christoffersen p = 0.66). The only change is which days count most. At 97.5%, where every equal-weight method failed the independence test, both adaptive methods pass comfortably (p = 0.37 and 0.52). So the 2022 failure was mostly a timing problem, not a distribution problem.

**FHS combines both fixes and is the best-calibrated model here.** It adapts to volatility like EWMA and keeps historical's fat tails. At 99% it produced 5 breaches against 5.0 expected, no clustering, and Z2 of -0.03. That makes it the first model in the project whose ES is on target out of sample, while historical, EWMA and parametric all understate it (Z2 from 0.43 to 1.34). With 502 days, these are directions rather than proofs (Experiment 9 showed how weak the tests are at this sample size), but every statistic points the same way.

**Better timing, not more capital.** EWMA's average 99% VaR (2.36%) is lower than equal-weight parametric's (2.46%), yet it breaches less. It asks for less when markets are calm and more when they are not. FHS asks for about 6% more on average than historical (2.65% vs 2.49%), and that buys calibrated ES.

**The cost is a jumpy number.** Adaptive VaR moves 5 to 8 times more from day to day (daily change sd 0.108% for EWMA vs 0.013% for parametric; 0.135% for FHS vs 0.027% for historical). For a risk limit or a capital charge, a number that swings daily is operationally painful: positions can breach a limit because the model moved, not because the portfolio did. That is the trade-off λ controls.

**λ trades responsiveness for stability, and the benefit survives a wide range.** From 0.90 to 0.97 both adaptive methods pass every test, and jumpiness falls by roughly 3x across that range. At λ = 0.99, roughly 100 days of effective memory, EWMA behaves like the equal-weight window again. All its 99% breaches return to 2022, and at 97.5% clustering is rejected again.

**The Christoffersen test misses slow bunching.** At 99%, both plain historical and λ = 0.99 EWMA pass Christoffersen (p = 0.66) even though every one of their breaches fell in 2022. The test is first-order: it asks whether a breach today makes one tomorrow more likely, and breaches a few weeks apart don't trigger it. Bunching over months is visible only in the by-year split, which is why the breakdown matters alongside the formal tests. λ around 0.97 looks like a reasonable middle on this data, but with breach counts this small the differences between 0.90 and 0.97 are within noise.

**Side note on Monte Carlo.** Its VaR moves about 3 times more day to day than parametric's (0.043% vs 0.013%) despite converging to the same answer. That is pure simulation noise from fresh random numbers every day, and a reason to fix the seed or raise the path count if Monte Carlo VaR is ever used for limits.

**Limitations.** One portfolio and one two-year window with a single regime shift. λ was swept on the same data it is judged on, so the "best" λ is in-sample. 502 days give only about 5 expected breaches at 99%, so differences of one or two breaches between methods are not meaningful. EWMA is zero-mean with a fixed decay; GARCH, which estimates its own decay and allows mean reversion, is the natural next model and needs a multi-parameter optimizer the engine does not have yet.

**Conclusion**  
Making volatility adaptive fixes the failure the backtest found. EWMA turns a rejected normal model into one that passes every test, without asking for more capital on average, and filtered historical simulation is the best-calibrated model in the project for both VaR and ES. The price is a far less stable daily number. For this portfolio, FHS with λ between 0.94 and 0.97 is the recommended default.

**File Paths**  
/cpp/experiments/adaptive-vol/adaptive_backtest.cpp and csv with results (method_comparison, lambda_sweep, daily_var99) in /cpp/experiments/adaptive-vol/results/
