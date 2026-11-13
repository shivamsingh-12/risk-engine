**Experiment 18 (Phase 13): An Adaptive Risk Engine — Can Scoring the Models Every Day Beat Any One of Them?**

**Objective**  
The engine now has eight VaR/ES models that pass their backtests, and Experiment 16 found them disagreeing by about ±9% on the same day. This experiment builds the obvious next step: a risk engine that runs every model side by side, scores each one daily on how its past forecasts turned out, and either picks the best or blends them. The questions are whether that adaptivity pays off, whether it tracks a change in market regime, and whether the data can even tell the models apart.

**Methodology**  
*Scoring.* A breach test says whether a model is acceptable, not which of two acceptable models is better. For ranking, the engine uses the FZ0 loss of Patton, Ziegel and Chen (2019), written for losses L, VaR v and ES e at confidence c (α = 1 − c):

FZ0 = 1{L > v}(L − v)/(α e) + v/e + ln e − 1

Its expected value is lowest at the true (VaR, ES) pair. ES has no such score on its own, but VaR and ES jointly do. Its differences depend only on forecast ratios, so volatile days do not dominate. The VaR-only quantile loss is also reported. Two series of scores are compared with the Diebold-Mariano test, using a Newey-West variance for autocorrelation.

*Engine.* Eight models (historical, parametric, EWMA, FHS, GARCH, FHS-GARCH, GARCH-t and conditional EVT) forecast on the usual rolling 250-day window. Each day, three rules combine them using only scores from earlier days:
- **Equal weight:** average all eight VaRs and ESs (no learning).
- **Select best:** use the model with the lowest mean FZ0 over the last 100 days.
- **Exponential weights:** weight each model by exp(−η × its summed FZ0 over the last 100 days), with η = 1. This sits between the other two: η → 0 gives equal weights, η → ∞ gives select best.

The first 100 forecast days only build up scores, and all methods are evaluated after that.

*Part 1, synthetic regime switch.* A single asset with three regimes of 750 days each. The first is calm (independent normal returns, 0.8% daily vol), the second clustered (GARCH(1,1) with α = 0.12, β = 0.86 and Student-t(5) shocks, 1.5% long-run vol), the third calm again. This gives 1,900 evaluated days of 97.5% forecasts, scored by regime, with how often the engine chose the static models (historical, parametric) early and late in each regime.

*Part 2, real data.* The 6-stock equal-weight portfolio (hardened data), 402 evaluated days at 97.5% and 99%. Each model and rule gets coverage tests, Z2, mean FZ0, mean quantile loss and a Diebold-Mariano test against the equal-weight blend.

*Part 3, sensitivity.* The adaptive rules with lookbacks of 50, 100 and 250 days and η of 0.1, 1 and 10 (real data, 97.5%).

C++ only: FZ0, quantile loss, Diebold-Mariano and the three rules match the Python reference. The Python tests confirm on a 2-million-draw sample that both scores are minimized at the true VaR and ES.

**A bug found on the way.** The first run of Part 1 showed GARCH-t breaching on 25% of calm days. On normal returns, its Student-t degrees of freedom ν drifted to about 10^16, since a normal is a t with infinite ν. At that size the C++ Student-t functions lose all precision and returned negative VaRs and ESs around 10^32. The likelihood's log-gamma terms also become noise, which pushes the optimizer around. ν is now capped at 1,000 in both languages: the t is effectively normal there (its 97.5% quantile is 0.1% above the normal one). Regression tests were added in both languages. The cap never binds on the real data: Experiments 12 and 16 reproduce byte for byte, and all results below are after the fix.

**Results**

Part 1: mean FZ0 by regime (lower is better; the best in each regime is marked):

| Method | Calm (400 days) | Clustered (750) | Calm again (750) | All 1,900 days | DM vs equal weight (p) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | -3.917 | -3.053 | -3.913 | -3.574 | 2.83 (0.005) |
| Parametric | **-3.942** | -3.075 | -3.875 | -3.573 | 2.56 (0.011) |
| EWMA | -3.840 | -3.365 | -3.943 | -3.694 | 1.26 (0.21) |
| FHS | -3.821 | -3.388 | -3.922 | -3.690 | 1.45 (0.15) |
| GARCH | -3.941 | -3.293 | -3.944 | -3.687 | 1.48 (0.14) |
| FHS-GARCH | -3.917 | -3.307 | -3.948 | -3.688 | 1.56 (0.12) |
| GARCH-t | -3.938 | -3.338 | -3.938 | -3.701 | 1.49 (0.14) |
| Conditional EVT | -3.929 | -3.322 | -3.937 | -3.693 | 1.55 (0.12) |
| **Equal weight** | -3.915 | **-3.395** | **-3.955** | **-3.726** | — |
| Select best | -3.929 | -3.281 | -3.952 | -3.682 | 1.59 (0.11) |
| Exponential weights | -3.932 | -3.301 | -3.949 | -3.690 | 1.48 (0.14) |

How often the engine relied on the static models (historical, parametric):

| Segment | Select best | Exponential weights (mean weight) |
| :---- | :---- | :---- |
| Calm | 53% | 37% |
| Clustered, first 250 days | 14% | 13% |
| Clustered, last 500 days | 27% | 27% |
| Calm again, first 250 days | 18% | 17% |
| Calm again, last 500 days | 49% | 37% |

Select best changed model 104 times in 1,900 days. In the clustered regime the static models breached on 4.1% to 4.3% of days against 2.5% expected. The adaptive models breached on 2.9% to 4.0%, and the equal-weight blend on 3.5%.

Part 2: real data, 402 days:

| Confidence | Method | Breaches (expected) | Kupiec p | Christoffersen p | Z2 | Mean FZ0 | DM vs equal (p) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 97.5% | Historical | 5 (10.1) | 0.07 | **0.0006** | -0.48 | -3.698 | 1.73 (0.08) |
| | Parametric | 6 | 0.16 | **0.002** | -0.32 | -3.645 | 1.56 (0.12) |
| | EWMA | 9 | 0.73 | 0.52 | -0.07 | -3.757 | 0.32 (0.75) |
| | FHS | 8 | 0.50 | 0.57 | -0.20 | -3.745 | 0.59 (0.56) |
| | GARCH | 9 | 0.73 | **0.012** | -0.10 | -3.762 | 0.35 (0.73) |
| | FHS-GARCH | 10 | 0.99 | 0.47 | -0.05 | -3.765 | 0.18 (0.85) |
| | GARCH-t | 8 | 0.50 | **0.007** | -0.22 | -3.756 | 0.89 (0.37) |
| | Conditional EVT | 6 | 0.16 | 0.67 | -0.38 | -3.764 | 0.25 (0.80) |
| | **Equal weight** | 7 | 0.30 | 0.62 | -0.31 | **-3.769** | — |
| | Select best | 8 | 0.50 | 0.57 | -0.19 | -3.743 | 0.97 (0.33) |
| | Exponential weights | 7 | 0.30 | 0.62 | -0.28 | -3.742 | 1.03 (0.30) |
| 99% | Best single (conditional EVT) | 3 (4.0) | 0.59 | 0.83 | -0.20 | -3.609 | 0.12 (0.91) |
| | Worst single (parametric) | 5 | 0.64 | **0.0006** | 0.27 | -3.472 | 1.50 (0.13) |
| | Equal weight | 3 | 0.59 | 0.83 | -0.26 | -3.614 | — |
| | Select best | 4 | 0.99 | 0.78 | 0.01 | -3.614 | -0.003 (1.00) |
| | **Exponential weights** | 4 | 0.99 | 0.78 | 0.01 | **-3.619** | -0.14 (0.89) |

At 97.5%, select best switched model 19 times in 402 days, and no model was chosen more than 23% of the time.

Part 3: real data, 97.5%, mean FZ0 for the adaptive rules (equal weight: -3.769):

| Lookback | Select best | Exp. weights η = 0.1 | η = 1 | η = 10 |
| :---- | :---- | :---- | :---- | :---- |
| 50 | -3.729 | **-3.771** | -3.745 | -3.732 |
| 100 | -3.743 | -3.760 | -3.742 | -3.742 |
| 250 | -3.756 | -3.763 | -3.750 | -3.755 |

**Discussion**  
**The simple average beat the adaptive rules.** On the synthetic data, equal weighting had the best score in the clustered regime, in the return to calm, and over the whole sample. Selecting the recent best model was worse than every one of the six adaptive models it was choosing between. On real data at 97.5%, the equal-weight blend again had the lowest FZ0, ahead of every individual model and both learning rules. At 99%, exponential weights edged it, by 0.005, with a Diebold-Mariano p-value of 0.89. This is the forecast-combination puzzle seen in macroeconomic forecasting. When candidate models are similar in quality, estimating which one is best adds more noise than it removes, while averaging cancels part of each model's own estimation error. Gentler learning did better: in Part 3, the rule closest to equal weights (η = 0.1) was the best adaptive setting at every lookback, and it was the only one that ever beat the plain average, by 0.002 at a 50-day lookback.

**Selection reacts to regimes, but too slowly and too noisily to profit.** When the clustered regime started, the share of days on which the engine used a static model fell from 53% to 14%. Once calm returned it climbed back to 49%, so the scores do carry the right signal. Two things limit it. First, a 100-day lookback means the engine reacts to the last regime's evidence: after the switch back to calm it kept favouring adaptive models for months. Second, within a regime the score differences between the good models are small next to their day-to-day noise. Select best changed model 104 times in 1,900 days, which mostly reflects noise rather than information. Even in the clustered regime, GARCH-family models did not beat EWMA and FHS. This is despite the data being generated by a GARCH-t process, because they have to estimate their parameters from a window that partly covers the calm regime.

**What the data can distinguish is static versus adaptive, not the adaptive models among themselves.** In the synthetic sample, 1,900 days were enough for Diebold-Mariano to reject the static models against the equal-weight blend (p = 0.005 and 0.011). The difference between any adaptive model and the blend was never significant (p 0.11 to 0.21). On the real 402 days nothing was significant at 5%; historical simulation came closest (p = 0.08). The coverage tests flag the same split from another angle. Historical and parametric fail Christoffersen's independence test at 97.5%, because their breaches cluster, as in Experiments 9 and 10. Every combination rule passed both coverage tests at both levels, without being built to. The blend also escapes the failures of its members: GARCH and GARCH-t fail independence at 97.5% (p = 0.012 and 0.007), but the blend containing them does not (p = 0.62).

**A combined engine is also a safer engine.** The GARCH-t bug shows why. A component that breaks in an unforeseen situation, here calm normal data, can be diluted by averaging or caught by the scores. A single-model engine passes it straight through. In the first (buggy) run, select best had stopped choosing GARCH-t in both calm regimes and exponential weights had cut it to under 4%, while the equal-weight blend inherited its errors (mean FZ0 of 14 and 26 instead of about -3.9). Scores are a cheap monitoring tool even when they are a poor selection tool.

**Limitations.** One synthetic path, one real portfolio over two fairly calm years, and one window length for the models. The combinations are linear in (VaR, ES). Score-optimal weights estimated by minimizing FZ0 directly, and regime-switching models (for example a hidden Markov model on volatility), are not tested. The Diebold-Mariano test relies on asymptotics, and with 402 days and rare breaches its p-values are approximate. The rules all weight on FZ0; the quantile loss gives the same ranking at the top but is not used for weighting.

**Conclusion**  
Running every model and scoring them daily is worth doing, for monitoring if nothing else: the scores separated static from adaptive models, tracked regime changes in the right direction, and isolated a broken component. Using the scores to pick a model each day did not pay. Equal weighting of the eight models gave the best or near-best score in every setting tested, passed every coverage test, and beat the learning rules in all but three of the settings tested, by margins no test could distinguish. With a few hundred days of data and several reasonable models, the engine cannot tell which adaptive model is best. The robust choice is to average them, and to use the scores to drop models that are clearly worse.

**File Paths**  
/cpp/experiments/10-adaptive-engine/adaptive_engine_experiment.cpp and csv with results (synthetic_regimes, synthetic_selection, synthetic_static_share, synthetic_overall, real_comparison, real_selection, sensitivity) in /cpp/experiments/10-adaptive-engine/results/
