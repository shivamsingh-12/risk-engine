**Experiment 16 (Phase 11): Tail Risk with Extreme Value Theory, and How Uncertain Risk Numbers Are**

**Objective**  
Historical simulation can only reuse losses already observed, and normal models put too little mass in the tail. Extreme value theory (EVT) offers a third route: fit the shape of the tail beyond a high threshold and extrapolate from it. This experiment tests how well that works when the true tail is known, what it says about the real portfolio, whether a conditional version improves the backtest, and how much the engine's ten risk models disagree compared with the statistical uncertainty of any one of them.

**Methodology**  
*Part 1, known tail.* Losses drawn from a Student-t with 4 degrees of freedom: a heavy tail whose index ξ = 1/4 is known. 2,000 observations (about 8 years of daily data), 200 replications. VaR at 99%, 99.5% and 99.9% from three methods:
- historical (empirical quantile)
- normal (fitted mean and standard deviation)
- peaks-over-threshold: a generalized Pareto distribution (GPD) fitted by maximum likelihood to the losses above the 90th percentile.

Each is compared with the true quantile. The tail index is also estimated by the GPD at thresholds from the 80th to the 98th percentile, and by the Hill estimator using the 200 largest losses.

*Part 2, real data.* The same three methods on the 752 daily losses of the equal-weight 6-stock portfolio (hardened data), plus Hill at several k. 90% bootstrap intervals for historical VaR/ES at 99% come from both an iid bootstrap and a moving-block bootstrap with 10-day blocks, which preserves volatility clustering.

*Part 3, backtest.* Conditional EVT (McNeil-Frey: GARCH filter, GPD on the standardized residuals' losses, rescaled by the next-day GARCH volatility) uses the same rolling 250-day setup as Experiments 9-12 and is compared with FHS, FHS-GARCH and GARCH-t. The GPD tail index is recorded on every window.

*Part 4, model risk.* The next-day (2024-01-02) 99% VaR and ES from all ten models in the engine, each using the last 250 days:
- historical, parametric, Monte Carlo
- EWMA, FHS
- GARCH, FHS-GARCH, GARCH-t
- unconditional EVT, conditional EVT

The spread across models is compared with the block-bootstrap interval of the historical estimate.

C++ only: the GPD fit reaches the same optimum as the scipy-based Python reference (whose fit is checked against `scipy.stats.genpareto`), and POT, Hill and conditional EVT match the Python reference in both test suites. The GPD tail index is restricted to (-0.5, 0.95): below -0.5 maximum likelihood is irregular, and at 1 or above ES is infinite. How often the restriction binds is reported.

**Results**

Part 1: Student-t tail (true ξ = 0.25), bias and root-mean-square error relative to the true quantile:

| Confidence | True VaR | Historical bias / RMSE | Normal bias / RMSE | EVT (POT) bias / RMSE |
| :---- | :---- | :---- | :---- | :---- |
| 99% | 3.75 | -0.4% / 6.6% | -12.4% / 13.0% | +0.9% / 5.4% |
| 99.5% | 4.60 | -0.2% / 8.4% | -21.1% / 21.3% | +0.6% / 7.2% |
| 99.9% | 7.17 | -5.3% / 16.0% | -39.2% / 39.3% | -2.2% / 14.1% |

Tail index estimates (true 0.25):

| Estimator | Exceedances | Mean ξ | Standard deviation | Restriction binding |
| :---- | :---- | :---- | :---- | :---- |
| GPD, 80th percentile | 400 | 0.11 | 0.06 | 0 of 200 |
| GPD, 90th percentile | 200 | 0.14 | 0.09 | 0 |
| GPD, 95th percentile | 100 | 0.17 | 0.12 | 0 |
| GPD, 98th percentile | 40 | 0.13 | 0.22 | 1 |
| Hill, k = 200 | 200 | 0.42 | 0.03 | — |

Part 2: real portfolio losses (daily, % of portfolio value):

| Confidence | Historical VaR / ES | Normal VaR / ES | EVT VaR / ES |
| :---- | :---- | :---- | :---- |
| 99% | 2.52 / 2.89 | 2.31 / 2.64 | 2.51 / 2.89 |
| 99.5% | 2.72 / 3.14 | 2.55 / 2.87 | 2.81 / 3.13 |
| 99.9% | 3.29 / 3.43 | 3.06 / 3.34 | 3.34 / 3.56 |

The GPD fitted above the 90th percentile (76 exceedances) gives ξ = -0.22, a thin tail with an implied maximum possible daily loss of 4.58%. Hill gives ξ = 0.21, 0.33, 0.40 and 0.49 for k = 25, 50, 75 and 100. 90% bootstrap intervals for historical 99% VaR: iid 2.23% to 2.72%, block 2.11% to 2.72%; for 99% ES: iid 2.56% to 3.13%, block 2.50% to 3.14%.

Part 3: backtest, 502 days:

| Confidence | Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Z2 (ES) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 99% (5.0 expected) | FHS | 5 (3 / 2) | 0.99 | 0.75 | -0.03 |
| | FHS-GARCH | 5 (4 / 1) | 0.99 | 0.75 | 0.05 |
| | GARCH-t | 5 (5 / 0) | 0.99 | 0.75 | 0.03 |
| | Conditional EVT | 5 (4 / 1) | 0.99 | 0.75 | 0.06 |
| 97.5% (12.6 expected) | FHS | 10 (4 / 6) | 0.45 | 0.52 | -0.19 |
| | FHS-GARCH | 13 (7 / 6) | 0.90 | 0.41 | 0.02 |
| | GARCH-t | 12 (9 / 3) | 0.87 | **0.027** | -0.03 |
| | Conditional EVT | 8 (5 / 3) | 0.16 | 0.61 | -0.31 |

On the 502 windows (25 exceedances each), the conditional-EVT tail index had median -0.32 (10th to 90th percentile -0.50 to -0.10) and sat on the -0.5 bound on 117 windows (23%).

Part 4: next-day risk on 2024-01-02, ten models:

| Model | 99% VaR | 99% ES |
| :---- | :---- | :---- |
| Historical | 1.88% | 2.03% |
| Parametric | 1.88% | 2.15% |
| Monte Carlo | 1.86% | 2.13% |
| EWMA | 1.56% | 1.79% |
| FHS | 1.60% | 1.69% |
| GARCH | 1.74% | 1.99% |
| FHS-GARCH | 1.70% | 1.83% |
| GARCH-t | 1.78% | 2.08% |
| EVT (POT) | 1.85% | 2.00% |
| Conditional EVT | 1.68% | 1.81% |
| **Range across models** | **1.56% to 1.88%** | **1.69% to 2.15%** |
| **Historical, 90% block bootstrap** | **1.53% to 2.08%** | **1.66% to 2.11%** |

**Discussion**  
**EVT extrapolates better than history, and much better than the normal.** With a known heavy tail, peaks-over-threshold had the lowest error at every confidence level. The gain over historical simulation is modest at 99% (RMSE 5.4% vs 6.6%), where history still has 20 observations beyond the quantile, and persists at 99.9% (14.1% vs 16.0%), where it has only 2. The normal is not competitive: it understates the 99.9% quantile by 39% on average.

**But EVT's tail index is hard to estimate, even with 2,000 observations.** The GPD consistently underestimated ξ (0.11 to 0.17 against 0.25), with less bias at higher thresholds but much more variance (standard deviation from 0.06 to 0.22). Hill overestimated it (0.42). This is the bias-variance trade-off at the heart of EVT. The Student-t's tail only approaches the GPD shape slowly, so moderate thresholds include non-tail behaviour, while high thresholds leave too few points. The quantiles stayed accurate anyway, because within and just beyond the fitted range the scale β compensates for a wrong ξ. That compensation fails the further one extrapolates, which is exactly where EVT is supposed to help.

**On the real portfolio, the tail shape cannot be pinned down.** The GPD fit gives ξ = -0.22, a thin tail with a hard maximum daily loss of 4.58%. Large equity portfolios have lost more than that in single days (in March 2020, for example), so the implied bound is not credible. It reflects two years of data without a crash, not a property of the portfolio. Hill keeps rising with k (0.21 to 0.49) without the stable plateau that would indicate a reliable estimate. At 99% and 99.5% all three methods except the normal agree within about 3%. At 99.9% EVT and historical differ by only 1.5%, but both rest on a tail that two calm-ish years cannot really inform.

**Conditional EVT passes the backtest but adds nothing here.** At 99% it matches the other adaptive methods (5 breaches, Z2 = 0.06). At 97.5% it is conservative: 8 breaches against 12.6 expected and the most negative Z2 (-0.31). The likely reason is visible in its fits. With only 25 standardized exceedances per window, the tail index sat on its lower bound in 23% of windows and had a median of -0.32, systematically thinner than the Student-t fits of Experiment 12 suggested (ν ≈ 15, ξ ≈ 0.07). McNeil and Frey worked with 1,000-day windows and about 100 exceedances. With 250 days the GPD fit is mostly noise, and FHS-GARCH, which uses the residuals directly, does at least as well.

**The models disagree less than the data are uncertain.** On 2024-01-02 the ten models put next-day 99% VaR between 1.56% and 1.88%, about ±9% around the middle. The adaptive models are lower because volatility had fallen through late 2023. That disagreement is model risk. The 90% block-bootstrap interval for the historical estimate alone is wider: 1.53% to 2.08%, about ±15%. A single VaR number reported to three significant figures overstates what one year of data can support. Both kinds of uncertainty are of a similar order, and both should be reported alongside the point estimate. The block bootstrap was wider than the iid one (2.11% vs 2.23% lower bound) because volatility clustering makes extreme days arrive together, reducing the effective sample size. Separately, the Python tests showed that the percentile bootstrap under-covers in the tail: a nominal 90% interval contained the true 99% VaR about 86% of the time and the true ES about 84%. So these intervals, if anything, understate the uncertainty.

**Limitations.** Synthetic tails are Student-t only. The threshold is fixed at the 90th percentile for the real-data and backtest fits, with no automated threshold selection. Conditional EVT uses the same 250-day window as the other methods, which is short for EVT. The model-risk comparison is for one date in a calm period, and the spread is larger in stressed periods. The bootstrap treats the historical estimator only.

**Conclusion**  
EVT is the right tool for quantiles beyond the data: with a heavy tail it beat both history and the normal at every level tested, and the normal's understatement grows to almost 40% at 99.9%. Its weakness is the tail index, which is biased and noisy even with 2,000 observations and unidentifiable from two years of real data, where it implied an unrealistic maximum loss. In a rolling 250-day risk system, conditional EVT adds nothing over filtered historical simulation. The broader lesson of the phase is about honesty in reporting. On this portfolio the spread across reasonable models (about ±9%) and the sampling uncertainty of a single model (about ±15%, probably understated) are both large compared with the precision risk numbers are usually quoted to.

**File Paths**  
/cpp/experiments/09-evt/evt_experiment.cpp and csv with results (known_tail, threshold_sensitivity, real_tail, real_bootstrap, backtest, conditional_evt_xi, model_risk, model_risk_summary) in /cpp/experiments/09-evt/results/
