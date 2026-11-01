**Experiment 12: Student-t GARCH — Do Fat-Tailed Shocks Fix the ES Understatement?**

**Objective**  
Every normal-based model so far has understated Expected Shortfall out of sample: positive Z2, meaning losses beyond VaR averaged more than the forecast ES. GARCH-normal had Z2 = 0.27 at 99%. GARCH explains part of the fat tails of returns through volatility clustering, since a mixture of calm and stressed days looks fat-tailed even if each day is normal. The question here is whether the shocks themselves are also fat-tailed, and whether modelling that with a Student-t fixes ES.

**Methodology**  
*Part 1, backtest.* Same setup as Experiments 9-11: real 6-stock equal-weight portfolio, hardened data, 502 daily 1-day forecasts from 2021-12-31 to 2023-12-29, each from the previous 250 days only. The new method is GARCH(1,1) with standardized Student-t shocks, refitted every day by maximum likelihood with the degrees of freedom ν estimated jointly with α and β (ν = 2 + e^c keeps the variance finite). VaR and ES come from the closed form for a scaled Student-t. Compared against historical, parametric, FHS (EWMA), GARCH-normal and FHS-GARCH.

*Part 2, tail diagnostics.* On the whole history (752 returns): the excess kurtosis of raw portfolio returns, the excess kurtosis after standardizing each return by its GARCH-normal volatility forecast, and the likelihood gained by the t model. Plus ν estimated on every 250-day backtest window.

C++ only: the Student-t cdf (implemented through the regularized incomplete beta function) matches scipy to 1e-14 and the quantile to 1e-10. The GARCH-t fit matches the Python reference, which agrees with the `arch` package. ES is verified by numerical integration in the Python tests.

**Results**

99% (expected 5.0 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Z2 (ES) | Mean VaR | Mean ES |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 7 (7 / 0) | 0.40 | 0.66 | 0.51 | 2.49% | 2.80% |
| Parametric | 11 (11 / 0) | **0.020** | **0.018** | 1.34 | 2.46% | 2.82% |
| FHS | 5 (3 / 2) | 0.99 | 0.75 | -0.03 | 2.65% | 3.10% |
| GARCH-normal | 6 (5 / 1) | 0.67 | 0.70 | 0.27 | 2.36% | 2.71% |
| FHS-GARCH | 5 (4 / 1) | 0.99 | 0.75 | 0.05 | 2.53% | 2.82% |
| GARCH-t | 5 (5 / 0) | 0.99 | 0.75 | 0.03 | 2.45% | 2.89% |

97.5% (expected 12.6 breaches):

| Method | Breaches (2022 / 2023) | Kupiec p | Christoffersen p | Z2 (ES) | Mean VaR | Mean ES |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 11 (11 / 0) | 0.65 | **0.018** | -0.03 | 2.18% | 2.51% |
| Parametric | 12 (12 / 0) | 0.87 | **0.027** | 0.14 | 2.08% | 2.48% |
| FHS | 10 (4 / 6) | 0.45 | 0.52 | -0.19 | 2.06% | 2.63% |
| GARCH-normal | 14 (10 / 4) | 0.68 | 0.054 | 0.14 | 1.99% | 2.37% |
| FHS-GARCH | 13 (7 / 6) | 0.90 | 0.41 | 0.02 | 2.02% | 2.48% |
| GARCH-t | 12 (9 / 3) | 0.87 | **0.027** | -0.03 | 2.01% | 2.48% |

Bold: rejected at the 5% level.

Tail diagnostics, full history:

| Quantity | Value |
| :---- | :---- |
| Excess kurtosis, raw portfolio returns | 1.09 |
| Excess kurtosis, GARCH-standardized residuals | 0.38 |
| Estimated ν, full history | 15.6 |
| Log-likelihood gain of t over normal | 1.80 (LR = 3.60, χ²(1) p = 0.058; about 0.029 allowing for ν = ∞ being a boundary) |
| ν on 250-day windows | median 17.6, 10th-90th percentile 12.0-47.4, above 30 on 23% of days |

(Kurtosis here uses population moments, so the raw figure differs slightly from the 1.08 reported in Experiment 7.)

**Discussion**  
**Volatility clustering explains most of the fat tails, but not all.** Standardizing returns by their GARCH volatility removes about two thirds of the excess kurtosis (1.09 to 0.38). Much of what looked like fat tails was really a mixture of calm and stressed periods. The remaining third is genuine tail fatness in the daily shocks, and the full-history fit puts ν at about 15.6. That is only mildly fat-tailed (single stocks commonly estimate lower values), which is plausible for a diversified six-stock portfolio, since averaging stocks thins the tails. The likelihood improvement is modest and borderline significant (p of about 0.03 to 0.06).

**Fat-tailed shocks fix ES.** GARCH-t has Z2 = 0.03 at 99% and -0.03 at 97.5%, the closest to zero of any model at both levels, against 0.27 and 0.14 for GARCH-normal. Its average 99% ES rises from 2.71% to 2.89% while its average VaR rises only from 2.36% to 2.45%: the t moves risk from the shoulders into the far tail, where ES lives. It is the first parametric model in the project whose ES is on target at both confidence levels.

**But it does not fix timing.** All five of GARCH-t's 99% breaches fell in 2022, and at 97.5% it fails the independence test (p = 0.027) where GARCH-normal narrowly passed (0.054) and FHS-GARCH passes comfortably (0.41). The tail shape and the volatility dynamics are separate problems. The t improves the first and does nothing for the second, which still depends on how quickly the variance estimate reacts. The two-sided results are also within noise of each other: at these breach counts, one breach moved between years changes a p-value substantially.

**ν is poorly identified on a year of data.** On 250-day windows the estimate ranges from about 12 to 47 (10th to 90th percentile), and on 23% of days it exceeds 30, which is practically normal. A year of daily data contains only a handful of tail events, which is all ν has to go on. This repeats Experiment 11's lesson about GARCH's own parameters: more parameters on a short window means noisier estimates.

**Filtered simulation gets fat tails for free.** FHS-GARCH, which takes the tail shape from the empirical standardized residuals instead of a fitted t, matches GARCH-t on ES at 97.5% (Z2 0.02 vs -0.03), is close at 99% (0.05 vs 0.03), and passes the independence test at both levels. It needs no extra parameter, which is a practical argument for filtered simulation whenever enough history is available.

**Limitations.** One portfolio, one two-year period, 1-day horizon. Symmetric t only (no skew, although skew was about zero here). The likelihood-ratio test is approximate because ν = ∞ sits on the boundary of the parameter space. Breach counts of 5 to 14 cannot separate methods whose Z2 values differ by a few hundredths.

**Conclusion**  
Fat-tailed shocks are real in this portfolio but modest (ν around 15), and most of the apparent tail fatness comes from volatility clustering. Modelling the remaining fat tail with a Student-t fixes the ES understatement of GARCH-normal at both confidence levels, but not the timing of breaches. Across Experiments 10-12, the filtered-simulation family (FHS and FHS-GARCH) remains the most robust overall: it is calibrated on VaR, ES and independence without having to estimate a tail parameter from too little data. GARCH-t is the best choice when a parametric model is required, for example for closed-form ES, multi-day scaling, or speed.

**File Paths**  
/cpp/experiments/garch-t/garch_t_backtest.cpp and csv with results (method_comparison, tail_diagnostics, daily_nu) in /cpp/experiments/garch-t/results/
