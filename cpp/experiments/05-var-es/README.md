**Experiment 7: Historical vs Parametric vs Monte Carlo VaR/ES on a Real Equal-Weight Portfolio**

**Objective**  
Compare the three VaR/ES methods built in Phase 7 on real data, see where and why they disagree, and check how well each method's 1-day VaR matches the loss frequency the portfolio actually experienced. Secondary goal: decompose portfolio risk into per-stock contributions.

**Methodology**  
The full Phase 7 pipeline (`qre::risk::run_risk_pipeline`: forward fill, simple returns, sample covariance, portfolio, simulation, VaR/ES) was run on the Phase 3 price history: AAPL, CAT, JNJ, JPM, PG, XOM, adjusted close, 2021-01-04 to 2023-12-29 (753 prices, 752 daily returns, no missing values), with equal weights of 1/6 each. Horizon is one day. Historical VaR uses the linearly interpolated empirical loss quantile and ES is the mean loss at or beyond it. Parametric uses a zero-mean normal with the sample portfolio volatility. Monte Carlo simulates one day of correlated GBM (Phase 6, 1,000,000 paths, antithetic) from the same sample covariance with zero expected return. Calibration check: for each method, the fraction of the 752 actual portfolio days with a loss strictly greater than that method's VaR, compared with the nominal 1 - c.

C++ only: the quantile, ES and risk-contribution maths are cross-validated against numpy and scipy in the test suite (agreement to about 1e-12), and this experiment is about how the methods behave on real data, not a new mathematical fact. The calibration check is in-sample and is not a backtest; a proper backtesting module (rolling out-of-sample windows, Kupiec-style tests) is a separate later piece.

**Results**

Portfolio return distribution (daily): mean +0.069%, volatility 0.991%, skew -0.02, excess kurtosis 1.08, worst day -3.43%.

| Confidence | Method | VaR | ES | ES / VaR | Days with loss > VaR | Expected days |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 95% | Historical | 1.609% | 2.172% | 1.35 | 38 | 37.6 |
| 95% | Parametric | 1.631% | 2.045% | 1.25 | 37 | 37.6 |
| 95% | Monte Carlo | 1.624% | 2.031% | 1.25 | 37 | 37.6 |
| 97.5% | Historical | 2.062% | 2.516% | 1.22 | 19 | 18.8 |
| 97.5% | Parametric | 1.943% | 2.318% | 1.19 | 25 | 18.8 |
| 97.5% | Monte Carlo | 1.933% | 2.296% | 1.19 | 25 | 18.8 |
| 99% | Historical | 2.525% | 2.891% | 1.15 | 8 | 7.5 |
| 99% | Parametric | 2.306% | 2.642% | 1.15 | 12 | 7.5 |
| 99% | Monte Carlo | 2.281% | 2.613% | 1.15 | 12 | 7.5 |

Risk contributions at the portfolio level (share of the 0.991% daily volatility):

| Stock | Weight | Share of portfolio vol | Standalone daily vol |
| :---- | :---- | :---- | :---- |
| CAT | 16.7% | 23.3% | 1.85% |
| XOM | 16.7% | 21.2% | 1.90% |
| JPM | 16.7% | 19.6% | 1.53% |
| AAPL | 16.7% | 18.5% | 1.75% |
| PG | 16.7% | 9.0% | 1.10% |
| JNJ | 16.7% | 8.4% | 1.02% |

**Discussion**  
**The methods agree at 95% and diverge in the tail.** At 95% all three VaRs are within 1.4% of each other. At 99% the parametric VaR is 8.7% below historical (2.31% vs 2.52%) and ES is 8.6% below (2.64% vs 2.89%). The portfolio has statistically clear fat tails: excess kurtosis of 1.08 against a standard error of about 0.18 (roughly six standard errors from normal). A normal distribution fitted to the volatility underestimates losses further out than the 95th percentile. The positive sample mean makes this stronger: it lowers historical VaR but not the zero-mean parametric figure, yet historical is still the larger.

**Monte Carlo does not fix a wrong model.** The Monte Carlo and parametric numbers differ by under 1.2% at every level and share the same tail understatement. Its simulated shocks are Gaussian, so it converges to the parametric answer; its value is flexibility (path-dependent payoffs, nonlinear positions, other distributions), not a better answer for a linear equity portfolio under normal shocks. The small gap that remains is probably GBM's slightly skewed lognormal returns, though this experiment did not isolate it.

**The calibration evidence is suggestive but not conclusive.** Parametric VaR was exceeded on 25 days at 97.5% (expected 18.8) and 12 days at 99% (expected 7.5). Those are 1.45 and 1.64 binomial standard deviations above expectation, short of the usual 1.96 threshold, and with 752 days the 99% level has under 8 expected exceedances, so one or two days swing the rate. The kurtosis result is the stronger evidence; the exceedance counts are consistent with it but do not prove it alone. Historical VaR matches its nominal rate by construction, since it is estimated from the same days.

**ES / VaR is a sensitivity measure for tail shape.** At 95% historical ES / VaR is 1.35 against 1.25 for the normal models: the actual tail beyond the 95th percentile is heavier than the normal assumes. At 99% the ratios converge (1.15), because with only about 8 tail days the historical ES is dominated by a few observations.

**Diversification is uneven.** JNJ and PG, together a third of the capital, contribute 17.4% of the risk, while CAT alone contributes 23.3% on 16.7% of the capital. Risk contributions sum exactly to the portfolio volatility, so they can be used for budgeting; weight alone says little about where risk comes from.

**Limitations.** One portfolio, one three-year window that includes the 2022 drawdown, equal weights only, in-sample calibration, 1-day horizon only (longer horizons use square-root-of-time scaling for historical and parametric, an approximation). Zero-mean treatment for parametric and Monte Carlo is a convention, not a finding.

**Conclusion**  
The pipeline runs end to end on real data and the three methods behave as theory predicts: they agree where the normal assumption holds and separate in the tail. For this portfolio, normal-based VaR at 97.5% and above likely understates risk by 6% to 9%, and on this evidence historical simulation is the better default when enough history exists. Backtesting, which would turn the exceedance observations into a formal test, is the natural next piece.

**File Paths**  
/cpp/experiments/05-var-es/var_es_methods.cpp and csv with results in /cpp/experiments/05-var-es/results/
