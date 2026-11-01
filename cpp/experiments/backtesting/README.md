**Experiment 9: Backtesting VaR/ES Out of Sample, and What 500 Days Can Prove**

**Objective**  
Phase 7 compared the three VaR/ES methods in-sample, where historical VaR matches its own data by construction. This experiment asks the question that matters: when each method forecasts tomorrow's risk using only the past, do the forecasts hold up? It also tests the tests, measuring how often the Kupiec test catches a bad model with the amount of data available.

**Methodology**  
*Part 1, rolling backtest.* Real 6-stock history (AAPL, CAT, JNJ, JPM, PG, XOM, equal weights), loaded through the hardened data layer and required to pass its quality report. Each trading day t from 2021-12-31 to 2023-12-29 (502 days) gets a 1-day forecast from the 250 daily returns ending the previous day (`qre::data::pit_window`, so there is no look-ahead; a test enforces this). The forecast is compared with that day's realized portfolio loss, and a breach means the loss exceeded VaR. The methods are historical, parametric (zero-mean normal) and Monte Carlo (correlated GBM, 20,000 paths). Confidence levels are 99% and 97.5%. Statistics computed:
- Kupiec: is the breach rate right?
- Christoffersen: are breaches independent, or do they cluster?
- Conditional coverage: both of the above together.
- The Basel traffic light.
- The Acerbi-Székely Z2 statistic for ES, in loss convention: zero for a correct model, positive when ES is too small.

*Part 2, size and power.* Breach counts are simulated from a binomial with a known true breach probability (20,000 repetitions per cell), and the Kupiec test at the 5% level is applied against the nominal rate. Size is the rejection rate when the model is correct. Power is the rejection rate when the true breach probability is 1.5x or 2x the nominal one.

C++ only: every test statistic is cross-validated against scipy in both test suites (agreement to 1e-10 or better), and the traffic light reproduces the Basel 250-day table exactly.

**Results**

Full period, 502 forecasts:

| Confidence | Method | Breaches (expected) | Kupiec p | Christoffersen p | Cond. coverage p | Z2 (ES) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 99% | Historical | 7 (5.0) | 0.40 | 0.66 | 0.64 | 0.51 |
| 99% | Parametric | 11 (5.0) | **0.020** | **0.018** | **0.004** | 1.34 |
| 99% | Monte Carlo | 11 (5.0) | **0.020** | **0.018** | **0.004** | 1.36 |
| 97.5% | Historical | 11 (12.6) | 0.65 | **0.018** | 0.056 | -0.03 |
| 97.5% | Parametric | 12 (12.6) | 0.87 | **0.027** | 0.087 | 0.14 |
| 97.5% | Monte Carlo | 12 (12.6) | 0.87 | **0.027** | 0.087 | 0.16 |

Bold: rejected at the 5% level.

By calendar year at 99%:

| Year | Method | Breaches / days | Kupiec p | Traffic light |
| :---- | :---- | :---- | :---- | :---- |
| 2022 | Historical | 7 / 251 | 0.019 | yellow |
| 2022 | Parametric | 11 / 251 | 0.00007 | **red** |
| 2022 | Monte Carlo | 11 / 251 | 0.00007 | **red** |
| 2023 | All three | 0 / 250 | 0.025 | green |

All 99% breaches for every method fell between 2022-04-22 and 2022-09-13. Over the most recent 250 forecasts (all of 2023), every method shows green.

Kupiec rejection rate at the 5% level (size when the true rate equals the nominal one, power otherwise):

| Nominal | Days | Correct model | 1.5x too many breaches | 2x too many breaches |
| :---- | :---- | :---- | :---- | :---- |
| 1% | 250 | 9.7% | 11.0% | 24.0% |
| 1% | 502 | 6.9% | 23.0% | 55.6% |
| 1% | 1,000 | 5.6% | 33.5% | 78.3% |
| 1% | 2,500 | 4.3% | 61.8% | 98.6% |
| 2.5% | 250 | 7.5% | 23.9% | 60.0% |
| 2.5% | 502 | 6.3% | 42.3% | 87.8% |
| 2.5% | 1,000 | 4.2% | 61.9% | 98.5% |
| 2.5% | 2,500 | 4.7% | 94.8% | 100% |

**Discussion**  
**Out of sample, the normal models fail at 99% and historical passes.** Parametric and Monte Carlo VaR were breached 11 times against 5 expected, which Kupiec rejects (p = 0.020). The breaches also cluster (Christoffersen p = 0.018), and the combined test rejects strongly (p = 0.004). Historical simulation's 7 breaches are consistent with 5 (p = 0.40). This confirms, out of sample and with a formal test, the Phase 7 in-sample finding that normal-based VaR understates this portfolio's tail. Monte Carlo again behaves exactly like parametric, because its shocks are Gaussian.

**ES is understated even where VaR passes.** Z2 is 1.34 for the parametric model and 0.51 for historical at 99%. Both are positive, meaning that when VaR was breached, losses averaged more than the forecast ES. Z2 has no p-value here, because its null distribution depends on the model, so treat it as direction and size rather than a verdict. At 97.5%, historical ES is on target (Z2 = -0.03) and parametric is mildly understated (0.14).

**The real problem is clustering, and a 250-day window causes it.** Every 99% breach came in five months of 2022, when volatility rose during the rate-hike selloff. The pattern is consistent with a 250-day window adapting slowly: in early 2022 it still mostly reflected a calmer 2021, so forecasts lagged reality and breaches bunched up. By 2023 the window was full of 2022's volatility (average parametric 99% VaR rose from 2.37% in 2022 to 2.55% in 2023), so forecasts sat above what happened, with zero breaches all year. Kupiec flags that too (p = 0.025): zero breaches in 250 days is itself evidence the model is mis-calibrated, in the conservative direction. At 97.5% the breach counts are right but Christoffersen still rejects independence for every method. The counts average out, but the timing does not.

**Which year you test decides the verdict.** Judged on 2022, the parametric model is in the Basel red zone and historical is yellow. Judged on 2023, the most recent 250 days and the window a regulator would look at, all three are green. The same models get opposite grades a year apart. A single 250-day traffic light is a snapshot, and reporting the whole history plus a clustering test gives a much more honest picture.

**500 days can reject a bad model, but cannot certify a good one.** With 502 days at 99%, Kupiec catches a model with twice the nominal breach rate only 56% of the time, and one with 1.5x only 23%. At 250 days it is worse: 24% power against a 2x model, and 9.7% false rejections of a correct one, because the test is discrete and the chi-square approximation is rough with so few expected breaches. Historical VaR passing (p = 0.40) is therefore weak evidence it is correct. Parametric failing is stronger evidence it is wrong, because a test this weak only rejects when the miss is large. 97.5% is much easier to test with the same data (88% power against 2x at 502 days), because there are more breaches to count.

**Limitations.** One portfolio and one two-year period dominated by a single volatility regime change. A 250-day equal-weight window only (an exponentially weighted or shorter window would adapt faster, which is a natural next experiment). 1-day horizon only. Overlapping windows mean consecutive forecasts share 249 of 250 data points, so forecasts are not independent even when realized breaches are. The size/power study treats breaches as independent Bernoulli draws, which the real breaches are not.

**Conclusion**  
On this data, historical simulation is the only one of the three methods whose 99% VaR survives an out-of-sample backtest, though its positive Z2 suggests its ES is also somewhat low. All three methods share a deeper flaw: a 250-day window reacts too slowly to volatility regimes, which produces clustered breaches in stressed periods and over-conservative forecasts afterwards. The next modelling improvement is volatility that adapts (EWMA or GARCH), and this backtest framework is how its value will be measured.

**File Paths**  
/cpp/experiments/backtesting/var_backtest.cpp and csv with results (backtest_summary, by_year_99, daily_forecasts_99, kupiec_size_power) in /cpp/experiments/backtesting/results/
