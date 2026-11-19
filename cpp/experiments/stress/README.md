**Experiment 25 (Phase 20): Stress Testing — What Past Crises Would Do to a Portfolio Today, and What VaR Misses**

**Objective**  
VaR describes ordinary bad days: the loss exceeded on 1 day in 100, estimated from the last year. Regulators and risk managers also ask a different question: what would this portfolio lose if a particular crisis happened again? This phase adds three stress-testing tools to the engine:
- **Historical replay** of ten crises from 2007 to 2025.
- **Hypothetical shocks** ("SPY falls 20%") by Gaussian conditioning on the covariance matrix.
- **Reverse stress tests:** the market fall that would cause a given loss.

The experiment uses 21 years of daily prices to ask four questions:
1. What do the crises do to seven different portfolios?
2. Would the engine's rolling VaR models have covered the crisis days?
3. How well does a correlation-based shock predict what each asset did in a crisis?
4. How wrong is the stand-in used for an asset that did not trade during a scenario?

**Methodology**  
Daily adjusted closes for 50 tickers, January 2005 to November 2026, from `tools/fetch_data.py`.

| Scenario | Window (close to close) | Trading days |
| :---- | :---- | :---- |
| Global financial crisis | 9 Oct 2007 to 9 Mar 2009 | 355 |
| Lehman collapse (inside the GFC) | 12 Sep to 20 Nov 2008 | 49 |
| US downgrade and euro crisis | 22 Jul to 3 Oct 2011 | 50 |
| China devaluation and oil crash | 17 Aug 2015 to 11 Feb 2016 | 123 |
| February 2018 volatility spike | 26 Jan to 8 Feb 2018 | 9 |
| Q4 2018 sell-off | 20 Sep to 24 Dec 2018 | 65 |
| COVID crash | 19 Feb to 23 Mar 2020 | 23 |
| 2022 inflation and rate shock | 3 Jan to 12 Oct 2022 | 195 |
| 2023 regional bank failures | 8 to 17 Mar 2023 | 7 |
| April 2025 tariff shock | 2 to 8 Apr 2025 | 4 |

Each window runs from the S&P 500's peak to its trough, except for the bank failures, which were a sector event.

The seven portfolios:
- S&P 500 (SPY)
- Nasdaq 100 (QQQ)
- 60/40 (SPY/IEF)
- the six bundled stocks, equal weight
- multi-asset: SPY 30%, EFA 15%, EEM 5%, IEF 25%, LQD 10%, GLD 10%, VNQ 5%
- corporate credit (HYG/LQD)
- financials (XLF)

*Part 1, replay.* Each asset's actual daily returns over the window, applied to the portfolio held buy-and-hold (fixed numbers of shares from the start). Reported: total return, maximum drawdown, worst day and worst 10 days.

*Part 2, VaR inside the crises.* On every crisis day, six of the engine's models forecast 99% one-day VaR from the previous 250 days, exactly as in the daily backtest (portfolio rebalanced daily). The models are historical, parametric normal, EWMA, filtered historical simulation (FHS), GARCH-t and conditional EVT.
- A breach is a day whose loss exceeds that day's forecast. A correct model breaches on 1% of days.
- The 10-day VaR from the √10 rule, measured on the scenario's first day, is compared with the scenario's worst 10 days.
- Lehman is reported, but left out of the totals because it lies inside the GFC.
- The credit portfolio is left out of the GFC because HYG had traded for less than 250 days before it began.

*Part 3, conditional shocks.* For each scenario, SPY's actual return over the window is the shock. Every other asset's expected return given that shock is β × shock, where β = Cov(asset, SPY) / Var(SPY) comes from one of two covariance matrices:
- **calm:** the 500 trading days before the scenario
- **crisis:** the daily returns of every other non-overlapping scenario, pooled

The predictions are compared with what 25 to 27 assets actually did. The assets are 15 ETFs (bonds, credit, gold, oil, real estate, regions) covering the asset classes, the nine sector ETFs, and the six stocks.

*Part 4, proxies.* A holding that did not trade during a scenario (a recent listing) is replaced by β × SPY's daily return. To test this, each of the 27 assets that did trade is replaced by its proxy, with β estimated from the three years after the scenario. That is the only data a new listing would have. The proxy's scenario return is then compared with the actual one, for single assets and for whole portfolios.

The C++ module (`cpp/include/qre/risk/stress.hpp`) and the Python reference (`python_reference/qre/risk/stress.py`) match to rounding in both test suites. The tests also check the formulas by hand (bivariate conditioning, a three-day replay with cash and a short position, reverse stress against the forward shock), and that the COVID replay of SPY falls 34% on the real data.

**Results**

Part 1: buy-and-hold return over each scenario (maximum drawdown in brackets where it differs):

| Portfolio | GFC | 2011 | 2015-16 | Feb 2018 | Q4 2018 | COVID | 2022 | Banks 2023 | Tariffs 2025 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| S&P 500 | −55.2% | −17.9% | −12.2% (12.8) | −10.1% | −19.3% | −33.7% | −24.5% | −1.9% (3.4) | −12.1% |
| Nasdaq 100 | −51.6% (53.4) | −14.0% (16.1) | −12.9% (16.1) | −10.2% | −22.0% (22.7) | −27.9% (28.6) | −34.3% | +2.5% (3.1) | −12.6% |
| 60/40 | −25.1% | −7.0% (8.1) | −5.1% (6.3) | −6.5% | −10.3% | −17.7% | −20.9% | +0.7% (1.0) | −7.4% |
| Six stocks | −46.0% (47.4) | −15.7% | −7.7% (12.8) | −10.9% | −18.5% (18.6) | −33.5% | −5.9% (15.2) | −4.4% | −12.9% |
| Multi-asset | −25.3% (27.6) | −8.5% | −5.2% (7.0) | −6.2% | −8.0% | −18.6% (19.1) | −21.3% (21.4) | +0.7% (0.6) | −7.4% |
| Corporate credit | −19.8% (25.5) | −4.1% (4.5) | −4.8% (5.8) | −2.4% | −3.5% (3.8) | −17.1% (20.0) | −17.8% (18.3) | +0.9% (0.3) | −3.6% |
| Financials | −81.6% (81.8) | −26.1% | −21.4% | −10.0% | −22.2% | −42.8% | −22.3% (25.8) | −11.4% | −12.4% |

Part 2: rolling 99% one-day VaR on the crisis days (7 portfolios × 9 non-overlapping scenarios, 5,462 portfolio-days, 54.6 breaches expected):

| Model | Breaches | Breach rate | Median worst loss / VaR that day | Largest | Loss / ES on breach days |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 317 | 5.8% | 1.99 | 3.92 | 1.19 |
| Parametric (normal) | 465 | 8.5% | 2.25 | 4.81 | 1.32 |
| EWMA | 232 | 4.2% | 1.45 | 3.24 | 1.20 |
| Filtered historical | 138 | 2.5% | 1.15 | 2.61 | 1.07 |
| GARCH-t | 200 | 3.7% | 1.26 | 2.54 | 1.08 |
| Conditional EVT | 163 | 3.0% | 1.15 | 2.34 | 1.14 |

Breach rate by scenario (all portfolios):

| Model | GFC | 2011 | 2015-16 | Feb 2018 | Q4 2018 | COVID | 2022 | Banks 2023 | Tariffs 2025 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Historical | 4.2% | 9.4% | 3.3% | 36.5% | 2.6% | 35.4% | 4.4% | 4.1% | 46.4% |
| Filtered historical | 1.5% | 4.6% | 2.1% | 22.2% | 2.2% | 11.8% | 1.2% | 4.1% | 42.9% |
| Conditional EVT | 1.9% | 6.0% | 2.8% | 27.0% | 2.4% | 8.7% | 1.5% | 2.0% | 46.4% |

The S&P 500's 10-day 99% VaR on each scenario's first day (historical, √10 rule) against what followed:

| Scenario | 1-day VaR | 10-day VaR | Worst 10 days | Whole scenario |
| :---- | :---- | :---- | :---- | :---- |
| GFC | 2.5% | 7.8% | 26.8% | 55.2% |
| 2011 | 1.9% | 6.1% | 16.1% | 17.9% |
| COVID | 2.5% | 8.0% | 23.3% | 33.7% |
| 2022 | 2.2% | 6.9% | 12.2% | 24.5% |
| Tariffs 2025 | 2.5% | 7.8% | 12.1% | 12.1% |

Across all 62 portfolio-scenario pairs, the worst 10 days exceeded the 10-day VaR in 54 (historical) and 56 (GARCH-t). The median worst 10 days was 1.7 times the historical 10-day VaR, and the median whole-scenario loss 2.3 times.

Part 3: predicting each asset's scenario return from SPY's, error across the assets (percentage points of return):

| Scenario | SPY | RMSE, calm covariance | RMSE, crisis covariance | Wrong sign, calm | Wrong sign, crisis |
| :---- | :---- | :---- | :---- | :---- | :---- |
| GFC | −55.2% | 17.6 | 11.1 | 1 of 25 | 0 of 25 |
| 2011 | −17.9% | 6.6 | 8.6 | 1 of 27 | 1 of 27 |
| 2015-16 | −12.2% | 10.0 | 9.7 | 4 of 27 | 5 of 27 |
| Feb 2018 | −10.1% | 3.9 | 3.0 | 5 of 27 | 4 of 27 |
| Q4 2018 | −19.3% | 7.2 | 8.3 | 1 of 27 | 1 of 27 |
| COVID | −33.7% | 13.7 | 10.2 | 1 of 27 | 1 of 27 |
| 2022 | −24.5% | 26.8 | 26.0 | 6 of 27 | 7 of 27 |
| Banks 2023 | −1.9% | 5.8 | 5.5 | 10 of 27 | 6 of 27 |
| Tariffs 2025 | −12.1% | 5.7 | 4.1 | 2 of 27 | 3 of 27 |
| All (excluding Lehman) | | 12.8 | 11.5 | | |

Selected assets, actual against predicted (calm / crisis covariance):

| Scenario | Asset | Actual | Predicted, calm | Predicted, crisis |
| :---- | :---- | :---- | :---- | :---- |
| GFC | Gold (GLD) | +23.9% | −21.7% | +2.9% |
| GFC | Long Treasuries (TLT) | +25.1% | +4.0% | +15.3% |
| COVID | Utilities (XLU) | −35.4% | −6.7% | −23.2% |
| COVID | High yield (HYG) | −21.9% | −8.9% | −13.5% |
| COVID | Oil (USO) | −56.4% | −25.3% | −20.1% |
| 2022 | Long Treasuries (TLT) | −29.3% | +7.1% | +6.2% |
| 2022 | Energy (XLE) | +44.5% | −33.1% | −31.4% |
| Banks 2023 | Financials (XLF) | −11.4% | −1.7% | −2.5% |

Part 4: assets replaced by β × SPY (β from the three years after the scenario):

| Scenario | Mean absolute error | RMSE | Mean error |
| :---- | :---- | :---- | :---- |
| GFC | 6.7 | 9.5 | −3.9 |
| COVID | 8.9 | 12.0 | +4.6 |
| 2022 | 13.0 | 22.0 | −2.0 |
| Feb 2018 | 2.0 | 2.7 | +0.6 |
| All (excluding Lehman) | 6.3 | 10.6 | |

The largest single errors were commodities in 2022. XOM rose 59.8% while its proxy fell 12.7%, and XLE rose 44.5% against a proxy of −17.9%. With every holding proxied, a portfolio's scenario return was off by 2.0 points on average for the 60/40, 2.3 for the multi-asset portfolio, 3.5 for credit, 3.8 for the Nasdaq 100, 4.4 for the six stocks and 4.9 for financials.

**Discussion**  
**Crises are far beyond what VaR describes, and they differ from each other.** The S&P 500 lost 55% in the GFC, 34% in five weeks of COVID and 25% in 2022. Its 99% one-day VaR on the eve of each was 2 to 2.5%. VaR is not meant to cover these losses, but the size of the gap is the point: the √10 rule turned a 2.5% one-day VaR into a 10-day VaR of 8%, and the worst 10 days of the GFC and COVID lost 27% and 23%. Across all portfolios and scenarios the worst 10 days beat the 10-day VaR in 54 of 62 cases. The scenarios also hurt different portfolios. The six stocks lost only 6% over 2022, when energy (XOM +60%) offset everything else, but 46% in the GFC. The 60/40 portfolio roughly halved the S&P 500's loss in the GFC, 2011, 2015-16, Q4 2018 and COVID, when Treasuries rallied. In February 2018, April 2025 and above all 2022, bonds fell too, and in 2022 it lost 21% against 25%. A single VaR number cannot show that a portfolio's protection depends on which crisis comes. A table of scenarios can.

**Models that adapt to volatility cover crisis days far better than ones that do not, but none reaches 1%.** On the 5,462 crisis days, the historical model's VaR was breached on 5.8% of days and the normal parametric model's on 8.5%. Both average the past year equally, so they react slowly. The three models that rescale by current volatility did much better: FHS breached on 2.5%, conditional EVT on 3.0% and GARCH-t on 3.7%. Their median worst crisis loss was 1.15 to 1.26 times the VaR forecast for that day, against 2 times for historical VaR. Their ES was also closer: on breach days, losses averaged 1.07 to 1.14 times the forecast ES. The adaptive models fail at the very start of a crisis, before volatility has risen. In the February 2018 spike, the most volatile COVID days and the four days of the April 2025 tariff shock, every model was breached on a large share of days. In the tariff shock that share was 43 to 57%, because the first large loss was also one of the largest. Calm-period data cannot forecast a jump from calm to crisis. This is the gap stress tests are meant to fill.

**Correlations estimated in calm times predict crisis moves only roughly, and they get the sign of bonds and commodities wrong when the crisis is not a classic flight to safety.** Predicting each asset from SPY's move had an RMSE of 12.8 percentage points with calm-period covariance and 11.5 with crisis-period covariance. Crisis covariance helped most where crises behave alike:
- **Gold in the GFC:** calm data said gold falls with stocks (−22% predicted); crisis data predicted +3%. It rose 24%.
- **Utilities in COVID:** calm data predicted a fall of 7%; crisis data predicted 23%. They fell 35%.
- **High yield in COVID:** calm data predicted −9% and crisis data −14%, against an actual −22%.

2022 is the case where neither helps. Every past crisis had Treasuries rallying when stocks fell, so both matrices predicted TLT +6 to +7%. It lost 29%. They also predicted energy would fall 31 to 33% with the market; it rose 45%. The error for 2022 was twice the average (26 to 27 points). A crisis caused by inflation and rising rates has a different correlation structure from one caused by a credit or growth shock, and no history of growth shocks contains it. A hypothetical shock built on any single covariance matrix inherits that blind spot. This is why the app lets you shock several assets at once (bonds and stocks together, say) rather than only the market.

**Proxying a missing asset by its beta keeps the market part and loses the rest.** Replacing an asset by β × SPY was off by 6.3 points on average across all assets and scenarios, but whole portfolios fared better (2 to 5 points), because individual errors partly cancel. The proxy is unbiased on average (mean errors between −4 and +5 points) and fails for assets whose crisis behaviour was their own: commodities in 2022 (XOM, XLE, USO), gold in the GFC, banks in 2023. For a holding without history the app therefore says that it was proxied, for how many days, and with which β.

**Limitations.**
- Ten scenarios chosen by hand, all US-centred. The windows are peak to trough, which makes them as bad as they can be for an S&P 500 holder; other portfolios had their worst points on different dates.
- Replay assumes the portfolio is held unchanged through the crisis, with no trading costs and no liquidity problems. Selling into a crisis costs more (Phase 25 will add liquidity).
- Conditional shocks use a normal model and linear combination of simple returns. They give an expected move and a spread, not a distribution of extreme outcomes, and they cannot represent the change of regime seen in 2022.
- The proxy uses only SPY. A sector or factor proxy (Phase 24) should do better for sector-specific assets.
- Prices are adjusted closes from Yahoo Finance, so dividends are included and intraday moves are not.

**Conclusion**  
Stress tests and VaR answer different questions, and this experiment shows why both are needed:
- The ten crises cost the S&P 500 between 2% and 55%, against one-day 99% VaR forecasts of 1.4 to 3.8% on their eve. The worst 10 days of a crisis exceeded the √10 VaR in 87% of cases.
- Volatility-scaled models (FHS, conditional EVT, GARCH-t) cut crisis-day breaches from 6% to 2.5 to 3.7%. No model handled the first days of a sudden crisis.
- Correlation-based shocks capture the market part of a crisis and miss what is specific to it. Covariance from past crises predicts better than calm-period covariance, but neither foresaw bonds and stocks falling together in 2022.
- A beta proxy for missing history is unbiased but, for commodities and gold, can be badly wrong.

The app now shows all three tools for any portfolio.

**File Paths**  
/cpp/experiments/stress/stress_experiment.cpp and csv with results (replay, var_coverage, var_coverage_summary, var_at_start, conditional_predictions, conditional_summary, proxy_accuracy, proxy_summary, proxy_portfolios) in /cpp/experiments/stress/results/
