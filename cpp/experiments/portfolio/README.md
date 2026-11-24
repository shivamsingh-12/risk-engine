**Experiment 26 (Phase 21): Portfolio Construction, Component Risk and 10-Day VaR**

**Objective**  
The engine so far measures the risk of weights someone else chose. This phase adds three tools:
- **Portfolio construction:** weights chosen to minimize risk, balance it, or trade it off against expected return, with or without short positions.
- **Component VaR and ES:** each holding's part of the tail risk.
- **Multi-day VaR:** risk over 10 days, the horizon of the Basel market-risk rules.

The experiment asks one question of each:
1. Which construction method actually delivers what it promises out of sample, and at what cost?
2. Does today's component ES tell you where next year's tail losses will come from?
3. Which way of getting a 10-day VaR from daily data holds up in a backtest?

**Methodology**  
Daily adjusted closes, January 2005 to 13 November 2026 (5,502 trading days; `tools/fetch_data.py`). Every estimate uses only data available on the day it is made.

*Part 1, construction.* Three universes:
- **Sector ETFs:** the 9 US sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY), all highly correlated equities.
- **Asset-class ETFs:** 9 ETFs: US large and small caps, developed and emerging markets, long and intermediate Treasuries, investment-grade credit, gold and real estate (SPY, IWM, EFA, EEM, TLT, IEF, LQD, GLD, VNQ).
- **Stocks:** 14 large US stocks with history since 2005 (AAPL, AMZN, BAC, CAT, GOOGL, GS, JNJ, JPM, KO, MSFT, NVDA, PG, WMT, XOM).

Every 21 trading days from 28 December 2006, each method chooses weights from the previous 500 days of returns. The portfolio is then held, with weights drifting with prices, until the next rebalance. That gives 238 rebalances and 19.9 years out of sample. The eight methods:

| Method | Weights | Covariance | Bounds per asset |
| :---- | :---- | :---- | :---- |
| Equal | 1/N | | |
| Inverse volatility | ∝ 1/σᵢ | sample | |
| Min variance (sample) | minimize w'Σw | sample | 0 to 100% |
| Min variance (Ledoit-Wolf) | minimize w'Σw | Ledoit-Wolf | 0 to 100% |
| Min variance with shorts | minimize w'Σw | Ledoit-Wolf | −30% to 100% |
| Risk parity | equal risk contributions | Ledoit-Wolf | long only |
| Mean-variance | maximize μ'w − (γ/2) w'Σw, γ = 5 | Ledoit-Wolf, μ = 500-day mean | 0 to 100% |
| Mean-variance with shorts | as above | as above | −30% to 100% |

All weights sum to 100%. The optimizations are solved by projected gradient descent with Nesterov acceleration (`cpp/include/qre/portfolio/construction.hpp`). The tests check the solver against scipy's SLSQP and the closed-form minimum-variance portfolio, and check that risk parity equalizes contributions to 10⁻¹¹.

Reported for each method:
- annualized return, volatility and their ratio (no risk-free rate is subtracted)
- maximum drawdown
- one-way turnover per rebalance, and the return net of a 10 basis point cost per unit of turnover
- average number of holdings above 1%, and average gross exposure Σ|wᵢ|
- a calibration ratio: the square root of realized over predicted variance in the month after each rebalance. 1 means the risk the optimizer saw was the risk it got.

*Part 2, component risk.* Four fixed-weight portfolios (daily rebalanced):
- 60/40 (SPY/IEF)
- multi-asset: SPY 30%, EFA 15%, EEM 5%, IEF 25%, LQD 10%, GLD 10%, VNQ 5%
- the six bundled stocks, equal weight
- corporate credit (HYG/LQD)

Every 21 days, each holding's share of the 97.5% ES is predicted in two ways from the previous 250 days:
- **parametric:** normal Euler allocation on the sample covariance
- **historical:** each holding's average loss on the days in the ES tail

Each prediction is compared with the holding's actual share of the ES over the next 250 days. So is the naive predictor, the holding's capital weight. Score: mean absolute error in share points.

*Part 3, 10-day VaR.* Seven fixed-weight portfolios: the four above plus SPY, QQQ and XLF. Non-overlapping 10-day periods from 2007 give 500 per portfolio, and 443 for credit, which starts later. Five forecasts of the 99% and 97.5% 10-day VaR are made at the start of each period:

| Method | How |
| :---- | :---- |
| √10 × historical | 1-day historical VaR on the last 250 days, × √10 |
| √10 × FHS (EWMA) | 1-day filtered historical simulation with EWMA volatility (λ = 0.94), × √10 |
| Overlapping | historical VaR of the 491 overlapping 10-day returns in the last 500 days |
| GARCH normal | normal VaR with the GARCH(1,1) 10-day variance: h·v + (σ²₁ − v)(1 − pʰ)/(1 − p) |
| FHS-GARCH | 5,000 simulated 10-day paths: bootstrapped standardized residuals through the GARCH recursion |

The GARCH models are fitted on the last 500 days. A breach is a period whose compounded 10-day loss exceeds the forecast, and the Kupiec test checks the breach rate.

C++ and Python (`python_reference/qre/portfolio/construction.py`, `qre/risk/component.py`, `qre/risk/horizon.py`) match to rounding in both test suites, on shared fixed inputs including the bootstrap draws.

**Results**

Part 1: out of sample, December 2006 to November 2026:

| Universe | Method | Return | Volatility | Return / vol | Max drawdown | Turnover per rebalance | Net of 10 bp | Gross exposure | Realized / predicted vol |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Sector ETFs | Equal | 10.4% | 18.7% | 0.56 | 53.5% | 1.3% | 10.4% | 1.00 | 1.00 |
| | Inverse volatility | 10.2% | 17.6% | 0.58 | 50.3% | 1.3% | 10.2% | 1.00 | 1.01 |
| | Min variance (sample) | 9.0% | 14.4% | 0.63 | 35.3% | 3.7% | 9.0% | 1.00 | 1.05 |
| | Min variance (Ledoit-Wolf) | 9.2% | 14.4% | 0.64 | 35.6% | 3.6% | 9.2% | 1.00 | 1.05 |
| | Min variance with shorts | 9.2% | 13.9% | 0.66 | 35.7% | 8.1% | 9.0% | 1.67 | 1.08 |
| | Risk parity | 10.2% | 17.5% | 0.59 | 50.1% | 1.4% | 10.2% | 1.00 | 1.02 |
| | Mean-variance | 7.5% | 19.4% | 0.39 | 46.2% | 22.9% | 7.2% | 1.00 | 1.05 |
| | Mean-variance with shorts | 5.2% | 25.1% | 0.21 | 44.4% | 55.9% | 4.5% | 3.61 | 1.14 |
| Asset-class ETFs | Equal | 6.7% | 12.6% | 0.53 | 35.2% | 1.4% | 6.7% | 1.00 | 0.99 |
| | Inverse volatility | 5.8% | 8.6% | 0.67 | 23.5% | 1.4% | 5.7% | 1.00 | 1.02 |
| | Min variance (sample) | 4.3% | 5.8% | 0.74 | 20.2% | 2.5% | 4.2% | 1.00 | 1.06 |
| | Min variance (Ledoit-Wolf) | 4.6% | 5.9% | 0.78 | 19.6% | 2.4% | 4.5% | 1.00 | 1.00 |
| | Min variance with shorts | 4.3% | 5.0% | 0.86 | 15.4% | 4.2% | 4.3% | 1.70 | 0.97 |
| | Risk parity | 5.7% | 7.9% | 0.72 | 23.7% | 1.4% | 5.7% | 1.00 | 1.03 |
| | Mean-variance | 6.4% | 15.0% | 0.42 | 31.5% | 20.2% | 6.1% | 1.00 | 1.13 |
| | Mean-variance with shorts | 11.5% | 22.9% | 0.50 | 36.9% | 44.0% | 10.9% | 3.73 | 1.15 |
| Stocks | Equal | 18.4% | 20.7% | 0.89 | 51.8% | 2.5% | 18.4% | 1.00 | 0.99 |
| | Inverse volatility | 16.2% | 18.2% | 0.89 | 44.6% | 2.4% | 16.2% | 1.00 | 1.01 |
| | Min variance (sample) | 10.0% | 14.6% | 0.69 | 34.7% | 4.5% | 10.0% | 1.00 | 1.05 |
| | Min variance (Ledoit-Wolf) | 10.2% | 14.6% | 0.70 | 34.5% | 4.2% | 10.1% | 1.00 | 1.05 |
| | Min variance with shorts | 9.7% | 14.6% | 0.67 | 29.7% | 7.2% | 9.6% | 1.39 | 1.10 |
| | Risk parity | 16.4% | 18.0% | 0.91 | 44.4% | 2.4% | 16.3% | 1.00 | 1.02 |
| | Mean-variance | 19.8% | 28.4% | 0.70 | 58.2% | 18.5% | 19.5% | 1.00 | 1.05 |
| | Mean-variance with shorts | 10.0% | 44.2% | 0.23 | 71.6% | 78.0% | 8.9% | 4.59 | 1.16 |

Calendar-year returns in the three worst equity years:

| Universe | Year | Equal | Min variance (LW) | Min variance with shorts | Risk parity | Mean-variance | Mean-variance with shorts |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Stocks | 2008 | −36.7% | −15.2% | −10.6% | −28.4% | −47.8% | −57.8% |
| Stocks | 2022 | −10.5% | +1.8% | +7.4% | −6.5% | −17.4% | −33.4% |
| Asset-class ETFs | 2008 | −17.6% | +1.0% | −0.4% | −1.0% | −10.3% | +1.3% |
| Asset-class ETFs | 2022 | −18.2% | −14.0% | −9.5% | −17.2% | −18.9% | −22.0% |

For the asset-class ETFs, mean-variance weights swung across the whole allowed range. SPY ranged from 0 to 100% and averaged 28%; GLD also ranged from 0 to 100%. Minimum variance held 37 to 83% in intermediate Treasuries (IEF) and averaged 62%. Risk parity kept every asset between 3% and 33%.

Part 2: each holding's share of the next 250 days' 97.5% ES (mean absolute error over 227 dates, share points):

| Portfolio | Capital weights | Parametric component ES | Historical component ES |
| :---- | :---- | :---- | :---- |
| 60/40 | 42.4 | 9.7 | 9.5 |
| Multi-asset | 11.0 | 4.5 | 5.1 |
| Six stocks | 5.4 | 4.3 | 4.8 |
| Corporate credit (200 dates) | 12.4 | 12.3 | 12.0 |

Average shares (capital weight / predicted / realized):
- **60/40:** SPY 60 / 101 / 102%; IEF 40 / −1 / −2%.
- **Multi-asset:** SPY 30 / 46 / 48%, EFA 15 / 25 / 25%, EEM 5 / 10 / 10%, IEF 25 / 1 / 0%, LQD 10 / 3 / 3%, GLD 10 / 6 / 6%, VNQ 5 / 8 / 8%.
- **Six stocks:** CAT and JPM 17 / 23 / 22%; JNJ and PG 17 / 10 / 10%.

Part 3: 10-day VaR breach rates (99%: 1% expected; 97.5%: 2.5% expected; Kupiec p-value in brackets):

| Portfolio | Level | √10 × historical | √10 × FHS (EWMA) | Overlapping | GARCH normal | FHS-GARCH |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 60/40 | 99% | 1.4% (0.40) | 1.2% (0.66) | 2.6% (0.00) | 2.0% (0.05) | 2.2% (0.02) |
| Multi-asset | 99% | 1.6% (0.21) | 1.4% (0.40) | 2.4% (0.01) | 2.0% (0.05) | 2.0% (0.05) |
| Six stocks | 99% | 1.4% (0.40) | 1.0% (1.00) | 2.6% (0.00) | 1.8% (0.11) | 2.0% (0.05) |
| S&P 500 | 99% | 1.8% (0.11) | 1.4% (0.40) | 2.6% (0.00) | 2.4% (0.01) | 2.4% (0.01) |
| Nasdaq 100 | 99% | 1.4% (0.40) | 0.8% (0.64) | 2.8% (0.00) | 1.6% (0.21) | 1.6% (0.21) |
| Financials | 99% | 2.0% (0.05) | 1.8% (0.11) | 3.2% (0.00) | 2.2% (0.02) | 2.4% (0.01) |
| Credit | 99% | 1.8% (0.13) | 2.5% (0.01) | 2.0% (0.06) | 2.3% (0.02) | 2.3% (0.02) |
| **All, 3,443 periods** | **99%** | **1.63% (0.00)** | **1.42% (0.02)** | **2.61% (0.00)** | **2.03% (0.00)** | **2.12% (0.00)** |
| **All** | **97.5%** | **3.19% (0.01)** | **2.73% (0.39)** | **4.15% (0.00)** | **2.96% (0.09)** | **3.54% (0.00)** |

Average 99% 10-day VaR across all periods: 8.3% (√10 × historical), 8.3% (√10 × FHS), 8.0% (overlapping), 6.9% (GARCH normal) and 7.1% (FHS-GARCH). On breach periods, the average loss was 1.07 to 1.35 times the forecast ES; it was closest for FHS-GARCH (1.07) and √10 × FHS (1.10).

**Discussion**  
**Minimum variance delivers what it promises; mean-variance does not.** Every minimum-variance portfolio had the lowest volatility in its universe: 14.4% against 18.7% for equal weights in sectors, and 5.9% against 12.6% in the asset-class ETFs. Its losses in crises were far smaller: the 14 stocks lost 15% in 2008 against 37% for equal weights, and gained 2% in 2022 when equal weights lost 10%. Its risk forecast was reliable too: realized volatility was 0 to 6% above predicted with long-only weights. That small excess is the optimizer's curse, since minimizing over an estimated matrix favours its errors. Lower risk did not come with lower risk-adjusted return everywhere. On the sector and asset-class ETFs, the minimum-variance portfolios had the best return-to-volatility ratios (0.63 to 0.66 and 0.74 to 0.86, against 0.56 and 0.53 for equal weights). On the stocks, whose returns over these 20 years were led by high-volatility technology names, equal weights and risk parity earned far more (18% against 10% a year). A mean-variance portfolio had the worst return-to-volatility ratio in every universe. Its weights swung between corners, so turnover was 18 to 23% a month against 2 to 4%. Realized volatility exceeded its forecast by 5 to 13%, and its return-to-volatility ratio was below equal weights in every universe. A 500-day sample mean is mostly noise, and the optimizer bets on it. This repeats the classic finding of DeMiguel, Garlappi and Uppal (2009) on 20 years of new data.

**Short positions amplify whatever the inputs say, good or bad.** Allowing 30% shorts lowered minimum-variance volatility further on the ETFs (5.9% to 5.0% on asset-class ETFs, 14.4% to 13.9% on sectors; unchanged on the stocks). It cut the maximum drawdown on the asset-class ETFs (19.6% to 15.4%) and the stocks (34.5% to 29.7%), at about twice the turnover and a gross exposure of 1.4 to 1.7. In mean-variance, shorts turned estimation error into leverage. Gross exposure averaged 3.6 to 4.6, volatility rose to 23 to 44%, and on the stocks the portfolio lost 58% in 2008 and 33% in 2022. One lucky run (11.5% a year on the asset-class ETFs) does not change that: its return-to-volatility ratio was still below that of the long-only minimum variance.

**Shrinkage helped a little here, because the universes are small.** Ledoit-Wolf and the sample covariance gave nearly the same minimum-variance portfolios: volatilities within 0.12 points, with slightly lower turnover and a slightly better return-to-volatility ratio with shrinkage. With 9 to 14 assets and 500 days the sample matrix is already well conditioned. Experiment 15 showed shrinkage matters as N approaches T.

**Risk parity is a middle road.** It held every asset (3% to 33% in the asset-class ETFs) and kept turnover as low as equal weights. It cut volatility by 6 to 37% against equal weights, and its return-to-volatility ratio was never the worst and beat equal weights in every universe. It does not chase the lowest risk: in the asset-class ETFs it held 8% SPY on average where minimum variance held 13%, but also 14% TLT where minimum variance held almost none.

**Capital weights say little about risk; component risk says much more.** In a 60/40 portfolio, stocks are 60% of the capital and, on average, 102% of the tail risk. The bonds' share of the ES was slightly negative: on the worst days they gained. Predicting each holding's share of next year's tail losses from its capital weight was off by 42 points in a 60/40 and 11 points in the multi-asset portfolio. Component ES from the past year cut those errors to 10 and 5 points. The parametric (normal) and historical versions did about equally well, so the normal model's tail shape matters less for the split than for the level. Component ES failed for corporate credit: high yield and investment grade swap places in the tail depending on whether a crisis is about defaults or rates, and the previous year does not say which comes next.

**For 10-day VaR, extrapolating today's volatility beat modelling its path.** The method with no model of the path, √10 × FHS with EWMA volatility, was the best calibrated: 1.42% breaches at 99% and 2.73% at 97.5%, the second not rejected by Kupiec. Overlapping 10-day historical returns were the worst (2.61%): 491 overlapping returns contain only about 50 independent 10-day periods, so the tail quantile rests on a handful of episodes. The GARCH methods breached about twice as often as they should (2.0 to 2.1%) and gave the lowest VaR on average (6.9 to 7.1% against 8.3%). Their extra breaches came in years when markets turned after a calm stretch: 2007 (9 against 3 for √10 × FHS), 2016 (6 against 0) and 2025 (7 to 9 against 3). In February 2018 every method was caught out. GARCH reverts volatility towards the average of its 500-day window, so after a quiet year it forecasts quiet 10-day periods. EWMA times √10 carries today's volatility forward unchanged and is slower to relax. The GARCH breaches were also less severe: FHS-GARCH losses on breach periods averaged 1.07 times its ES, the closest of any method. None of the five methods reached 1% at 99%. The √10 rule is a rule of thumb, and the backtests suggest that, applied to an adaptive 1-day model, it is about as good as anything more elaborate here.

**Limitations.**
- **Universes:** small (9 to 14 assets) and US-centred. The stocks are survivors that have traded since 2005, which flatters equal weights and risk parity. Returns ignore dividends' timing within the adjusted closes, taxes, borrowing costs for shorts and margin.
- **Construction choices:** one estimation window (500 days), one rebalancing frequency (21 days) and one risk aversion (γ = 5). Mean-variance would do better with shrunk or model-based expected returns, which this experiment does not try.
- **Component ES:** realized shares rest on about seven tail days per year, so part of every error is noise in the target itself.
- **10-day VaR:** periods within a portfolio are independent, but the seven portfolios share market moves, so the pooled Kupiec test overstates the evidence.

**Conclusion**  
- **Construction:** minimum variance delivered the risk reduction it promised, in every universe and every crisis. Mean-variance with historical means bet on noise, and shorts magnified the bet. Risk parity sat between equal weights and minimum variance, with the low turnover of the first.
- **Component risk:** a portfolio's tail risk can be very different from its capital split; a 60/40 is all stock risk. Component ES from the past year tells you which holdings carry it, with a typical error of 4 to 10 share points, except for credit.
- **10-day VaR:** √10 times a 1-day model that adapts to current volatility was the best-calibrated forecast. Multi-day GARCH forecasts relaxed too quickly after calm periods.

The app now suggests weights by these methods, shows component VaR and ES, accepts short positions and reports 10-day risk.

**File Paths**  
/cpp/experiments/portfolio/portfolio_experiment.cpp and csv with results (construction, construction_weights, construction_by_year, component_prediction, component_summary, component_latest, horizon_forecasts, horizon_summary) in /cpp/experiments/portfolio/results/
