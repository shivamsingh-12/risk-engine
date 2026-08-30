**Experiment 3: Volatility Estimate Stability Across Lookback Windows**

**Objective:**  
Empirically characterize how rolling-window length affects the level of estimated volatility and the stability (noisiness) of that estimate over time, using real daily equity price data.

**Methodology:**  
Daily adjusted close prices for six large-cap, cross-sector equities (AAPL, CAT, JNJ, JPM, PG, XOM; Yahoo Finance via yfinance, 2021-01-04 to 2023-12-29, \~753 trading days) were converted to log returns, then rolling volatility was computed at four window lengths — 10, 20, 60, and 90 trading days. For each (window, ticker) pair, two statistics were computed: mean\_vol (average estimated volatility) and stddev\_of\_vol (standard deviation of the rolling-volatility series itself — the noisiness metric).

Implemented in Python only, departing from the C++/Python cross-validation used in Experiments 1 and 2\. Cross-validation earns its cost where a subtle bug in closed-form math or a numerical routine could hide silently across implementations — not here, since this experiment is data orchestration built on Phase 1 estimators already cross-validated in both languages. Reimplementing it in C++ would duplicate proven code without adding correctness confidence, at the cost of hand-rolling CSV parsing Python already handles reliably.

Unlike Experiments 1 and 2's single-draw designs, rolling windows over a fixed dataset are not independent trials — consecutive windows share nearly all their observations. A smoother, more monotonic result than Experiment 1's single-trial noise was expected.

**Results:**

| Window | Ticker | Mean Vol | Std Dev of Vol   |
| :---- | :---- | :---- | :---- |
| 10 | AAPL | 0.016262 | 0.006701 |
| 20 | AAPL | 0.016563 | 0.005816 |
| 60 | AAPL | 0.016944 | 0.004880 |
| 90 | AAPL | 0.017132 | 0.004506 |
| 10 | JNJ | 0.009702 | 0.003360 |
| 20 | JNJ | 0.009928 | 0.002570 |
| 60 | JNJ | 0.010062 | 0.001598 |
| 90 | JNJ | 0.010130 | 0.001374 |
| 10 | XOM | 0.018190 | 0.005775 |
| 20 | XOM | 0.018493 | 0.004481 |
| 60 | XOM | 0.018763 | 0.003338 |
| 90 | XOM | 0.018810 | 0.003134 |

Cross-sectional volatility ranking, stable at every window: XOM \> CAT \> AAPL \> JPM \> PG \> JNJ.

**Discussion:**

**Noisiness decreases monotonically with window length, across all six tickers with zero reversals.** AAPL's stddev\_of\_vol falls \~33% (0.00670 → 0.00451) from a 10- to 90-day window; JNJ falls \~59%. This is expected, not incidental: consecutive rolling windows share nearly all their observations, mechanically smoothing the estimate — unlike Experiments 1 and 2's fully independent draws.

**Mean volatility drifts modestly upward with window length, consistently across tickers (AAPL: 0.01626 → 0.01713).** This isn't necessarily evidence longer windows are "more correct" — with only one historical sample period, this can't be cleanly separated from a genuine volatility regime shift within 2021-2023 that longer windows partially average over.

**Cross-sectional ranking is invariant to window choice**, and sector-intuitive (XOM/CAT highest, JNJ/PG lowest) — an informal sanity check that the pipeline produces sensible output on real data.

**Conclusion:**

Volatility estimation shows a clear bias-variance tradeoff by lookback window: shorter windows are noisier but more responsive; longer windows are smoother but slower to react and blend across regimes. No window length is unconditionally correct. Cross-sectional ranking is robust to this choice; absolute magnitude is not.

**File Paths:**

 /python\_reference/experiments/03-statistics for vol\_stab.py and csv with results
