**Experiment 20 (Phase 15): The Portfolio Risk App on Six Example Portfolios**

**Objective**  
Phase 15 puts the engine behind an interactive app. A user enters any tickers and weights; the app downloads prices, runs all eight risk models plus their combination, and shows tomorrow's VaR and ES, volatility (realized, EWMA and GARCH), where the risk comes from, correlations, drawdowns and how well each model's past forecasts held up. This experiment runs the app's own analysis on six example portfolios, to check that its numbers tell a coherent story and to see what it teaches about concentration, diversification and model agreement.

**Methodology**  
The six portfolios are built from the bundled stocks (AAPL, CAT, JNJ, JPM, PG, XOM, 2021-2023), so the run is offline and reproducible:
- equal weight across all six
- a single stock (AAPL)
- defensive (PG and JNJ, 50/50)
- cyclical (XOM and CAT, 50/50)
- tech and banks (AAPL and JPM, 50/50)
- minimum variance

The minimum-variance weights are Ledoit-Wolf minimum-variance weights (Experiment 15), estimated on the first 250 days only, with negative weights set to zero. Most of the evaluated period is therefore out of sample for them.

Each portfolio goes through exactly the code path behind the app's Analyze button:
- data preparation and quality checks
- the C++ engine with all eight models and an equal-weight combination at 99% and 97.5%, a 250-day window and 402 evaluated backtest days
- the descriptive analytics: volatility, Euler risk contributions, the diversification ratio Σ wᵢσᵢ / σₚ and drawdowns

Risk contributions and correlations use the last 250 days.

Building the app turned up two engine problems, both fixed in both languages with regression tests:
- **GARCH-t was slow on near-normal returns.** It took 32 seconds on one 3-stock portfolio. Its degrees of freedom ν kept running into the cap of 1,000 added in Phase 13, which returned an infinite likelihood above the cap, and Nelder-Mead crawled along that wall. ν is now clamped there instead, so the likelihood is flat above the cap, and the same run takes 8 seconds. Results that never reached the cap are unchanged byte for byte: the real-data parts of Experiments 12, 16, 18 and 19. The synthetic calm regimes of Experiment 18 do reach it. Their scores moved in the fourth decimal and that README was updated.
- **Conditional EVT stopped the whole run on the minimum-variance portfolio.** In one window, the method-of-moments starting point for the GPD fit placed the largest exceedance outside the distribution's support, and the optimizer refused the infinite start. That start is now skipped; the second start, with ξ = 0.1, is always feasible.

**Results**

| Portfolio | Vol, last year | Vol, 3 years | 99% VaR (combined) | 99% ES | Max drawdown | Return p.a. | Diversification ratio |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Equal weight, 6 stocks | 12.8% | 15.7% | 1.73% | 1.92% | 15.8% | 17.6% | 1.64 |
| Single stock (AAPL) | 20.3% | 27.8% | 2.67% | 3.38% | 30.9% | 14.9% | 1.00 |
| Defensive (PG, JNJ) | 12.8% | 14.6% | 1.91% | 2.15% | 18.0% | 4.1% | 1.22 |
| Cyclical (XOM, CAT) | 22.5% | 26.0% | 3.13% | 3.67% | 23.8% | 31.3% | 1.19 |
| Tech and banks (AAPL, JPM) | 16.5% | 21.6% | 1.84% | 2.19% | 29.6% | 15.6% | 1.25 |
| Minimum variance | 11.1% | 13.8% | 1.58% | 1.80% | 16.6% | 9.6% | 1.60 |

VaR and ES are next-day forecasts for 2024-01-02. Every portfolio's combined forecast passed the Kupiec and Christoffersen tests at both 99% and 97.5%. The smallest p-value was 0.16, for minimum variance at 97.5% (6 breaches against 10 expected).

Share of capital vs share of risk (last 250 days):

| Portfolio | Largest risk source | Weight | Share of risk |
| :---- | :---- | :---- | :---- |
| Equal weight | CAT | 16.7% | 27.3% |
| Equal weight | XOM | 16.7% | 21.4% |
| Equal weight | PG / JNJ | 16.7% each | 8.1% / 9.8% |
| Defensive | JNJ | 50% | 53.9% |
| Cyclical | CAT | 50% | 54.7% |
| Minimum variance | PG, JNJ | 35.2%, 29.9% | 34.1%, 31.6% |

GARCH(1,1) persistence α + β on the last year: AAPL 0.96, tech and banks 0.995, minimum variance 0.99, equal weight 0.68, cyclical 0.57, defensive 0.02.

**Discussion**  
**Diversification shows up exactly where the theory says.** Six equally weighted stocks had a volatility of 12.8% over the last year, although none of the six stocks was below 14.9% on its own. The diversification ratio of 1.64 says the portfolio's volatility was 39% below the weighted average of its parts. Holding AAPL alone gave 20.3% volatility, a 2.67% one-day 99% VaR (about $26,700 per $1 million) and a 31% drawdown, with no diversification benefit by definition (ratio 1.00).

**Capital weights are not risk weights.** In the equal-weight portfolio, CAT held a sixth of the capital but contributed 27% of the risk, and PG only 8%. Volatile stocks that move with the rest of the portfolio dominate its risk, while low-volatility, weakly correlated stocks contribute little. The minimum-variance portfolio is the extreme case. Each stock's share of risk is within 2 percentage points of its weight. For the exact minimum-variance portfolio of a covariance matrix the two are equal, since every asset's marginal risk is the same there. It reached the lowest volatility (11.1%), VaR (1.58%) and ES of the six, even though its weights were fixed using only the first year.

**Two names do not diversify, even defensive ones.** PG and JNJ together had the same last-year volatility as the six-stock portfolio (12.8%), but a higher 99% VaR (1.91% vs 1.73%) and a larger drawdown (18% vs 16%). With a diversification ratio of only 1.22, a bad day for one of them is felt in full. The cyclical pair (XOM and CAT) had the highest risk on every measure (22.5% volatility, 3.13% VaR) and the highest return (31% a year). Over 2021-2023 the extra risk paid. Nothing in the app predicts whether it will again.

**Volatility behaves differently by portfolio.** The GARCH fits show strong volatility clustering for AAPL alone and for tech and banks (persistence 0.96 to 0.995), and almost none for the defensive pair (0.02). The defensive pair's risk is close to constant, so the adaptive models have nothing to adapt to. Tech and banks show the gap this produces. Its historical VaR (2.24%) is well above the combined forecast (1.84%), because the historical window still contains 2022's volatile days while GARCH and EWMA see that volatility has faded. The app shows that disagreement model by model rather than hiding it in one number.

**Limitations.** One three-year period ending in a calm market, six large US stocks, and long-only weights rebalanced daily. The app's live data (Yahoo Finance, via yfinance) could not be tested from the build machine. Its parsing is tested against stand-in data in the layouts yfinance returns, and the first live download happens on the user's machine. Risk contributions and correlations use one fixed 250-day window.

**Conclusion**  
The app turns the engine into something anyone can use: enter tickers and weights, get tomorrow's VaR and ES with each model's track record, plus where the risk comes from. On the example portfolios its answers are coherent and teach the standard lessons with real numbers. Diversification across six stocks cut risk by about 40% relative to the stocks' own volatilities. A single stock or a single sector raised the one-day VaR by 50% to 80%. Capital weights can hide very unequal risk contributions. And a minimum-variance portfolio fixed on one year of data stayed the least risky portfolio for the two years that followed.

**File Paths**  
/python_reference/experiments/portfolio-app/example_portfolios.py with csv results (portfolios, risk_contributions) in /python_reference/experiments/portfolio-app/results/. The app: /python_reference/qre/web/ (server, data, analytics, runner, static front end).
