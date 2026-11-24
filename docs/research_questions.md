**Research questions**

Each experiment asks one question. This file lists them with the short answer and where the write-up is. Folders are under `cpp/experiments/` unless marked (py), which are under `python_reference/experiments/`.

| # | Question | Short answer | Folder |
| :---- | :---- | :---- | :---- |
| 1 | How fast do sample mean, variance and quantiles converge? | At 1/√N, and tail quantiles need far more data than the mean | 01-probability |
| 2 | When does a sample covariance matrix stop being usable? | As the number of assets approaches the number of observations | 02-covariance-estimation |
| 3 | How stable is a volatility estimate across lookback windows? | Short windows are noisy and long ones stale | 03-statistics (py) |
| 4 | When does Newton-Raphson fail? | Flat derivatives and bad starting points | 04-numerical-methods |
| 5 | Which solver recovers implied volatility most reliably? | Brent; Newton is fastest where it works | 04-greeks |
| 6a | How does a Monte Carlo price converge, and what does variance reduction buy? | 1/√N; antithetic and control variates cut the constant | 01-mc-convergence |
| 6b | Where is Monte Carlo worth its cost against Black-Scholes? | Only where no closed form exists | 03-bs-vs-mc |
| 7 | Do historical, parametric and Monte Carlo VaR/ES agree? | Roughly at 95%, and they diverge in the tail | 05-var-es |
| 8 | How much do data errors distort VaR/ES? | One bad tick can dominate ES; hardening recovers most of it | data-hardening |
| 9 | What can 500 days of backtesting prove? | Less than expected: the tests have little power at 99% | backtesting |
| 10 | Do EWMA and filtered historical simulation fix the backtest? | They fix clustering of exceptions | adaptive-vol |
| 11 | Does estimating volatility dynamics (GARCH) beat fixing them? | Slightly, and the estimates are noisy | garch |
| 12 | Do fat-tailed shocks fix the ES understatement? | Yes, most of it | garch-t |
| 13 | Do the identities of stochastic calculus hold numerically? | Yes, at the rates theory predicts | stochastic-processes |
| 14a | How biased is Heston simulation, and what shapes the smile? | Euler is biased at low Feller ratios; ρ sets the skew | 06-heston |
| 14b | Can Heston be calibrated reliably? | Some parameters are barely identified | 07-calibration |
| 15 | Do shrinkage and random-matrix clipping fix large covariance matrices? | Yes, out-of-sample risk falls | 08-shrinkage |
| 16 | How uncertain is a risk number? | More than the spread between models | 09-evt |
| 17 | What do quasi-random points and AAD buy? | Orders of magnitude in paths; all Greeks for about 2 prices | advanced-mc |
| 18 | Can scoring models daily beat any one of them? | Averaging beats selecting | 10-adaptive-engine |
| 19 | Is the production run reproducible and thread-invariant? | Yes, to the last bit | production |
| 20 | What does the app say about six example portfolios? | Diversification shows in risk shares, not weights | portfolio-app (py) |
| 21 | What spreads and default probabilities does Merton imply? | Right shapes, near-zero short spreads | credit |
| 22 | Can a neural network hedge better than Black-Scholes delta? | Yes, once trading costs or stochastic volatility matter | deep-hedging |
| 23 | How should a large order be sold? | Front-loaded, and faster the more volatile the stock | execution |
| 24 | What does each operation cost, and what dominates a risk run? | GARCH fits; AAD gives 202 Greeks for 1.6 prices | docs/benchmarks.md |
| 25 | What would past crises do to a portfolio today, and do VaR models cover crisis days? | Losses of 2 to 55% against one-day VaR of 1.4 to 3.8%; volatility-scaled models breach 2.5 to 3.7% of crisis days, not 1% | stress |
| 26 | Which construction methods hold up out of sample, does component ES predict where losses come from, and how should 10-day VaR be computed? | Minimum variance yes, mean-variance no; component ES errors of 4 to 10 points against 11 to 42 for capital weights; √10 × an adaptive model | portfolio |

**Open questions**

These are not answered yet. Phases 20 onward take up several of them.

- How do the risk models behave on portfolios with options, where losses are not linear in returns?
- Does Heston calibrate to a real option surface as well as it does to its own synthetic prices?
- How much of a stock portfolio's risk is market beta, and how much is specific to the stocks?
- What does it cost to actually liquidate a position, and how much does that add to its VaR?
- Does Merton's distance to default rank real companies sensibly when given their real balance sheets?
- Do Gaussian correlations understate joint crashes, and by how much?
