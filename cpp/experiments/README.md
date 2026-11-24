**Experiments**

One folder per experiment, each with its code, a `results/` folder of CSV files and a README write-up (Objective, Methodology, Results, Discussion, Limitations, Conclusion). The folder names come from when each experiment was written, so their numeric prefixes are not the experiment numbers. This table is the index. Folders marked py are under `python_reference/experiments/`.

| Experiment | Topic | Folder | Language |
| :---- | :---- | :---- | :---- |
| 1 | Convergence of empirical estimators | 01-probability | cpp, py |
| 2 | Sample covariance stability as N approaches T | 02-linear-algebra (code), 02-covariance-estimation (results) | cpp, py |
| 3 | Volatility estimates across lookback windows | 03-statistics | py |
| 4 | Newton-Raphson failure modes | 04-numerical-methods | cpp |
| 5 | Implied volatility recovery by three solvers | 04-greeks | cpp |
| 6a | Monte Carlo convergence and variance reduction | 01-mc-convergence | cpp |
| 6b | Monte Carlo vs Black-Scholes | 03-bs-vs-mc | cpp |
| 7 | Historical vs parametric vs Monte Carlo VaR/ES | 05-var-es | cpp |
| 8 | How data errors distort VaR/ES | data-hardening | cpp |
| 9 | What 500 days of backtesting can prove | backtesting | cpp |
| 10 | EWMA and filtered historical simulation | adaptive-vol | cpp |
| 11 | GARCH(1,1) | garch | cpp |
| 12 | Student-t GARCH | garch-t | cpp |
| 13 | Stochastic calculus checked numerically | stochastic-processes | cpp |
| 14a | Heston simulation bias and smile shape | 06-heston | cpp |
| 14b | Heston calibration | 07-calibration | cpp |
| 15 | Shrinkage and random-matrix clipping | 08-shrinkage | cpp |
| 16 | Extreme value theory and model risk | 09-evt | cpp |
| 17 | Quasi-Monte Carlo and Greeks by AAD | advanced-mc | cpp |
| 18 | Scoring and combining models | 10-adaptive-engine | cpp |
| 19 | Reproducibility, parallelism, cross-language checks | production | cpp, py |
| 20 | Six example portfolios in the app | portfolio-app | py |
| 21 | Merton credit: spreads, first passage, asset estimation | credit | cpp |
| 22 | Deep hedging | deep-hedging | cpp |
| 23 | Optimal execution (Almgren-Chriss) | execution | cpp |
| 24 | Benchmarks: where the time goes | cpp/benchmarks (write-up in docs/benchmarks.md) | cpp |
| 25 | Stress testing: past crises, VaR in a crisis, conditional shocks, proxies | stress | cpp |
| 26 | Portfolio construction, component risk and 10-day VaR | portfolio | cpp |
| 27 | Streaming risk: incremental updates, drift, queues, pipeline latency | streaming | cpp |

Every experiment is its own executable, built into `build/experiments/`. Run one from the repo root and it rewrites its `results/` folder. `docs/research_questions.md` lists the question each one asks and the short answer.
