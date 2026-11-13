**qre: a quantitative risk engine, built phase by phase**

qre is a C++23 risk engine with a Python reference implementation of every model. I built it as a learning project, to learn the math used in finance by implementing it from first principles: from probability and linear algebra through option pricing, stochastic volatility, tail risk and credit, to hedging and trade execution.

**Status:** all 18 planned phases are done. The engine prices derivatives, simulates portfolios, forecasts and backtests risk across ten models, and runs as a command-line tool and as a local web app. I'm keeping it open for new topics.

**How it was built**  
Every phase follows the same order:
1. Research the math (the notes behind each phase).
2. Write a Python reference implementation.
3. Write the C++ implementation.
4. Test both, including tests that check the C++ against the Python to rounding.
5. Run an experiment that asks one concrete question about the method, and write up the answer.

The experiment write-ups (in `cpp/experiments/` and `python_reference/experiments/`) record what each method gets right, where it breaks, and what I found along the way, including bugs that only an experiment exposed.

**What it can do**

| Area | What's implemented |
| :---- | :---- |
| Foundations | RNG, distributions (normal, Student-t), covariance and Cholesky, root finding (bisection, Newton, Brent), Nelder-Mead |
| Data | validated CSV loading, corporate actions, quality checks (stale prices, bad ticks, unadjusted splits), point-in-time windows |
| Pricing | Black-Scholes with Greeks and implied volatility; Monte Carlo with variance reduction; Merton jump-diffusion; Heston (closed form, simulation, calibration) |
| Advanced Monte Carlo | Sobol quasi-random numbers, Brownian bridge, pathwise and likelihood-ratio Greeks, adjoint algorithmic differentiation (AAD) |
| Risk models | historical, parametric, Monte Carlo, EWMA, filtered historical simulation, GARCH, GARCH-t, extreme value theory (POT, conditional EVT) |
| Validation | rolling backtests, Kupiec and Christoffersen tests, Basel traffic light, ES backtest (Z2), FZ0 scoring, Diebold-Mariano tests |
| High dimensions | PCA, Ledoit-Wolf shrinkage, random-matrix eigenvalue clipping |
| Model combination | an adaptive engine that scores every model daily and combines them |
| Credit | Merton structural model, KMV asset estimation, Black-Cox first-passage default |
| Hedging | deep hedging: a neural-network policy trained to minimize CVaR, with hand-written backpropagation |
| Execution | Almgren-Chriss optimal liquidation and its efficient frontier |
| Tools | `qre_run`, a config-driven, multi-threaded daily risk report; a local web app for analyzing any portfolio |

**Quick start**

    cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
    ctest --test-dir build                                    # C++ tests
    (cd python_reference && python -m pytest -q)              # Python tests

    build/apps/qre_run --config configs/daily_risk.conf       # daily risk report -> runs/daily/report.md
    pip install yfinance                                      # optional, for live prices in the app
    PYTHONPATH=python_reference python -m qre.web --open      # portfolio risk app at http://127.0.0.1:8000

`configs/README.md` documents the run configuration, and `python_reference/qre/web/README.md` the app. On macOS, if the Apple compiler is too old for C++23, configure with `-DCMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++`.

**Roadmap**

| Phase | Focus | Experiment | Status |
| :---- | :---- | :---- | :---- |
| 0 | Foundation and repository (CMake, CI, tooling) | | Done |
| 1 | Probability and randomness | 1: convergence of empirical estimators | Done |
| 2 | Calculus and linear algebra | 2: sample covariance stability as N approaches T | Done |
| 3 | Statistics and financial data | 3: volatility estimates across lookback windows | Done |
| 4 | Numerical methods and optimization | 4: Newton-Raphson failure modes | Done |
| 5 | Financial math and derivatives (Black-Scholes) | 5: implied volatility recovery by three solvers | Done |
| 6 | Monte Carlo | 6a, 6b: convergence, variance reduction, Monte Carlo vs Black-Scholes | Done |
| 7 | Portfolio simulation and risk (VaR/ES) | 7: historical vs parametric vs Monte Carlo VaR/ES | Done |
| — | Data-layer hardening | 8: how data errors distort VaR/ES | Done |
| — | Backtesting | 9: what 500 days of backtesting can prove | Done |
| — | Adaptive volatility (EWMA, GARCH, GARCH-t) | 10 to 12: EWMA and FHS, GARCH, Student-t GARCH | Done |
| 8 | Stochastic processes | 13: stochastic calculus checked numerically | Done |
| 9 | Stochastic volatility (Heston) | 14a, 14b: simulation bias, smile shape, calibration | Done |
| 10 | High-dimensional risk (RMT) | 15: shrinkage and random-matrix clipping | Done |
| 11 | Tail risk and model uncertainty | 16: extreme value theory and model risk | Done |
| 12 | Advanced Monte Carlo and AAD | 17: quasi-Monte Carlo and Greeks by AAD | Done |
| 13 | Adaptive risk engine | 18: scoring and combining models | Done |
| 14 | Production research system | 19: reproducibility, parallelism and cross-language checks | Done |
| 15 | Interactive portfolio risk app | 20: six example portfolios | Done |
| 16 | Merton credit model | 21: spreads, first passage and asset estimation | Done |
| 17 | Deep hedging | 22: neural hedging with costs and stochastic volatility | Done |
| 18 | Optimal execution | 23: Almgren-Chriss trade-offs | Done |

Two changes from the original plan:
- **Data layer.** I hadn't planned a data layer, but by Phase 7 it was clear it was needed before going further. Real prices have gaps, stale quotes, bad prints and splits, and every later result depends on catching them.
- **Backtesting and validation.** A risk model only matters if its forecasts hold up against what actually happened. After hardening the data I added out-of-sample backtesting, and every later model is judged by it.

**Some things the experiments found**
- **Model choice vs uncertainty:** the ten risk models disagreed on next-day VaR by about ±9%, but one model's own sampling uncertainty was larger (±15%).
- **Combining models:** averaging the models beat picking the best recent one. The data could not tell the good models apart.
- **Optimizer model risk:** on two of 503 windows, the C++ and Python GARCH optimizers found different peaks of an almost flat likelihood, and VaR moved by up to 9%.
- **Quasi-Monte Carlo:** Sobol points with a Brownian bridge needed 1/4,000th of the paths plain Monte Carlo did, and AAD computed 202 sensitivities for about 2 times the cost of one price.
- **Credit:** Merton's model puts near-zero spreads on short-maturity debt (the credit spread puzzle). Estimating asset volatility from historical equity volatility understates a deteriorating firm's default probability badly.
- **Deep hedging:** a small network trained only on the tail of hedging losses matched Black-Scholes delta when trading was free. With 1% trading costs it cut the 95% CVaR of hedging losses by 21%, mostly by trading 29% less.
- **Execution:** the first reduction in execution risk away from TWAP is nearly free, and a risk-averse seller front-loads the order and finishes within a few days, however long it is allowed.
- **Bugs found by experiments, now fixed with regression tests:**
  - GARCH-t degrees of freedom drifting to 10¹⁶
  - a Student-t function that failed on macOS only
  - a GPD fit started outside its support

**Why Python and C++?**  
Python is for prototyping algorithms and checking closed forms; C++23 is for performance. Every C++ module is checked against its Python counterpart before I treat it as correct, which separates language-level bugs from mathematical errors. CI builds and tests both on every push to main, and also runs a daily risk report through both engines on the real data, failing if they disagree beyond 10⁻⁹.

**Repository layout**

| Path | Contents |
| :---- | :---- |
| cpp/include/qre/ | the engine (header-only): core, data, distributions, linalg, pricing, montecarlo, models, risk, stats, credit, hedging, execution, engine |
| cpp/apps/ | qre_run |
| cpp/tests/, python_reference/tests/ | GoogleTest and pytest suites |
| cpp/experiments/, python_reference/experiments/ | one folder per experiment: code, results (CSV) and write-up |
| python_reference/qre/ | Python reference implementations, the Python engine and the web app |
| configs/ | run configurations for qre_run and the app |
| data/ | the bundled daily prices (six large US stocks, 2021 to 2023) |
