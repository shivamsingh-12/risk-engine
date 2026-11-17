**Architecture**

qre is two implementations of the same models, plus the tools built on them.

**Layout**

| Layer | C++ (`cpp/include/qre/`) | Python (`python_reference/qre/`) | What it holds |
| :---- | :---- | :---- | :---- |
| Foundations | `core`, `distributions`, `linalg`, `optimization`, `estimators`, `stats` | `linalg`, `stats` (numpy and scipy cover the rest) | RNG, normal and Student-t, covariance, Cholesky, shrinkage, PCA, root finding, Nelder-Mead |
| Data | `data` | `data` | CSV loading, corporate actions, quality checks, returns, rolling windows, point-in-time views |
| Pricing | `pricing`, `models`, `montecarlo`, `autodiff` | same names | Black-Scholes, Heston, Merton jumps, OU, GBM and SDE simulation, Sobol, Brownian bridge, Greeks, AAD |
| Risk | `risk` | `risk` | VaR/ES models, EWMA, GARCH, EVT, backtests, scoring, model combination |
| Applications of the models | `credit`, `hedging`, `execution` | same names | Merton credit, deep hedging, Almgren-Chriss |
| Engine | `engine`, `cpp/apps/qre_run.cpp` | `engine` | config parser, thread pool, the daily risk run, report writer |
| App | | `web` | local HTTP server, data download, analytics, React front end |

**Design decisions**

- **Header-only C++.** Every module is a header under `cpp/include/qre/`, and the CMake target `qre` is an INTERFACE library that carries the include path and Eigen. Nothing has to be linked or installed, and each experiment and test compiles only what it includes.
- **Python is the reference, C++ is the product.** A model is written in Python first, where numpy and scipy make the math easy to check against closed forms and other libraries. The C++ version is then tested against the Python numbers to rounding, usually with the Python values hard-coded in the C++ test. A disagreement is a bug in one of them, and finding which has caught real errors (see the README's list).
- **Errors are exceptions with types.** `core/exceptions.hpp` defines the hierarchy. Functions validate their inputs and throw, and the engine and CLI turn exceptions into messages and exit codes.
- **Randomness is explicit.** Every simulation takes a seed or an RNG. Parallel work derives one stream per task from the seed and the task index, so results do not depend on the number of threads.
- **The engine is a pure function of its config.** `qre_run` reads a `key = value` file, runs every model over a rolling window, and writes CSV files, a Markdown report and a manifest with a fingerprint of the inputs and outputs. The Python engine reads the same config and writes the same files, and `qre.engine.compare` checks the two runs against each other. CI does this on every push.
- **The web app is a thin layer.** It downloads and checks prices, writes a config, runs `qre_run` (or the Python engine if there is no build), and reads the report back. The risk numbers in the browser are the engine's own.

**How a daily risk number is produced**

1. `data/csv_loader.hpp` loads and validates the price file, and `data/quality.hpp` flags stale prices, bad ticks and unadjusted splits.
2. Prices become portfolio returns under the configured weights (`risk/pipeline.hpp`).
3. For each day in the backtest, each model forecasts next-day VaR and ES from the previous `window` days (`risk/backtest.hpp`). Days are independent, so they run on the thread pool (`engine/parallel.hpp`).
4. The forecasts are scored against what happened (FZ0 loss, `risk/scoring.hpp`) and combined (`risk/adaptive.hpp`).
5. Coverage tests (Kupiec, Christoffersen, Basel traffic light, the ES Z2 test) judge each model and the combination.
6. `engine/report.hpp` writes the results.

**Testing**

- `cpp/tests/` (GoogleTest) and `python_reference/tests/` (pytest), one file per topic with matching names.
- Three kinds of test: closed forms and identities, C++ against Python values, and regression tests for bugs found by experiments (`test_regressions.cpp`).
- The C++ tests execute 97% of the lines in `cpp/include/qre/` (measured with gcov; CI reports it on every run).
- CI builds with GCC on Linux and Clang on macOS, runs both suites, runs the C++ tests again under AddressSanitizer and UndefinedBehaviorSanitizer, and compares the two engines end to end.
