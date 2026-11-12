**Experiment 19 (Phase 14): The Production Engine — Reproducible, Parallel, and Checked Against Its Own Reference**

**Objective**  
Phases 0 to 13 built the engine's models one experiment at a time. Phase 14 joins them into one program: `qre_run` reads a config file, loads and checks the prices, runs every model, scores and combines them, backtests them and writes a daily risk report. Its Python twin, `python -m qre.engine.run`, does the same. A production risk number has to meet three conditions that no single earlier experiment tested:
- the same inputs always give the same output, however the work is scheduled
- it arrives fast enough to be used
- an independent implementation agrees with it

This experiment tests all three on the real portfolio, and looks at the one place where the two implementations do not agree.

**Methodology**  
The job is `configs/daily_risk.conf`: the 6-stock equal-weight portfolio (hardened data, 753 prices), eight models (historical, parametric, EWMA, FHS, GARCH, FHS-GARCH, GARCH-t, conditional EVT) at 99% and 97.5%, a 250-day window, equal-weight combination. That is 8 × 2 × 503 = 8,048 forecasts: the 502 backtest days plus the forecast for the next, unseen day. Each forecast depends only on its own model, confidence level and day. The engine therefore spreads them over threads, with each worker taking the next task from a shared counter and writing to its own slot.

*Part 1, reproducibility.* The full job is run with 1, 2, 3, 4 and 8 threads, twice each. Every forecast, combined forecast, combination weight and next-day number is compared bit for bit with the single-thread run.

*Part 2, scaling.* Wall time of the forecasting stage for each thread count. This depends on the machine. Here it is a 2-core build machine, so the results show only the first doubling; on a laptop with more cores the experiment shows how far it goes.

*Part 3, cost per model.* Serial time per forecast for each model.

*Part 4, C++ vs Python.* `python_reference/experiments/production/cross_check.py` runs the same config through both engines and compares every forecast, every backtest verdict and the next-day report. Both engines also fit GARCH(1,1) on each of the 503 return windows and record the parameters and log-likelihood, to explain any differences.

**Results**

Part 1: reproducibility:

| Threads | Runs | Identical to 1 thread, bit for bit | Largest difference |
| :---- | :---- | :---- | :---- |
| 1 | 2 | yes | 0 |
| 2 | 2 | yes | 0 |
| 3 | 2 | yes | 0 |
| 4 | 2 | yes | 0 |
| 8 | 2 | yes | 0 |

The C++ tests check the same at small scale on every build, for the report files as well. The Python tests check it for 1 and 3 worker processes.

Part 2: forecasting stage on the 2-core build machine:

| Threads | Time | Speed-up | Efficiency |
| :---- | :---- | :---- | :---- |
| 1 | 15.6 s | 1.00× | 100% |
| 2 | 7.8 s | 1.99× | 99% |
| 3, 4, 8 | 7.7 to 8.0 s | 1.96× to 2.02× | 25% to 65% |

Part 3: cost per model (serial, 1,006 forecasts each):

| Model | Time per forecast | Share of the job |
| :---- | :---- | :---- |
| Historical | 0.011 ms | 0.07% |
| Parametric | 0.006 ms | 0.04% |
| EWMA | 0.009 ms | 0.06% |
| FHS | 0.011 ms | 0.07% |
| GARCH | 0.78 ms | 5.0% |
| FHS-GARCH | 0.81 ms | 5.2% |
| Conditional EVT | 0.81 ms | 5.2% |
| GARCH-t | 13.1 ms | 84.3% |

Part 4: C++ against Python, largest relative difference over all 502 days, and days where the two differ by more than 1e-6:

| Series | 99%: largest difference | 99%: days > 1e-6 | 97.5%: largest difference | 97.5%: days > 1e-6 |
| :---- | :---- | :---- | :---- | :---- |
| Historical, parametric, EWMA, FHS | ≤ 1.2e-14 | 0 | ≤ 1.1e-14 | 0 |
| GARCH-t | 3.4e-7 | 0 | 2.8e-7 | 0 |
| GARCH | 9.3% | 1 | 9.3% | 1 |
| FHS-GARCH | 8.2% | 1 | 12.8% | 1 |
| Conditional EVT | 8.5% | 5 | 8.8% | 4 |
| Combination | 3.4% | 3 | 4.1% | 3 |

Backtest verdicts: all 18 rows (9 series × 2 levels) have the same breach count, traffic light and Kupiec p-value in both engines; mean FZ0 differs by at most 9e-5 in relative terms. Run time for the full job: C++ 7.9 s on 2 threads, Python 287 s on 2 worker processes, so C++ is 36 times faster.

GARCH(1,1) fits, 503 return windows (the backtest days and the next day):

| Window | C++ α, β | Python α, β | C++ log-likelihood | Python log-likelihood |
| :---- | :---- | :---- | :---- | :---- |
| 501 of 503 | same to 1e-4 | | same to 5e-10 | |
| 2023-12-15 | 0.018, 0.965 | 0.052, 0.769 | 847.5597 | 847.5614 (+0.0017) |
| Next day (after 2023-12-29) | 0.018, 0.966 | 0.080, 0.604 | 850.6040 | 850.6123 (+0.0084) |

Effect on the next-day report (99% VaR, C++ / Python): GARCH 1.741% / 1.718%, FHS-GARCH 1.699% / 1.685%, conditional EVT 1.682% / 1.641%, combination 1.729% / 1.719%. The other five models agree to every printed digit.

**Discussion**  
**Parallel and reproducible are compatible if the work is split by what it depends on.** Every forecast is a pure function of its model, confidence level, day and the price history, and each writes to its own slot. So the order in which threads finish cannot change anything. All ten runs matched the single-thread run bit for bit, including the combination weights and the report text. The Monte Carlo model, which is not in this config, ties its seed to the day rather than the thread for the same reason. Reproducibility here is designed in, not checked after the fact: the manifest records the config and a fingerprint of the input file, and no result depends on the thread count.

**The job is GARCH-t, and it parallelizes almost perfectly.** On two cores the forecasting stage ran 1.99 times faster with 2 threads (99% efficiency), and more threads than cores gained nothing, as expected. Because each worker takes the next task as it finishes, the 13 ms GARCH-t fits never hold up the cheap ones. One model, GARCH-t, is 84% of the work: its Student-t degrees of freedom add a third parameter and a slower likelihood. The four closed-form and EWMA models together cost 0.24%. Any speed-up effort should start with GARCH-t, for example by starting each day's fit from the previous day's estimate instead of three fixed starting points.

**Where both engines compute the same thing deterministically, they agree to rounding.** Historical, parametric, EWMA and FHS matched to about 1e-14, a few units in the last digit, over 4,000 forecasts. GARCH-t matched to 3e-7, the convergence tolerance of two different Nelder-Mead implementations (the engine's own in C++, scipy's in Python). Every backtest verdict was identical. CI now runs this comparison on every push to main, for the fast models, and fails if the engines drift apart by more than 1e-9.

**The disagreement is an optimizer finding a different peak, and it is model risk.** On 501 of 503 windows the two GARCH(1,1) fits are the same optimum. On the other two, the likelihood has two almost equally high peaks:
- one with high persistence (β ≈ 0.97, small α), where a shock fades slowly
- one with lower persistence (β ≈ 0.6 to 0.8, larger α), where it fades within days

The C++ optimizer settled on the first, scipy on the second. Scipy's peak is higher in both cases, but only by 0.002 and 0.008 log-likelihood units: no test could tell these two models apart on 250 days. Yet they forecast different volatility. On 2023-12-15 the two fits put 99% VaR 9% apart, and the difference carries through to FHS-GARCH, conditional EVT (both use the same filter) and the combination. The next-day forecast, the number this report actually exists to produce, is one of the two affected windows. GARCH's 99% VaR is 1.741% or 1.718% depending on which optimizer ran, with the same data, model and starting points.

This is the identifiability problem of Experiment 11 (GARCH parameters are poorly pinned down on a year of data), now seen from the user's side. Two reasonable implementations of the same model give different risk numbers, and nothing in the backtest flags it. The combination halves the effect: the reported 99% VaR differs by 0.6% (1.729% vs 1.719%). That is far smaller than the ±15% sampling uncertainty measured in Experiment 16, but it is not zero. I have left the C++ optimizer unchanged. Making it more thorough (more starting points, or a restart from the best point) would very likely close this gap. It would also change GARCH results on a handful of days in Experiments 11 to 18, so it is a decision to make deliberately.

**Limitations.** Scaling was measured on 2 cores only. The cross-check covers one portfolio and two calm years, and the GARCH disagreement could be more frequent in other data. The engine runs one portfolio with fixed weights; there is no scheduling, database or monitoring around it. Those belong to the system that would call `qre_run`, not to the engine.

**Conclusion**  
The engine now runs from a config file to a report in under 8 seconds in C++ and about 5 minutes in Python. It produces bit-identical results for any thread count and agrees with its independent Python implementation to rounding for every closed-form model, to 3e-7 for GARCH-t, and on every backtest verdict. The one systematic disagreement is a lesson rather than a bug. On 2 of 503 windows, GARCH(1,1)'s likelihood has two near-equal peaks with different volatility forecasts, and which one an optimizer finds moves a single model's VaR by up to 9% and the combined report by 0.6%. A risk system that runs only one implementation would never see this. Running two, and comparing them automatically, is what makes the numbers trustworthy.

**File Paths**  
/cpp/experiments/production/production_experiment.cpp and /python_reference/experiments/production/cross_check.py, with csv results (reproducibility, thread_scaling, model_cost, cpp_garch_fits, cross_language, verdicts, garch_optima, runtime) in /cpp/experiments/production/results/. The engine: /cpp/include/qre/engine/, /cpp/apps/qre_run.cpp, /python_reference/qre/engine/, /configs/.
