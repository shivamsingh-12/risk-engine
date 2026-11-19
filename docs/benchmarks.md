**Experiment 24 (Phase 19): Benchmarks — Where the Time Goes**

**Objective**  
The reason for writing the engine in C++ is speed, but until now the repository had no numbers to support that, only timings scattered through experiment write-ups. This experiment measures the building blocks with a proper benchmark harness: pricing, random number generation, covariance estimation, model fitting and the full daily risk run. The aim is to know what each operation costs, what dominates a risk run, and how it scales with problem size and threads.

**Methodology**  
The suite in `cpp/benchmarks/` uses Google Benchmark, which repeats each operation until the timing is stable and reports the mean time per call. It is built with `-DQRE_BUILD_BENCHMARKS=ON` in a Release build (`-O3`, no `-march=native`) and run from the repo root:

    cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release -DQRE_BUILD_BENCHMARKS=ON && cmake --build build -j
    build/benchmarks/qre_bench

- **Pricing:** a Black-Scholes price, an implied-volatility solve, a Monte Carlo European price at 10⁴ to 10⁶ paths, and 16,384 Sobol points turned into normals in 1, 16 and 64 dimensions.
- **Greeks:** all 2N + 2 sensitivities of an N-asset basket call on 4,096 paths, by AAD, by the pathwise method and by bumping each input, for N = 10, 50 and 100. The price alone is timed for comparison.
- **Covariance:** sample covariance, Ledoit-Wolf shrinkage and random-matrix clipping on 500 days of returns for 10, 50 and 200 assets.
- **Risk models:** historical VaR/ES, a GARCH(1,1) fit and a Student-t GARCH fit, each on a 250-day window.
- **Daily risk run:** the whole of `configs/daily_risk.conf` (8 models, 2 confidence levels, 503 backtest days on the bundled data) with 1, 2, 4 and 8 threads, timed by wall clock.

The numbers below are from the 2-core build machine (Intel Xeon at 2.1 GHz, GCC 13). They are a baseline, not a claim about any other machine. Run the suite to get your own.

**Results**

Pricing and random numbers:

| Operation | Time | Throughput |
| :---- | :---- | :---- |
| Black-Scholes price | 26 ns | 39 million per second |
| Implied volatility solve | 421 ns | 2.4 million per second |
| Monte Carlo European price, 10,000 paths | 0.36 ms | 29 million paths per second |
| Monte Carlo European price, 1,000,000 paths | 37 ms | 27 million paths per second |
| Sobol normals, 16,384 points × 1 dimension | 0.57 ms | 29 million draws per second |
| Sobol normals, 16,384 points × 64 dimensions | 41 ms | 26 million draws per second |

Greeks of an N-asset basket call (2N + 2 sensitivities, 4,096 paths):

| Assets | Sensitivities | Price only | AAD | Pathwise | Bumping | AAD / price | Bumping / AAD |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 10 | 22 | 0.32 ms | 0.88 ms | 0.45 ms | 19 ms | 2.7 | 22 |
| 50 | 102 | 2.3 ms | 5.4 ms | 2.2 ms | 406 ms | 2.4 | 75 |
| 100 | 202 | 6.5 ms | 10.5 ms | 7.5 ms | 2,769 ms | 1.6 | 264 |

Covariance estimation on 500 days:

| Assets | Sample covariance | Ledoit-Wolf | Random-matrix clipping |
| :---- | :---- | :---- | :---- |
| 10 | 15 μs | 44 μs | 15 μs |
| 50 | 0.20 ms | 0.75 ms | 0.41 ms |
| 200 | 3.1 ms | 11.6 ms | 9.5 ms |

Risk models on a 250-day window:

| Operation | Time |
| :---- | :---- |
| Historical VaR/ES | 1.7 μs |
| GARCH(1,1) fit | 0.79 ms |
| Student-t GARCH fit | 4.1 ms |

Daily risk run:

| Threads | Wall time | Speed-up |
| :---- | :---- | :---- |
| 1 | 13.5 s | 1.00 |
| 2 | 6.9 s | 1.97 |
| 4 | 6.8 s | 1.98 |
| 8 | 7.0 s | 1.93 |

**Discussion**  
**Closed forms are effectively free, and simulation is cheap per path.** A Black-Scholes price takes 26 nanoseconds, and an implied volatility takes 16 times that. Monte Carlo runs at 27 to 29 million paths per second and the rate holds from 10,000 to a million paths, so the cost is linear with no fixed overhead worth noticing. A million-path price takes 37 ms, about 1.4 million times the closed form: the measured version of Experiment 6b's conclusion that Monte Carlo is for problems with no formula. Sobol points cost about the same per draw as pseudo-random ones, so the accuracy gain of Experiment 17 comes at no extra cost per path.

**AAD's cost does not grow with the number of sensitivities; bumping's does.** For 100 assets, AAD returned all 202 sensitivities in 1.6 times the cost of one price. Bumping took 427 prices' worth, 264 times slower than AAD. The ratio grew from 22 at 10 assets to 264 at 100, as theory says it should: bumping does one or two repricings per input, while the adjoint sweep goes backwards through the calculation once. At 10 assets AAD's fixed overhead (recording the tape) made it 2.7 times a price, and the hand-derived pathwise method was about twice as fast as AAD. Pathwise needs the derivative worked out by hand for each payoff. AAD does not.

**Covariance cleaning costs a few times the sample covariance.** Ledoit-Wolf took 3 to 4 times as long as the sample covariance, and random-matrix clipping 1 to 3 times, because it needs an eigendecomposition. Even at 200 assets both are around 10 ms. Experiment 15 showed what they buy in out-of-sample risk, and at this cost there is no reason to skip them.

**GARCH fitting dominates a risk run.** Historical VaR takes 1.7 microseconds per window. A GARCH fit takes 0.79 ms, about 470 times longer, and a Student-t GARCH fit 4.1 ms, about 2,400 times longer, because each is a numerical optimization of a likelihood. The daily run refits its models on each of 503 backtest days, and four of its eight models are GARCH-based. That is where its 13.5 seconds go.

**The engine scales with cores, not threads.** Two threads on two cores gave a speed-up of 1.97, almost exactly linear, because each backtest day is independent. Four and eight threads gave nothing more on this machine: there are only two cores. On a machine with more cores the run should keep scaling until it runs out of them. That was not measured here.

**Limitations.** One machine, with two cores and a shared host, so timings vary by a few percent between runs and the thread scaling stops at 2. The build does not use `-march=native`, which would allow wider vector instructions on newer CPUs. The Python reference is not benchmarked here: Experiment 19 compares the two engines on the full run, where C++ was 36 times faster.

**Conclusion**  
Formulas cost nanoseconds, simulation about 35 nanoseconds a path, and fitting a volatility model about a millisecond. A daily risk run is therefore dominated by GARCH fits, and it parallelizes almost perfectly across days. AAD delivers what it promises: 202 sensitivities for 1.6 times the cost of a price, where bumping needs 427 times.

**File Paths**  
/cpp/benchmarks/bench_pricing.cpp, bench_linalg.cpp, bench_risk.cpp. Raw output of the run above in /docs/benchmarks_raw.txt
