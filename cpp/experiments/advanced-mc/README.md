**Experiment 17 (Phase 12): Advanced Monte Carlo — Quasi-Random Points, the Brownian Bridge, and Greeks by AAD**

**Objective**  
Experiment 6a showed plain Monte Carlo converging at the textbook rate of N^(-1/2): four times the paths for half the error. This experiment tests two ways around that cost. The first is to make the simulation itself converge faster, by replacing random points with a low-discrepancy (Sobol) sequence and building paths with a Brownian bridge. The second is about sensitivities. A risk system needs Greeks as well as prices, and computing them by re-running the simulation once per input gets expensive quickly. The experiment compares the main Monte Carlo Greek estimators, including adjoint algorithmic differentiation (AAD), on accuracy and on cost.

**Methodology**  
*Part 1, QMC convergence.* A geometric-average Asian call (S = K = 100, r = 5%, σ = 20%, T = 1) has an exact price, so the error can be measured directly. Each path needs one normal per monitoring date, so 4, 16 and 64 dates mean 4, 16 and 64 dimensions. Three methods are compared:
- **MC:** pseudo-random normals, path built from increments.
- **Sobol:** a randomly shifted Sobol sequence, path built from increments.
- **Sobol + bridge:** the same points, with the path built by the Brownian bridge. The first coordinate sets the final value, the second the midpoint, and so on.

Each runs at N = 2^7 to 2^16 paths, with RMSE over 32 independent randomizations and the convergence rate fitted as the slope of log RMSE against log N. The arithmetic-average Asian (the traded contract, no closed form) is compared by its standard error across 32 randomizations at N = 2^14.

*Part 2, Greek estimators.* Three estimators for a European call (S = 100, K = 105, r = 3%, σ = 25%, T = 0.8; 100,000 paths):
- pathwise (differentiate the payoff along each path)
- likelihood ratio (differentiate the density)
- central-difference bumping with common random numbers

A digital call (pays 1 if S_T > K) is used for delta, because its payoff has a jump. There the pathwise estimate is identically 0, so delta comes from the likelihood ratio and from bumping with spot bumps from 0.01 to 20.

*Part 3, AAD cost.* A basket call on N = 1 to 100 assets (spots 100, vols spread from 15% to 35%, correlation 0.3, equal weights, K = 100, 20,000 paths). All 2N + 2 sensitivities (every spot, every vol, the rate and the strike) are computed three ways:
- AAD: the engine's own tape, recording each path, sweeping backwards and rewinding.
- The hand-derived pathwise formulas.
- Central-difference bumping, which takes 2(2N + 2) extra pricings.

Each is timed against one price (median of repeated runs). Timing depends on the machine and is written to its own file.

C++ only: the Python reference's Sobol points match `scipy.stats.qmc.Sobol` bit for bit in all 64 dimensions, and the C++ points match the Python ones. The Python AAD reproduces the analytic Black-Scholes delta, vega, rho and theta to 1e-13. In both languages, AAD on the basket reproduces the hand-derived pathwise gradient to rounding.

**Results**

Part 1: geometric Asian, RMSE at N = 65,536 paths (relative to the exact price), fitted convergence rate, and how many times more paths plain MC would need for the same error:

| Dates (dimension) | Method | RMSE | Relative | Rate | MC paths needed |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 4 | MC | 0.0381 | 0.57% | -0.51 | 1× |
| | Sobol | 0.00054 | 0.008% | -0.89 | 4,900× |
| | Sobol + bridge | 0.00032 | 0.005% | -0.97 | 14,600× |
| 16 | MC | 0.0333 | 0.57% | -0.50 | 1× |
| | Sobol | 0.00195 | 0.033% | -0.68 | 290× |
| | Sobol + bridge | 0.00063 | 0.011% | -0.85 | 2,800× |
| 64 | MC | 0.0366 | 0.65% | -0.45 | 1× |
| | Sobol | 0.00379 | 0.067% | -0.66 | 93× |
| | Sobol + bridge | 0.00057 | 0.010% | -0.87 | 4,100× |

Arithmetic Asian, standard error at 16,384 paths (variance reduction relative to MC):

| Dates | MC | Sobol | Sobol + bridge |
| :---- | :---- | :---- | :---- |
| 16 | 0.0695 | 0.0047 (214×) | 0.0021 (1,144×) |
| 64 | 0.0522 | 0.0072 (52×) | 0.0021 (647×) |

Part 2: Greek estimators (100,000 paths):

| Option / Greek | Exact | Pathwise | Likelihood ratio | Bump |
| :---- | :---- | :---- | :---- | :---- |
| Call delta | 0.5004 | 0.5040 ± 0.0019 | 0.5062 ± 0.0042 | 0.5041 ± 0.0019 (h = 1) |
| Call vega | 35.68 | 35.99 ± 0.22 | 36.84 ± 0.77 | 35.99 ± 0.22 (h = 0.0025) |
| Digital delta | 0.02793 | 0 (always) | 0.02814 ± 0.00013 | see below |

Digital delta by bumping:

| Bump h | Estimate | Standard error | Error vs exact |
| :---- | :---- | :---- | :---- |
| 0.01 | 0.0272 | 0.0037 | -2.5% |
| 0.1 | 0.0282 | 0.0012 | +1.0% |
| 1 | 0.0276 | 0.00036 | -1.2% |
| 5 | 0.0274 | 0.00014 | -1.8% |
| 20 | 0.0209 | 0.00003 | -25.3% |

Part 3: basket call, all sensitivities (timing on the build machine; the ratios are what matter):

| Assets N | Sensitivities | AAD time / price time | Bump time / price time | Bump / AAD | AAD vs pathwise, max difference |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | 4 | 2.9 | 9 | 3 | 2e-13 |
| 5 | 12 | 3.0 | 25 | 9 | 2e-13 |
| 10 | 22 | 2.8 | 45 | 16 | 2e-13 |
| 20 | 42 | 2.6 | 81 | 31 | 2e-13 |
| 50 | 102 | 2.3 | 191 | 81 | 2e-13 |
| 100 | 202 | 1.9 | 361 | 191 | 2e-13 |

Bumped deltas and vegas agreed with AAD to within 1.4e-4 in relative terms everywhere. The differences come from the few paths where a bump moves the basket across the strike.

**Discussion**  
**Quasi-random points change the convergence rate, not just the constant.** Plain MC converged at a rate of -0.45 to -0.51, the expected N^(-1/2). Randomized Sobol points converged at -0.89 with 4 dimensions. To match the Sobol error at 65,536 paths, MC would have needed about 4,900 times as many paths: roughly 320 million instead of 65,536. Because the rate is faster, that advantage keeps growing with N; it is not a fixed variance-reduction factor like the antithetic and control-variate techniques of Experiment 6a. The random digital shift keeps the estimator unbiased and gives an honest standard error from independent shifts, which plain (deterministic) QMC cannot.

**Dimension erodes the gain unless the important randomness comes first.** With paths built from increments, Sobol's rate fell from -0.89 at 4 dimensions to -0.68 at 16 and -0.66 at 64. Its advantage over MC shrank from 4,900× to 93×. Sobol points are most evenly spread in their first few coordinates, and the incremental construction gives every coordinate equal weight in the final average. The Brownian bridge puts the coordinates that matter most first: the first sets the endpoint, which dominates the average. With the bridge the rate stayed at -0.85 to -0.97 and the advantage held at 2,800× to 14,600× at every dimension. The paths have exactly the same distribution either way, as the tests confirm; only the mapping from coordinates to paths changed. The same pattern holds for the arithmetic Asian, where the bridge gave 650 to 1,100 times less variance than MC.

**Pathwise derivatives are the default; the likelihood ratio is the fallback.** For the European call, pathwise and likelihood-ratio estimates were both consistent with the exact Greeks (within 2 standard errors). The likelihood ratio was much noisier, by 2.2 times in standard error for delta and 3.5 for vega, so 5 to 12 times more paths for the same precision. Central-difference bumping with common random numbers gave essentially the pathwise numbers, because for a continuous payoff a small bump differentiates the same paths. The digital call shows where pathwise fails. Its payoff is flat on either side of the strike, so the pathwise delta is exactly 0, whatever the true delta. The likelihood ratio, which never differentiates the payoff, got it right (0.0281 ± 0.0001 against 0.0279). Bumping a jump is a trade-off with no good setting. A small bump (h = 0.01) has 28 times the likelihood-ratio standard error, because only the rare paths that end within h of the strike contribute. A large one (h = 20) is precise but 25% wrong, because it measures the average slope over a wide range. AAD inherits pathwise's blind spot: it differentiates the program as written, and an indicator has derivative 0.

**AAD makes the cost of Greeks independent of how many there are.** AAD reproduced the hand-derived pathwise gradient to 2e-13 for every basket size, so it is the same estimator, obtained mechanically. Its cost stayed at 1.9 to 3 times one price, from 4 sensitivities up to 202. Bumping grew linearly with the number of inputs, to 361 times one price at N = 100: 191 times slower than AAD for the same numbers. The AAD ratio actually falls with N, because the per-path bookkeeping is spread over more work. The hand-coded pathwise formulas were cheapest of all, but they had to be derived and coded for this one payoff. AAD gives the same result for any payoff the pricer can compute.

**Limitations.** QMC was tested on GBM Asian options only. Paths with jumps or stochastic volatility, or with barriers (discontinuous payoffs), are known to benefit less. Up to 64 Sobol dimensions are available. The randomization is a digital shift rather than full Owen scrambling, which converges somewhat faster for smooth integrands. The basket uses one time step. AAD is a simple scalar tape with no expression templates or checkpointing, so its constant could be lowered further; and correlation sensitivities, which AAD could also provide, are not computed. Smoothing of digital payoffs (replacing the jump with a narrow ramp so that pathwise and AAD work) is not tested.

**Conclusion**  
For smooth, path-dependent payoffs, randomized Sobol points with a Brownian bridge beat plain Monte Carlo by three to four orders of magnitude in path count at 65,536 paths, with a convergence rate near N^(-1). The bridge is what keeps that true at 64 dimensions. For sensitivities, the pathwise estimator should be the default and the likelihood ratio kept for discontinuous payoffs, where pathwise and AAD both return 0. AAD delivers every pathwise Greek at under 3 times the cost of one price, however many inputs there are, while bumping costs grow with the number of inputs. At 100 assets that is a factor of 191.

**File Paths**  
/cpp/experiments/advanced-mc/advanced_mc_experiment.cpp and csv with results (qmc_convergence, qmc_rates, qmc_arithmetic, greek_estimators, aad_accuracy, aad_timing) in /cpp/experiments/advanced-mc/results/
