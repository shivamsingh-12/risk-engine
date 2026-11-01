**Experiment 13 (Phase 8): Stochastic Calculus, Checked Numerically**

**Objective**  
Phase 8 adds general SDE solvers, the Ornstein-Uhlenbeck mean-reverting process and Merton's jump-diffusion. This experiment checks four results from stochastic calculus numerically:
1. Euler-Maruyama converges at strong order 1/2 and Milstein at order 1, with both at weak order 1.
2. Brownian motion has quadratic variation T, so Itô and Stratonovich integrals differ by T/2.
3. Jumps produce a volatility smile that flattens with maturity.
4. Mean reversion is much harder to estimate than volatility.

**Methodology**  
*Part 1, convergence orders.* GBM with μ = 5% and σ = 40% over one year, whose exact solution is known. The strong error is the average absolute gap between each scheme's terminal value and the exact solution, using the same Brownian path at every step size (20,000 paths, a 512-step path summed down to coarser grids). The weak error is the gap in E[X_T²], with 200,000 paths. For Euler-Maruyama it is also compared with its exact theoretical value ((1+μh)² + σ²h)ⁿ − e^((2μ+σ²)T). The error in E[X_T] itself is smaller than the Monte Carlo noise for GBM, so it cannot be used.

*Part 2, Itô calculus.* 2,000 Brownian paths on [0, 1] with 10 to 10,000 steps. The sum of squared increments (quadratic variation) is compared with T and its predicted spread T√(2/n). The left-point (Itô) and average-point (Stratonovich) sums of W dW are compared.

*Part 3, Merton.* Spot 100, rate 5%, diffusion vol 20%, one jump per year on average with mean log-jump -10% and jump volatility 15%, maturities 0.1, 0.5 and 2 years, strikes 70 to 130. The closed-form price is compared with Monte Carlo (400,000 paths), and the closed-form prices are converted to Black-Scholes implied volatilities with the Phase 5 solver.

*Part 4, Ornstein-Uhlenbeck.* True κ = 3 (half-life of about 58 trading days), θ = 4%, σ = 5%, sampled daily. 1,000 independent samples of 1, 5 and 25 years were each fitted by exact maximum likelihood.

C++ only: every function is cross-validated against the Python reference (agreement to 1e-14 on a fixed Brownian path), and the Merton closed form is checked against direct numerical integration in the Python tests.

**Results**

Convergence (fitted log-log slope; theory in brackets):

| Error | Euler-Maruyama | Milstein |
| :---- | :---- | :---- |
| Strong, all step sizes (4-256) | 0.49 [0.5] | 0.91 [1] |
| Strong, finest three halvings (32-256) | 0.50 [0.5] | 0.97 [1] |
| Weak, E[X_T²] (2-64 steps) | 0.96 [1] | 0.99 [1] |

At 256 steps the Milstein strong error (0.00011) is 56 times smaller than Euler-Maruyama's (0.0060). The measured Euler-Maruyama weak errors track the exact theoretical bias, for example 0.00261 vs 0.00251 at 16 steps.

Itô calculus (T = 1):

| Steps | Mean quadratic variation | Its sd (theory) | Mean Stratonovich − Itô (theory 0.5) |
| :---- | :---- | :---- | :---- |
| 10 | 1.001 | 0.453 (0.447) | 0.5005 |
| 100 | 1.002 | 0.144 (0.141) | 0.5012 |
| 1,000 | 1.001 | 0.046 (0.045) | 0.5003 |
| 10,000 | 1.000 | 0.014 (0.014) | 0.4999 |

Merton implied volatilities (diffusion vol alone is 20%):

| Strike | T = 0.1 | T = 0.5 | T = 2 |
| :---- | :---- | :---- | :---- |
| 70 | 46.7% | 31.2% | 27.6% |
| 80 | 39.5% | 29.0% | 27.1% |
| 90 | 29.8% | 27.1% | 26.7% |
| 100 | 23.9% | 25.7% | 26.4% |
| 110 | 23.1% | 24.8% | 26.1% |
| 120 | 25.8% | 24.3% | 25.8% |
| 130 | 30.4% | 24.2% | 25.6% |
| Skew, IV(80) − IV(120) | 13.7 pts | 4.7 pts | 1.3 pts |

Monte Carlo agreed with the closed form at all 39 strike-maturity points (largest gap 2.4 standard errors).

Ornstein-Uhlenbeck estimates (true κ = 3, θ = 0.04, σ = 0.05; 1,000 samples each):

| Sample | κ mean (median) | κ sd | θ sd | σ mean | σ sd | Failed fits |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1 year | 8.26 (7.24) | 5.08 | 0.0172 | 0.0502 | 0.0022 | 2 |
| 5 years | 3.89 (3.69) | 1.39 | 0.0075 | 0.0500 | 0.0010 | 0 |
| 25 years | 3.14 (3.09) | 0.54 | 0.0033 | 0.0500 | 0.0005 | 0 |

**Discussion**  
**The convergence orders come out as theory says.** Euler-Maruyama's strong order is 0.50 and Milstein's approaches 1 (0.97 on the finer grids; the coarsest grid, with only 4 steps, is not yet in the asymptotic regime and drags the all-steps fit to 0.91). Both weak orders are about 1. The practical point: if only the distribution of X_T matters, as in option pricing, Euler-Maruyama is as good as Milstein. If paths themselves matter (path-dependent payoffs, or hedging along a path), Milstein's extra term, which is one application of Itô's lemma, buys a 56-fold accuracy gain at 256 steps for almost no extra cost.

**Quadratic variation is the whole difference between Itô and Stratonovich.** The sum of squared Brownian increments converges to T with exactly the predicted spread T√(2/n), which shrinks tenfold for every 100-fold increase in steps. The Stratonovich sum equals W_T²/2 exactly (it telescopes), and the Itô sum is smaller by half the quadratic variation, about 0.5 at every resolution. That is the dt term in Itô's lemma, the reason GBM has the -σ²/2 drift used since Phase 6. The Itô integral's mean is zero within noise, and its variance is close to the theoretical 0.5. The 1,000-step estimate (0.60) is about 2.4 standard errors high: the integral has a skewed χ²-type distribution, so its sample variance is noisy with 2,000 paths.

**Jumps create a smile, but it fades too fast.** With 20% diffusion volatility, Merton prices imply volatilities from 23% to 47%. At 0.1 years there is a pronounced smile: crash-like negative jumps lift low strikes, and jump variance lifts both wings. The skew collapses with maturity, from 13.7 to 4.7 to 1.3 volatility points, because over long horizons many jumps average out towards a normal distribution (the central limit theorem). Equity index skews observed in practice tend to decay more slowly than that, which is a well-known limitation of pure jump models and the usual motivation for stochastic volatility models such as Heston, the next step on the roadmap.

**Volatility is easy to estimate; mean reversion is not.** With one year of daily data, σ is estimated to within about 4% (sd 0.0022 on 0.05), but κ averages 8.3 against a true 3, and the long-run mean θ has a standard deviation of 1.7 percentage points on a level of 4%. Precision for σ comes from the number of observations; precision for κ and θ comes from the length of time covered, measured in half-lives. One year is only about four half-lives here. The upward bias of κ (+175% at 1 year, +30% at 5, +5% at 25) is the finite-sample bias of least squares on persistent series. In practice, a mean-reversion speed fitted on a year or two of data will overstate how fast deviations decay. The Python tests also show the converse: on a random walk with no mean reversion, the fit typically reports a small spurious κ rather than failing.

**Limitations.** One-dimensional SDEs only; GBM is the only test case with a closed-form solution for the convergence study. Merton uses lognormal jumps and fixed parameters, with no calibration to market prices. The OU study assumes the model is correct; with real data, model misspecification adds to the estimation error shown here.

**Conclusion**  
The Phase 8 machinery reproduces the core results of stochastic calculus to within Monte Carlo noise: the convergence orders of the two schemes, quadratic variation, and the Itô-Stratonovich gap. Merton's jump-diffusion matches its closed form everywhere, and the smile it produces flattens with maturity, which motivates stochastic volatility (Heston) next. The Ornstein-Uhlenbeck study is a caution for any mean-reverting model fitted to short histories: volatility is pinned down quickly, the speed of mean reversion is not, and it is biased upwards.

**File Paths**  
/cpp/experiments/stochastic-processes/stochastic_processes.cpp and csv with results (convergence, ito_calculus, merton_smile, ou_estimation) in /cpp/experiments/stochastic-processes/results/
