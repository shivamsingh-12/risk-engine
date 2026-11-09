**Experiment 14a (Phase 9): Heston Stochastic Volatility — Simulation Bias and the Shape of the Smile**

**Objective**  
Phase 8 ended with Merton's jump-diffusion producing a skew that collapses with maturity. Heston's model lets the variance itself follow a random, mean-reverting process that is correlated with the stock. This experiment answers two questions. How reliable is Monte Carlo simulation of Heston, given that the variance can approach zero? And how does Heston's smile behave across maturities compared with Merton's?

**Methodology**  
*Part 1.* An at-the-money one-year call is priced with the semi-closed form (Lewis integral of the characteristic function) and with full-truncation Euler Monte Carlo (200,000 paths, 4 to 256 time steps), for three parameter sets:
- the Feller condition 2κθ ≥ σ² holds (ratio 3.3)
- it is mildly violated (0.64)
- it is badly violated (0.08)

The bias is the Monte Carlo price minus the exact price. A separate simulation counts how many paths step the variance below zero at least once.

*Part 2.* Implied volatilities at 90%, 100% and 110% of spot are compared for maturities from one week to five years, for Heston (κ = 2, θ = 0.04, σ = 0.6, ρ = -0.7, v0 = 0.04) and for the Phase 8 Merton jump-diffusion. "Skew" is IV(90) minus IV(110).

C++ only: Heston prices match the Python reference to 1e-8. In the Python tests the reference agrees with the original 1993 two-probability formula to 1e-7, reduces to Black-Scholes when the volatility of variance vanishes, and matches Monte Carlo.

**Results**

Monte Carlo bias, ATM one-year call:

| Parameters | 2κθ/σ² | Steps | Bias | Bias in standard errors | Paths with variance below zero |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Feller holds | 3.33 | 4 | +0.058 | 2.0 | 22.3% |
| | | 64 | -0.009 | -0.3 | 0.4% |
| | | 256 | -0.015 | -0.5 | 0.02% |
| Mildly violated | 0.64 | 4 | +0.419 | 18.1 | 57.5% |
| | | 64 | +0.012 | 0.6 | 58.6% |
| | | 256 | -0.029 | -1.4 | 57.5% |
| Badly violated | 0.08 | 4 | +2.634 | 101.7 | 75.9% |
| | | 16 | +0.800 | 42.6 | 89.0% |
| | | 64 | +0.173 | 10.3 | 92.5% |
| | | 256 | +0.050 | 3.0 | 93.8% |

Smile across maturities (implied vol in %, skew in vol points):

| Maturity | Heston ATM | Heston skew | Merton ATM | Merton skew |
| :---- | :---- | :---- | :---- | :---- |
| 1 week | 19.9 | 9.0 | 22.0 | 13.2 |
| 0.1 year | 19.3 | 8.8 | 23.7 | 6.3 |
| 0.25 | 18.5 | 8.2 | 24.8 | 3.5 |
| 0.5 | 17.8 | 7.0 | 25.4 | 2.1 |
| 1 | 17.5 | 5.1 | 25.8 | 1.1 |
| 2 | 17.6 | 3.3 | 26.1 | 0.6 |
| 5 | 18.1 | 1.7 | 26.2 | 0.2 |

**Discussion**  
**When the Feller condition holds, simulation is easy.** With enough steps (64 or more) the bias is indistinguishable from Monte Carlo noise. Variance going negative is purely a discretization artifact: it happens on 22% of paths with 4 steps and almost never with 256, because the true process never reaches zero.

**When it fails, the variance genuinely reaches zero, and the bias depends on how badly.** In the mild case about 58% of paths touch zero at every step size, because the true process does. Even so, full truncation handles it well from 16 steps. In the bad case (2κθ/σ² = 0.08; calibrations to equity surfaces often violate Feller, though usually less severely) the bias is 101 standard errors with 4 steps and still 3 standard errors with 256 steps. It shrinks roughly in proportion to the step size, so cutting it further is expensive. Any Monte Carlo pricer for path-dependent products under a calibrated Heston model needs a bias check like this. Schemes designed for this regime, such as Andersen's quadratic-exponential scheme, are the standard fix.

**Heston and Merton produce complementary skews.** Heston's skew comes from the negative correlation between spot and volatility. It is almost flat at short maturities (9.0 to 8.2 vol points from one week to three months) and decays slowly after that (5.1 at one year, 1.7 at five). Merton's comes from jumps: it is steeper at one week (13.2) but collapses quickly (1.1 at one year). Equity index markets typically show short-dated skews steeper than Heston's and long-dated skews more persistent than Merton's. That is why practitioners combine the two (stochastic volatility plus jumps, the Bates model). Experiment 14b tests what happens when Heston alone is asked to fit a jump-driven surface.

**Heston's ATM vol sits below √θ.** Even with v0 = θ (both 20% vol), the ATM implied vol dips to 17.5% at one year and drifts back up towards 20% at long maturities. Volatility of volatility makes the return distribution fat-tailed, which pushes the centre of the implied vol curve down and the wings up.

**Limitations.** The bias study covers one ATM option and one maturity; deep out-of-the-money and path-dependent options are more sensitive. Full truncation is the only scheme tested. The smile comparison uses illustrative parameters for both models, not fitted ones.

**Conclusion**  
Heston prices can be computed accurately and quickly in semi-closed form. Monte Carlo simulation is reliable when the Feller condition holds and needs care when it fails badly, which is common in practice. The two models produce skews with opposite maturity profiles, flat-then-slow for Heston and steep-then-fast for Merton, which the calibration experiment builds on.

**File Paths**  
/cpp/experiments/06-heston/heston_experiment.cpp and csv with results (mc_bias, smile_term_structure) in /cpp/experiments/06-heston/results/
