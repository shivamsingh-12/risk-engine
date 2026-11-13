**Experiment 21 (Phase 16): The Merton Credit Model — Spreads, First-Passage Default, and Estimating a Firm's Assets**

**Objective**  
Everything so far has priced or risk-managed market prices. Credit risk asks a different question: will a borrower pay back? Merton's structural model answers it with the option pricing of Phase 5. Shareholders hold a call option on the firm's assets with the debt as strike, and creditors hold riskless debt minus a put. This experiment looks at:
- the credit spreads the model implies
- how much a more realistic default rule (Black-Cox first passage) changes default probabilities
- how well the unobservable asset value and volatility can be recovered from equity prices, which is the step every practical use of the model (KMV-style distance to default) depends on

**Methodology**  
*Part 1, spread term structures.* Four firms with assets of 100 and r = 3%, at maturities from 0.1 to 30 years. Each gets the credit spread, the risk-neutral default probability and the expected recovery.

| Firm | Debt face | Asset volatility |
| :---- | :---- | :---- |
| Safe | 40 | 20% |
| Medium | 70 | 25% |
| Risky | 95 | 30% |
| Distressed | 120 | 30% |

*Part 2, first passage.* Real-world default probabilities for the medium firm (asset drift 8%) under two default rules:
- **Merton:** default only if assets are below the debt at maturity.
- **Black-Cox:** default the first time assets touch the debt level, as a covenant would trigger.

The Black-Cox closed form is checked by Monte Carlo (100,000 paths, daily steps, continuity-corrected barrier).

*Part 3, estimation.* 300 simulated firms per scenario, each with assets of 100, debt of 60, asset volatility 25% and one year of daily equity prices computed from the model. Two estimators of asset volatility:
- **Static two-equation solve:** the year's historical equity volatility, plugged into E = C(V, σ) and σ_E E = N(d1) σ V.
- **KMV iteration:** invert every day's equity into assets with a trial σ, set σ to the volatility of that asset path, and repeat to a fixed point.

There are two scenarios, a normal year (asset drift +8%) and a year of falling assets (−30%). The estimators are scored on asset volatility and on the risk-neutral default probability they imply at the end of the year.

*Part 4, real equities with hypothetical debt.* The bundled data has equity prices but no balance sheets. For each of the six stocks, its last year of prices is scaled to an equity value of 100 and given a hypothetical one-year debt of 50, 100 and 200 (0.5, 1 and 2 times equity), with r = 4%. The KMV iteration then gives asset volatility, and distance to default with the asset drift set to r (one year of returns cannot pin down the real drift). This only shows how equity volatility and leverage translate into the model's outputs. **It is not a credit assessment of these companies.**

C++ only: every closed form, the asset solver, the KMV iteration and the Black-Cox formula match the Python reference in both test suites. Equity and debt also match Black-Scholes call and riskless-minus-put prices to rounding.

**Results**

Part 1: credit spread (basis points) by maturity:

| Maturity (years) | 0.1 | 0.5 | 1 | 2 | 5 | 10 | 30 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Safe (40, 20%) | 0.0 | 0.0 | 0.0 | 0.2 | 4.3 | 11.7 | 16.8 |
| Medium (70, 25%) | 0.0 | 27 | 80 | 128 | 145 | 127 | 81 |
| Risky (95, 30%) | 1,733 | 1,180 | 904 | 668 | 429 | 299 | 163 |
| Distressed (120, 30%) | 18,050 | 3,918 | 2,173 | 1,257 | 642 | 397 | 191 |

The risk-neutral default probability over 5 years was 1.5% for the safe firm, 27% for the medium firm, 51% for the risky firm and 65% for the distressed firm. Expected recovery given default for the medium firm was 84% of face value at 2 years.

Part 2: medium firm, real-world probability of default by each horizon:

| Horizon | Merton | Black-Cox | Ratio | Monte Carlo (± 1 s.e.) |
| :---- | :---- | :---- | :---- | :---- |
| 3 months | 0.16% | 0.33% | 2.06 | 0.32% ± 0.02% |
| 1 year | 5.2% | 11.5% | 2.19 | 11.5% ± 0.1% |
| 2 years | 9.9% | 23.2% | 2.34 | 23.3% ± 0.1% |
| 5 years | 14.1% | 38.2% | 2.70 | 38.3% ± 0.2% |
| 10 years | 14.3% | 46.7% | 3.27 | 46.8% ± 0.2% |

Part 3: 300 simulated firms per scenario (true asset volatility 25%):

| Scenario | Estimator | Asset vol bias | Asset vol RMSE | PD bias | PD RMSE | Mean true PD |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Normal year | Static two-equation | +0.4 pts | 5.5 pts | −2.2 pts | 6.6 pts | 6.0% |
| | KMV iteration | −0.03 pts | 1.2 pts | −0.06 pts | 0.9 pts | |
| Falling assets | Static two-equation | −8.0 pts | 10.5 pts | −15.3 pts | 20.9 pts | 32.7% |
| | KMV iteration | −0.1 pts | 1.4 pts | −0.2 pts | 1.8 pts | |

Part 4: hypothetical debt equal to market equity (1× leverage), one year:

| Stock | Equity vol | Implied asset vol | Distance to default |
| :---- | :---- | :---- | :---- |
| PG | 14.9% | 7.6% | 9.3 |
| JNJ | 16.5% | 8.5% | 8.4 |
| JPM | 20.7% | 9.4% | 7.5 |
| AAPL | 20.3% | 9.6% | 7.4 |
| CAT | 28.5% | 13.1% | 5.4 |
| XOM | 25.1% | 13.2% | 5.3 |

Even at 2× leverage, the lowest distance to default was 4.6 (XOM). Every implied one-year default probability was below 2 in a million, and every spread below 0.01 basis points.

**Discussion**  
**The model's spreads have the right shapes but vanish at short maturities, the credit spread puzzle.** Low leverage gives an upward-sloping curve, moderate leverage a hump and high leverage a downward slope. These are the three shapes Merton derived, and they reflect where default risk sits in time. A safe firm can only default if its assets fall a long way, which takes time. A firm close to its debt level is most at risk soon. But for any firm not already near default, the spread goes to zero at short maturities. The safe firm's spread is under 1 basis point out to 2 years, and the medium firm's is zero at 0.1 years (about five weeks). Under GBM, assets cannot fall far in a short time. Observed short-term spreads for investment-grade firms are clearly positive. The gap between Merton spreads and observed ones is a long-standing empirical finding (Jones, Mason and Rosenfeld 1984; Huang and Huang 2012). It is usually attributed to jumps (Phase 8's Merton jump-diffusion makes sudden default possible), uncertain asset values, and liquidity and tax components of spreads.

**When default can happen matters as much as whether it can.** Allowing default the first time assets touch the debt level doubled the 3-month default probability. The ratio grew to 3.3 at 10 years, because a path that dips below the debt and recovers counts as a default under Black-Cox but not under Merton. The Monte Carlo check agrees with the closed form within about 2 standard errors at every horizon, confirming the formula and its implementation. Merton's PD flattens out at about 14% beyond 5 years, while Black-Cox keeps rising towards the probability that assets ever touch the barrier.

**Equity prices pin down the firm's assets only with the right estimator.** In a normal year the static two-equation solve was unbiased on average but noisy: an RMSE of 5.5 volatility points and 6.6 points of default probability. The KMV iteration was four to seven times more accurate, essentially at the sampling limit of one year of data. When assets fell through the year, the static method failed badly. It understated asset volatility by 8 points and default probability by 15 points on average, against a true mean of 33%. The reason is that equity volatility rises as leverage rises. A historical equity volatility averages over a year in which the firm was less levered than it is now, so it understates today's risk exactly when it matters. The KMV iteration uses each day's own leverage and so stayed accurate (1.8 points RMSE on PD). Missing deteriorating credits in this way is the failure that makes the iteration, not the textbook two-equation solve, the practical standard.

**For large, liquid, low-volatility equities the model sees no near-term default at all.** With hypothetical debt equal to market equity, the six stocks' asset volatility was about half their equity volatility, and their distances to default ranged from 5.3 (XOM) to 9.3 (PG). That is far enough that the normal distribution assigns effectively zero one-year default probability and zero spread. This is the credit spread puzzle seen from the other side, and the reason KMV-style systems map distance to default to observed default frequencies empirically rather than through N(−DD). The ranking is still informative: by asset volatility, the commodity and industrial names carry about 60% more risk than the consumer staples (13.1 to 13.2% against 7.6 to 8.5%).

**Limitations.** The model has a single debt maturity, constant interest rates and volatility, and no jumps. Black-Cox uses a barrier equal to the debt face. The estimation study draws data from the model itself, so it measures estimator error, not model error. Part 4 uses hypothetical debt, and real distance-to-default work needs actual liabilities (KMV uses short-term debt plus half of long-term debt). The real-world default probabilities use an asset drift that cannot be estimated from one year of data: in Part 3's normal year its error dwarfed everything else, which is why the estimators are compared on the risk-neutral default probability.

**Conclusion**  
Merton's model turns option pricing into credit analysis. Equity is a call on the firm's assets, debt is riskless debt minus a put, and default probabilities, recoveries and spreads follow in closed form. Its spread curves have the right shapes but near-zero short-maturity spreads, the credit spread puzzle. Letting default happen at the first touch of the debt level roughly doubles to triples default probabilities. To use the model on real firms, asset volatility should come from the KMV iteration, not from a historical equity volatility. The textbook shortcut understated a deteriorating firm's default probability by 15 points on average; the iteration got within 2.

**File Paths**  
/cpp/experiments/credit/credit_experiment.cpp and csv with results (term_structure, first_passage, estimation, estimation_summary, hypothetical_leverage) in /cpp/experiments/credit/results/
