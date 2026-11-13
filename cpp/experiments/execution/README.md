**Experiment 23 (Phase 18): Optimal Execution — Trading Off Market Impact Against Price Risk**

**Objective**  
Every risk number in this engine assumes a position can be bought or sold at the market price. A large order cannot be. Selling it quickly pushes the price down (market impact), while selling it slowly leaves the unsold shares exposed to the market moving against them. Almgren and Chriss (2000) turned this into a mean-variance problem with a closed-form answer. This experiment reproduces their example and checks the closed form against a direct numerical solution and against simulation. It then looks at what risk aversion buys and costs, how the available time horizon matters, and how the schedule depends on volatility.

**Methodology**  
The model sells X shares in N equal periods.
- **Temporary impact:** each trade is filled below the current price by ε + η × (trading rate).
- **Permanent impact:** each trade also moves the price down permanently by γ × (trading rate) × (period length).
- **Price risk:** the price itself moves as a random walk with volatility σ.

The cost is the implementation shortfall: the arrival value X·S₀ minus what the sale actually raised. Its mean E and variance V have closed forms. Minimizing E + λV gives holdings that decay like sinh(κ(T − t)) / sinh(κT), where κ grows with risk aversion λ and volatility and shrinks with the impact cost η. λ = 0 gives TWAP, a straight line.

*Part 1.* The paper's example: sell 1 million shares at $50 over 5 days (one period per day), 30% annual volatility (σ = $0.95 per share per day), ε = $0.0625, η = 2.5×10⁻⁶ and γ = 2.5×10⁻⁷, with λ = 0, 10⁻⁷, 10⁻⁶ and 10⁻⁵. Every schedule is also computed by solving the optimality conditions directly as a tridiagonal linear system.

*Part 2.* The efficient frontier of E against √V for λ from 10⁻⁸ to 10⁻⁴.

*Part 3.* 200,000 simulated executions of three strategies: TWAP, the optimal λ = 10⁻⁶ schedule, and selling everything in the first period. Their mean, standard deviation and 95% VaR and CVaR are compared with the formulas.

*Part 4.* The same order with λ = 10⁻⁶ and 1 to 20 days available, at five trading periods per day. The periods are finer than in Part 1, so the 5-day numbers differ slightly from Part 1's.

*Part 5.* The paper's impact parameters combined with each bundled stock's daily return volatility (last 250 days), applied to a $50 price. **The bundled data has no volumes or trades, so impact cannot be estimated from it. These are illustrations of how volatility changes the schedule, not cost estimates for these stocks.**

C++ only: schedules, costs, variances and κ match the Python reference to rounding in both test suites. Both languages also check the closed form against the numerical solve, and the formulas against Monte Carlo.

**Results**

Part 1: schedules for the paper's example (percentage of the order sold on day 1; costs also in basis points of the $50 million order):

| Risk aversion λ | κ (per day) | Sold on day 1 | Expected cost | Std. deviation |
| :---- | :---- | :---- | :---- | :---- |
| 0 (TWAP) | 0 | 20% | $662,500 (133 bp) | $1,040,673 (208 bp) |
| 10⁻⁷ | 0.19 | 24% | $670,057 (134 bp) | $961,623 (192 bp) |
| 10⁻⁶ | 0.61 | 46% | $911,227 (182 bp) | $603,431 (121 bp) |
| 10⁻⁵ | 1.73 | 82% | $1,845,211 (369 bp) | $171,712 (34 bp) |

The closed form and the direct linear solve agreed to within 5×10⁻¹⁰ shares for every λ in Parts 1 and 2.

Part 2: along the frontier, raising λ from 10⁻⁸ to 10⁻⁷ cut the standard deviation by 7% ($1.03M to $0.96M) and raised the expected cost by 1.1%. From 10⁻⁵ to 10⁻⁴, the standard deviation fell by a further $148,000 while the expected cost rose by $601,000.

Part 3: 200,000 simulated executions:

| Strategy | Mean (formula) | Mean (simulated) | Std. dev. (formula / simulated) | 95% VaR | 95% CVaR |
| :---- | :---- | :---- | :---- | :---- | :---- |
| TWAP | $662,500 | $659,987 | $1.041M / $1.042M | $2.37M | $2.80M |
| Optimal, λ = 10⁻⁶ | $911,227 | $909,849 | $0.603M / $0.604M | $1.90M | $2.15M |
| Sell all at once | $2,562,500 | $2,562,500 | 0 / 0 | $2.56M | $2.56M |

Part 4: λ = 10⁻⁶, with more time available:

| Days available | Expected cost | Std. deviation | E + λV | Sold on day 1 | Still held at the midpoint |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1 | $2.67M | $0.45M | $2.87M | 100% | 58% |
| 2 | $1.47M | $0.65M | $1.89M | 58% | 42% |
| 5 | $0.96M | $0.80M | $1.60M | 46% | 23% |
| 10 | $0.93M | $0.81M | $1.595M | 45% | 4.9% |
| 20 | $0.93M | $0.81M | $1.595M | 45% | 0.2% |

Part 5: λ = 10⁻⁶ with each stock's volatility (illustrative impact parameters):

| Stock | Daily vol | Sold on day 1 | Expected cost | Std. dev. (optimal) | Std. dev. (TWAP) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| PG | 0.94% | 29% | 140 bp | 86 bp | 103 bp |
| JNJ | 1.04% | 31% | 142 bp | 92 bp | 114 bp |
| AAPL | 1.28% | 35% | 151 bp | 104 bp | 140 bp |
| JPM | 1.31% | 36% | 152 bp | 105 bp | 143 bp |
| XOM | 1.58% | 40% | 165 bp | 114 bp | 173 bp |
| CAT | 1.79% | 44% | 176 bp | 119 bp | 197 bp |

**Discussion**  
**The first reduction in risk is almost free.** At the TWAP end the frontier is flat: TWAP minimizes expected cost, so moving away from it costs nothing to first order. Going from λ = 10⁻⁸ to 10⁻⁷ cut the cost's standard deviation by 7% for a 1.1% higher expected cost. At the other end, each further dollar of risk reduction costs about four dollars of expected impact. Any risk-averse trader should therefore deviate at least a little from TWAP, and the right amount depends on λ, which is a preference rather than something the model can estimate.

**Risk aversion front-loads the selling.** With λ = 10⁻⁶ the trader sells 46% of the order on day 1 instead of TWAP's 20%. The holdings decay with a time scale of 1/κ = 1.6 days. That raises the expected cost from 133 to 182 basis points and cuts its standard deviation from 208 to 121 basis points. In the simulated tail the optimal schedule is clearly better: its 95% CVaR ($2.15M) is 23% below TWAP's ($2.80M) and 16% below selling everything at once ($2.56M). Selling everything at once has no price risk at all, but its certain impact cost exceeds what either strategy loses in its bad cases. The simulations reproduced the formulas: means within about one standard error, and standard deviations within 0.2%.

**The optimal trader does not use all the time it is given.** With 10 or 20 days available, the risk-averse schedule still sells 45% on day 1. It holds less than 5% of the order at the midpoint of a 10-day horizon and 0.2% at the midpoint of a 20-day one, and the objective barely changes beyond 10 days ($1,594,542 at 10 days against $1,594,525 at 20). Once the horizon is several times 1/κ, the price risk of holding inventory outweighs the impact saved by trading more slowly. The opposite constraint is expensive: forcing completion within one day costs $2.67M, almost three times the 5-day optimum.

**Volatility sets the urgency.** With the same impact costs, the most volatile stock (CAT, 1.79% a day) was sold 44% on day 1 and the least volatile (PG, 0.94%) 29%. The benefit of optimizing grows with volatility. For PG, the optimal schedule's standard deviation is 17% below TWAP's; for CAT it is 40% below. In practice volatility can be measured well (Phases 7 to 13), while the impact parameters η and γ need trade-level data this engine does not have. The schedule depends on both through κ ≈ √(λσ²/η).

**Limitations.** Impact is linear and permanent impact is constant, whereas empirical studies find temporary impact closer to a square root of trading rate. The price follows an arithmetic random walk with constant volatility, and the strategy is static: it is fixed in advance and does not react to prices during execution. The paper's parameters are its own illustration, not estimates. Part 5 combines real volatilities with those illustrative impact parameters.

**Conclusion**  
Almgren-Chriss reduces optimal execution to one number, κ, which combines risk aversion, volatility and impact cost. The closed-form schedule matched the numerical optimum to rounding and the simulations matched its mean and variance. The trade-off it describes is steep at both ends: the first reduction in risk away from TWAP is nearly free, and the last is very expensive. A risk-averse seller front-loads the order and finishes within a few multiples of 1/κ however long it is allowed. Volatility, which this engine measures well, drives how urgent that is.

**File Paths**  
/cpp/experiments/execution/execution_experiment.cpp and csv with results (schedules, schedule_summary, frontier, monte_carlo, horizon, volatility_sensitivity) in /cpp/experiments/execution/results/
