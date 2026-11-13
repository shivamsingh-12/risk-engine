**Experiment 22 (Phase 17): Deep Hedging — Can a Neural Network Hedge Better Than Black-Scholes When Its Assumptions Fail?**

**Objective**  
Black-Scholes delta hedging replicates an option exactly only under three conditions: trading is continuous, trading is free, and volatility is known and constant. None of them holds in practice. Deep hedging (Buehler, Gonon, Teichmann and Wood 2019) drops the search for a formula. A neural network chooses the hedge at each date and is trained on simulated paths to minimize a risk measure of the final hedging loss. This experiment tests whether that learned strategy beats Black-Scholes delta when each assumption is relaxed in turn, and looks at what the network learns.

**Methodology**  
We are short one at-the-money call (S = K = 100, 30 trading days, zero interest rates) and hedge with the stock on each of the 30 days.
- **Loss:** the call's payoff, minus the hedge's trading gains, plus proportional trading costs. The position is unwound at expiry.
- **Objective:** CVaR at 95%, the mean of the worst 5% of losses.
- **Network:** the hedge on day k is a network's output given the time, ln(S/K)/√T, the previous day's position and (under Heston) the current volatility. It has two layers of 16 tanh units and is shared across all dates.
- **Training:** gradients by hand-written backpropagation through time, since today's position is an input to tomorrow's decision. 3,000 Adam steps on fresh batches of 2,048 paths, learning rate decaying from 0.005 to 0.00025. Each case took about 4.5 minutes on the 2-core build machine.
- **Evaluation:** on 100,000 independent test paths.

Four cases:
- **A:** geometric Brownian motion (GBM), 20% volatility, no costs. The only flaw in Black-Scholes is that hedging is daily, not continuous.
- **B and C:** GBM with proportional costs of 0.2% and 1% of the value traded.
- **D:** Heston stochastic volatility (v₀ = θ = 0.04, κ = 2, vol of vol 0.6, ρ = −0.7), no costs. The network sees the current variance. It is compared with Black-Scholes delta at the current volatility √vₖ and at a fixed 20%.

The no-trade band is read off the trained network at the middle date: for each moneyness, the range of previous positions for which the network changes the position by less than 0.02.

The C++ forward pass, loss and gradients match the Python reference to rounding (better than 1e-12) on fixed weights and paths. Both languages check the gradients against finite differences and check that a short training run cuts CVaR to within 15% of Black-Scholes.

**Results**

The Black-Scholes price of the call is 2.75. Losses are per call, in the same units as the price:

| Case | Strategy | Mean loss | Std. dev. | 95% VaR | 95% CVaR | Turnover | Costs paid |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| A: GBM, no costs | Deep hedge | 2.754 | 0.53 | 3.50 | **3.69** | 2.59 | 0 |
| | Black-Scholes delta | 2.756 | 0.43 | 3.47 | 3.76 | 2.72 | 0 |
| | No hedge | 2.77 | 4.19 | 11.74 | 15.05 | 0 | 0 |
| B: GBM, 0.2% costs | Deep hedge | 3.22 | 0.56 | 4.01 | **4.21** | 2.31 | 0.47 |
| | Black-Scholes delta | 3.30 | 0.48 | 4.15 | 4.49 | 2.71 | 0.55 |
| C: GBM, 1% costs | Deep hedge | 4.69 | 0.73 | 5.77 | **6.04** | 1.92 | 1.94 |
| | Black-Scholes delta | 5.49 | 0.94 | 7.13 | 7.63 | 2.71 | 2.74 |
| D: Heston, no costs | Deep hedge (sees variance) | 2.64 | 0.79 | 3.84 | **4.23** | 2.32 | 0 |
| | Black-Scholes at current vol | 2.64 | 0.78 | 4.04 | 4.45 | 2.74 | 0 |
| | Black-Scholes at 20% | 2.64 | 0.77 | 3.95 | 4.57 | 2.67 | 0 |
| | No hedge | 2.63 | 3.35 | 9.34 | 11.40 | 0 | 0 |

Turnover is the total number of shares traded per call. With no hedge, the costs are zero in Cases B and C too, so those rows match Case A's.

CVaR improvement of the deep hedge over Black-Scholes delta: 1.9% in A, 6.1% in B, 20.8% in C, 4.9% in D (7.3% against Black-Scholes at 20%).

How far the network's hedge is from Black-Scholes delta (mean absolute difference over test paths):

| Case | Day 0 | Day 10 | Day 20 | Day 29 |
| :---- | :---- | :---- | :---- | :---- |
| A: no costs | 0.007 | 0.031 | 0.032 | 0.087 |
| C: 1% costs | 0.005 | 0.090 | 0.088 | 0.116 |

No-trade band at day 15 (range of previous positions the network leaves within 0.02):

| Moneyness S/K | Black-Scholes delta | A: no costs | B: 0.2% | C: 1% |
| :---- | :---- | :---- | :---- | :---- |
| 0.94 | 0.11 | 0.14 to 0.17 | 0.17 to 0.20 | 0.23 to 0.28 |
| 0.98 | 0.35 | 0.36 to 0.39 | 0.39 to 0.42 | 0.46 to 0.51 |
| 1.00 | 0.51 | 0.49 to 0.52 | 0.51 to 0.55 | 0.56 to 0.61 |
| 1.02 | 0.67 | 0.62 to 0.65 | 0.63 to 0.67 | 0.65 to 0.71 |
| 1.06 | 0.89 | 0.84 to 0.86 | 0.82 to 0.85 | 0.82 to 0.89 |

**Discussion**  
**Without frictions the network rediscovers delta hedging, tuned to the tail.** With GBM and no costs, the network's hedge was within 0.007 of Black-Scholes delta on day 0 and about 0.03 through most of the option's life. Its CVaR was 1.9% lower, but its standard deviation was 22% higher (0.53 against 0.43). That is not a training failure. Black-Scholes delta minimizes variance in the continuous limit, while the network was asked to minimize the average of the worst 5% of losses, and with daily trading these are different targets. It over-hedges out-of-the-money (0.14 to 0.17 instead of 0.11 at 94% moneyness) and under-hedges in-the-money. This gives up some precision in ordinary outcomes to cut the losses that come from sharp rallies against the short call. The two strategies differ most near expiry (0.087 on the last day), where the Black-Scholes delta becomes almost a step function and the network's smooth policy cannot follow it.

**With trading costs the network learns to trade less, and the gain grows with the cost.** At 0.2% costs the deep hedge cut CVaR by 6%. At 1% it cut it by 21%: its CVaR was 6.04 against 7.63, and its standard deviation was also lower. Black-Scholes delta ignores costs and keeps trading as if they were zero, with a turnover of 2.71 shares per call whatever the cost. The network traded 15% less at 0.2% costs and 29% less at 1%, and paid 29% less in costs in the 1% case (1.94 against 2.74). It does this in two ways. First, it pulls its positions towards the middle (0.23 to 0.28 at 94% moneyness instead of 0.11), so it has less to buy back if the price returns. Second, it ignores small moves: the range of previous positions it leaves alone widens from the measurement tolerance alone (about 0.03 to 0.04 wide without costs) to 0.05 to 0.07 at 1% costs. Asymptotic results for hedging with costs (Whalley and Wilmott 1997) have the band widening with the cube root of the cost. How close the learned band is to that optimum was not tested here, nor whether a larger network or longer training would widen it.

**Under stochastic volatility, knowing the volatility is not enough; using it well helps.** In the Heston market, Black-Scholes delta at the current volatility improved CVaR over a fixed 20% (4.45 against 4.57). But it still ignores that volatility will move, and that it moves against the hedger: with ρ = −0.7, volatility rises when the price falls. The network, given the same current variance, did better again (4.23, 4.9% below Black-Scholes at the current volatility). All three strategies had essentially the same mean and standard deviation. The network's gain is concentrated in the tail it was trained on.

**Limitations.** A single at-the-money call and a single maturity, zero interest rates, and daily hedging with the underlying only (no options as hedging instruments, which is how volatility risk is hedged in practice). Training is noisy: the batch objective still fluctuates by about ±0.05 at the end, and the networks are small (2 × 16 units) to keep training to minutes. Results reproduce exactly on the same machine and build. Training is a long chain of floating-point steps, so other CPUs or compilers can end at slightly different networks. The no-trade band is measured with a fixed tolerance and gives only a rough width.

**Conclusion**  
A small network trained only to minimize the tail of hedging losses learned a strategy close to Black-Scholes delta when Black-Scholes is nearly right, and beat it where its assumptions fail. The improvement grew from 2% (daily trading only) to 6% (0.2% costs) to 21% (1% costs), and was 5% under stochastic volatility. It did this without a formula, by trading less, avoiding round trips and reacting to volatility. The method's strength is its generality: the same code hedges any payoff in any simulated market with any risk measure. That generality costs training time, noise and opacity. Black-Scholes is still the benchmark that makes the learned strategy interpretable.

**File Paths**  
/cpp/experiments/deep-hedging/deep_hedging_experiment.cpp and csv with results (hedging_performance, delta_comparison, no_trade_band, training_curves) in /cpp/experiments/deep-hedging/results/
