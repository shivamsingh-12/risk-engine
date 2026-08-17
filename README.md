**Qf-risk-engine (qre)**  
A quantitative risk engine, built with C++ 23 with Python reference implementations, meant to be a learning project that helps me learn math actively used in the financial sector, ranging from probability theory to stochastic volatility model.

**Status:** early work in progress. This is not a production system – it’s meant to be built from scratch where every mathematical component is derived, implemented, and then tested.

**What it actually is**  
This project differs from the quant projects I have seen by actively working in phases that are split in a specific order. Research on math topics and formulas (seen in notes/) and then a corresponding implementation in Python and C++, tested for correctness and then finally validated with a small experiment that attempts to ask a question about the method.  
The end goal is a system that can price derivatives, simulate portfolios, and estimate risk across multiple models including but not limited to analytical, Monte Carlo, and stochastic volatility.

A short roadmap is listed below:

| Phase | Focus | Status |
| :---- | :---- | :---- |
| 0 | Foundation \+ repository | Done |
| 1 | Probability \+ randomness | In progress |
| 2 | Calculus \+ linear algebra | Not started |
| 3 | Statistics \+ financial data | Not started |
| 4 | Numerical methods \+ optimization | Not started |
| 5 | Financial math \+ derivatives (Black-Scholes) | Not started |
| 6 | Monte Carlo | Not started |
| 7 | Portfolio simulation \+ risk (VaR/ES) | Not started |
| 8 | Stochastic processes \+ GBM | Not started |
| 9 | Stochastic volatility \+ Heston | Not started |
| 10 | High-dimensional risk \+ RMT | Not started |
| 11 | Tail risk \+ model uncertainty | Not started |
| 12 | Advanced Monte Carlo \+ AAD | Not started |
| 13 | Adaptive risk engine | Not started |
| 14 | Production quant research system | Not started |

Two changes to note:

* Data layer is a topic that has caught me attention that I didn’t account for initially but will be much more necessary to implement after phase 7 and before proceeding any further.  
* Backtesting and validation \- After data layer hardening \-\> very important for a risk engine to be validated against realized outcomes

**Why Python and C++?**  
A dual implementation approach for which Python accounts for algorithmic prototyping and closed-form validation, C++ 23 for production-grade performance. Every C++ module is cross validated against its Python counterpart before assumed correct, which isolates language-level bugs from mathematical errors.

**Notes**  
This is a personal learning project, evolving as I learn and understand more math and how to implement it to be an active and useful resource in my project. Structure, phases, files, may change as I move forward, docs/[progress.md](http://progress.md) has my notes on why I choose to write code the way I do.
