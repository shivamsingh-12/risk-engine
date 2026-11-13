"""Phase 18 reference: optimal execution, the Almgren-Chriss (2000) model.

Sell X shares over [0, T] in N periods of length tau = T/N. Holdings x_0 = X, ..., x_N = 0; trade n_k = x_{k-1} - x_k
in period k at rate v_k = n_k / tau. The mid price moves by volatility and by the trader's own permanent impact,
    S_k = S_{k-1} + sigma sqrt(tau) xi_k - tau g(v_k),          g(v) = gamma v   (permanent, linear)
and each trade is filled below the mid by a temporary impact,
    S~_k = S_{k-1} - h(v_k),                                    h(v) = eps sgn(v) + eta v   (temporary)
Implementation shortfall IS = X S_0 - sum n_k S~_k (the cost relative to selling everything at the arrival price):
    E[IS]   = gamma X^2 / 2 + eps sum |n_k| + (eta~ / tau) sum n_k^2,      eta~ = eta - gamma tau / 2
    Var[IS] = sigma^2 tau sum_{k=1}^{N} x_k^2      (x_k held over period k+1 ... ; with x_N = 0)
Selling fast lowers the risk (less inventory exposed to sigma) but costs more temporary impact; selling slowly does
the opposite. Minimizing E + lambda Var (lambda = risk aversion) gives, for a sell program (all n_k >= 0),
    x_j = X sinh(kappa (T - t_j)) / sinh(kappa T),   (2 / tau^2)(cosh(kappa tau) - 1) = lambda sigma^2 / eta~
lambda = 0 gives the straight line (TWAP); larger lambda front-loads the selling, with half-life about 1 / kappa.
The same optimum solves a tridiagonal linear system, which the tests use as an independent check.
"""
import math

import numpy as np


class Params:
    def __init__(self, X, T, N, sigma, eta, gamma, eps=0.0, S0=50.0):
        if not (X > 0 and T > 0 and N >= 1 and sigma >= 0 and eta > 0 and gamma >= 0 and eps >= 0 and S0 > 0):
            raise ValueError("need X, T, eta, S0 > 0, N >= 1 and sigma, gamma, eps >= 0")
        self.X, self.T, self.N, self.sigma, self.eta, self.gamma, self.eps, self.S0 = X, T, int(N), sigma, eta, gamma, eps, S0
        self.tau = T / N
        self.eta_tilde = eta - 0.5 * gamma * self.tau
        if self.eta_tilde <= 0:
            raise ValueError("need eta > gamma tau / 2 (otherwise splitting trades has no cost)")


def kappa(p, lam):
    if lam < 0:
        raise ValueError("risk aversion must be >= 0")
    k2 = lam * p.sigma ** 2 / p.eta_tilde
    return math.acosh(1.0 + 0.5 * k2 * p.tau ** 2) / p.tau


def optimal_holdings(p, lam):
    """x_0 .. x_N for the mean-variance optimal sell schedule."""
    t = p.tau * np.arange(p.N + 1)
    k = kappa(p, lam)
    if k * p.T < 1e-10:
        return p.X * (1.0 - t / p.T)                                  # lambda -> 0: TWAP
    if k * p.T > 700:                                                  # sinh overflow: ratio is exp(-kappa t)
        return p.X * np.exp(-k * t) * (1 - np.exp(-2 * k * (p.T - t))) / (1 - np.exp(-2 * k * p.T))
    return p.X * np.sinh(k * (p.T - t)) / math.sinh(k * p.T)


def twap_holdings(p):
    return p.X * (1.0 - np.arange(p.N + 1) / p.N)


def expected_cost(p, x):
    n = -np.diff(np.asarray(x, dtype=float))
    return 0.5 * p.gamma * p.X ** 2 + p.eps * np.sum(np.abs(n)) + p.eta_tilde / p.tau * np.sum(n * n)


def cost_variance(p, x):
    x = np.asarray(x, dtype=float)
    return p.sigma ** 2 * p.tau * np.sum(x[1:] ** 2)


def solve_numerically(p, lam):
    """Minimize E + lambda V over x_1..x_{N-1} directly (the eps term is constant for a monotone sell program).
    With a = 2 eta~ / tau the first-order conditions are the tridiagonal system
        a (2 x_j - x_{j-1} - x_{j+1}) + 2 lambda sigma^2 tau x_j = 0,   j = 1 .. N-1,   x_0 = X, x_N = 0."""
    m = p.N - 1
    x = np.zeros(p.N + 1)
    x[0] = p.X
    if m <= 0:
        return x
    a = 2 * p.eta_tilde / p.tau
    diag = 2 * a + 2 * lam * p.sigma ** 2 * p.tau
    A = np.diag(np.full(m, diag)) + np.diag(np.full(m - 1, -a), 1) + np.diag(np.full(m - 1, -a), -1)
    b = np.zeros(m)
    b[0] = a * p.X
    x[1:p.N] = np.linalg.solve(A, b)
    return x


def efficient_frontier(p, lambdas):
    """(lambda, E, sqrt(V)) for each risk aversion."""
    out = []
    for lam in lambdas:
        x = optimal_holdings(p, lam)
        out.append((lam, expected_cost(p, x), math.sqrt(cost_variance(p, x))))
    return out


def simulate_shortfall(p, x, n_paths=100_000, seed=0):
    """Monte Carlo implementation shortfall of the schedule x under the model (arithmetic Brownian price)."""
    x = np.asarray(x, dtype=float)
    n = -np.diff(x)
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal((n_paths, p.N))
    S_prev = np.full(n_paths, p.S0)
    proceeds = np.zeros(n_paths)
    for k in range(p.N):
        v = n[k] / p.tau
        proceeds += n[k] * (S_prev - (p.eps * np.sign(v) + p.eta * v))
        S_prev = S_prev + p.sigma * math.sqrt(p.tau) * xi[:, k] - p.tau * p.gamma * v
    return p.X * p.S0 - proceeds


def paper_example():
    """Almgren and Chriss (2000), section 3 example: 1 million shares at $50 over 5 days, 30% annual volatility,
    daily volume 5 million shares; eps = 1/16 dollar, eta = 2.5e-6, gamma = 2.5e-7, lambda = 1e-6."""
    return Params(X=1e6, T=5.0, N=5, sigma=0.95, eta=2.5e-6, gamma=2.5e-7, eps=0.0625, S0=50.0)
