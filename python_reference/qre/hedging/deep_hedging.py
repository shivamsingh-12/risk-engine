"""Phase 17 reference: deep hedging (Buehler, Gonon, Teichmann and Wood 2019).

Problem. We are short one call (strike K, maturity T) and may trade the underlying at n dates t_0 < ... < t_{n-1}.
A hedging strategy holds delta_k shares over (t_k, t_{k+1}]. Trading costs c * S_k per share bought or sold
(proportional costs), and the position is unwound at T. The loss of the hedged short call (zero rates) is
    L = max(S_T - K, 0) - sum_k delta_k (S_{k+1} - S_k) + c sum_{k=0}^{n} S_k |delta_k - delta_{k-1}|
with delta_{-1} = delta_n = 0. Black-Scholes says: with continuous trading, no costs and known volatility,
delta_k = N(d1) makes L = BS price (constant). Discrete dates, costs and unknown volatility break that, and there is
no formula for the best strategy any more.

Deep hedging. Let a neural network choose delta_k from what is known at t_k,
    delta_k = f_theta(t_k / T, ln(S_k / K) / sqrt(T), delta_{k-1} [, sqrt(v_k) under stochastic volatility])
(one network shared by all dates; moneyness is scaled by sqrt(T) so inputs are of order one; the previous
position is an input because costs make it matter), and choose
theta to minimize a risk measure of L over simulated paths. Here the risk measure is CVaR at level alpha, the mean
of the worst (1 - alpha) share of losses, in the Rockafellar-Uryasev form
    J(theta, w) = w + E[(L - w)^+] / (1 - alpha),      min over w gives CVaR_alpha(L)
For fixed theta the best w is the alpha-quantile (VaR) of L, so training sets w to each batch's empirical
alpha-quantile and moves theta along the gradient of J at that w (by the envelope theorem this is the gradient of
CVaR itself). The minimized J is the price at which the hedger is indifferent to selling the call under this risk
measure.

Network: two hidden layers of tanh units and a linear output, weights W1 (n_in x H), W2 (H x H), W3 (H x 1).
Gradients by hand-written backpropagation through time: delta_k affects the loss directly (gains and costs at k
and k+1) and through the network's input at k+1, so the backward pass runs from the last date to the first,
carrying the gradient with respect to delta_k into date k's network. Adam updates the parameters.
The C++ version (qre/hedging/deep_hedging.hpp) implements the same forward and backward pass and is checked against
this one to rounding.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------- markets
def simulate_gbm(n_paths, n_steps, s0, mu, sigma, T, seed=0):
    """Exact GBM at n_steps + 1 equally spaced dates; returns S (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    logs = np.cumsum((mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), logs]))


def simulate_heston(n_paths, n_steps, s0, mu, v0, kappa, theta, xi, rho, T, seed=0, substeps=4):
    """Full-truncation Euler (Phase 9) with `substeps` per hedging interval; returns S, v at the hedging dates."""
    rng = np.random.default_rng(seed)
    m = n_steps * substeps
    dt = T / m
    s = np.full(n_paths, float(np.log(s0)))
    v = np.full(n_paths, float(v0))
    S = np.empty((n_paths, n_steps + 1))
    V = np.empty((n_paths, n_steps + 1))
    S[:, 0], V[:, 0] = s0, v0
    for i in range(m):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        s += (mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        if (i + 1) % substeps == 0:
            k = (i + 1) // substeps
            S[:, k] = np.exp(s)
            V[:, k] = np.maximum(v, 0.0)
    return S, V


def bs_call(s, k, sigma, tau):
    if tau <= 0:
        return max(s - k, 0.0)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))
    return s * 0.5 * math.erfc(-d1 / SQRT2) - k * 0.5 * math.erfc(-(d1 - sigma * math.sqrt(tau)) / SQRT2)


def bs_delta(S, K, sigma, tau):
    """Black-Scholes call delta N(d1), zero rates; sigma may be an array (one per path)."""
    d1 = (np.log(S / K) + 0.5 * sigma * sigma * tau) / (sigma * math.sqrt(tau))
    return 0.5 * erfc(-d1 / SQRT2)


# ---------------------------------------------------------------------------- losses
def hedge_loss(S, deltas, K, cost):
    """Loss of the short call for given positions deltas (n_paths, n_steps)."""
    n = deltas.shape[1]
    gains = np.sum(deltas * np.diff(S, axis=1), axis=1)
    pos = np.hstack([np.zeros((S.shape[0], 1)), deltas, np.zeros((S.shape[0], 1))])
    costs = cost * np.sum(S[:, :n + 1] * np.abs(np.diff(pos, axis=1)), axis=1)
    return np.maximum(S[:, -1] - K, 0.0) - gains + costs


def bs_hedge_deltas(S, K, T, sigma, v=None):
    """Black-Scholes deltas at the hedging dates: constant sigma, or sqrt(v_k) per path if v is given."""
    n = S.shape[1] - 1
    out = np.empty((S.shape[0], n))
    for k in range(n):
        tau = T * (n - k) / n
        sig = np.sqrt(np.maximum(v[:, k], 1e-8)) if v is not None else sigma
        out[:, k] = bs_delta(S[:, k], K, sig, tau)
    return out


def cvar(L, alpha):
    """Empirical CVaR: mean of the worst (1 - alpha) share of losses (Rockafellar-Uryasev minimum)."""
    L = np.sort(np.asarray(L, dtype=float))
    w = np.quantile(L, alpha, method="inverted_cdf")
    return float(w + np.mean(np.maximum(L - w, 0.0)) / (1 - alpha))


# ---------------------------------------------------------------------------- network
@dataclass
class Policy:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    w: float = 0.0                          # CVaR threshold, trained with the network

    NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")

    @property
    def n_in(self):
        return self.W1.shape[0]

    @staticmethod
    def init(n_in, hidden, seed=0, w0=0.0):
        """Glorot-uniform weights, zero biases; output bias 0.5 (a reasonable starting delta)."""
        rng = np.random.default_rng(seed)
        g = lambda a, b: rng.uniform(-1, 1, (a, b)) * math.sqrt(6.0 / (a + b))
        return Policy(g(n_in, hidden), np.zeros(hidden), g(hidden, hidden), np.zeros(hidden),
                      g(hidden, 1), np.array([0.5]), w0)

    def copy(self):
        return Policy(*(getattr(self, n).copy() for n in self.NAMES), self.w)


def _features(S, v, K, T, k, prev):
    n = S.shape[1] - 1
    cols = [np.full(S.shape[0], k / n), np.log(S[:, k] / K) / math.sqrt(T), prev]
    if v is not None:
        cols.append(np.sqrt(v[:, k]))
    return np.column_stack(cols)


def policy_deltas(p, S, K, T, v=None):
    """Positions chosen by the network along each path (n_paths, n_steps)."""
    _check(p, S, v)
    n = S.shape[1] - 1
    prev = np.zeros(S.shape[0])
    out = np.empty((S.shape[0], n))
    for k in range(n):
        x = _features(S, v, K, T, k, prev)
        h1 = np.tanh(x @ p.W1 + p.b1)
        h2 = np.tanh(h1 @ p.W2 + p.b2)
        prev = (h2 @ p.W3 + p.b3)[:, 0]
        out[:, k] = prev
    return out


def _check(p, S, v):
    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError("need at least one hedging date")
    if p.n_in != (4 if v is not None else 3):
        raise ValueError("network inputs must be 3 (no variance) or 4 (with variance)")
    if v is not None and v.shape != S.shape:
        raise ValueError("v must have the shape of S")


def objective_and_grad(p, S, K, T, cost, alpha, v=None):
    """J = w + mean((L - w)^+)/(1 - alpha) on these paths, and its gradient {name: array, 'w': float}."""
    _check(p, S, v)
    if not (0 < alpha < 1) or cost < 0:
        raise ValueError("need 0 < alpha < 1 and cost >= 0")
    n_paths, n = S.shape[0], S.shape[1] - 1
    xs, h1s, h2s, ds = [], [], [], []
    prev = np.zeros(n_paths)
    for k in range(n):
        x = _features(S, v, K, T, k, prev)
        h1 = np.tanh(x @ p.W1 + p.b1)
        h2 = np.tanh(h1 @ p.W2 + p.b2)
        prev = (h2 @ p.W3 + p.b3)[:, 0]
        xs.append(x); h1s.append(h1); h2s.append(h2); ds.append(prev)
    deltas = np.column_stack(ds)
    L = hedge_loss(S, deltas, K, cost)
    excess = L - p.w
    J = p.w + np.mean(np.maximum(excess, 0.0)) / (1 - alpha)
    gL = (excess > 0).astype(float) / ((1 - alpha) * n_paths)
    grad = {name: np.zeros_like(getattr(p, name)) for name in Policy.NAMES}
    grad["w"] = 1.0 - np.mean(excess > 0) / (1 - alpha)

    pos = np.hstack([np.zeros((n_paths, 1)), deltas, np.zeros((n_paths, 1))])
    sgn = np.sign(np.diff(pos, axis=1))                      # sgn[:, k] = sign(delta_k - delta_{k-1}), k = 0..n
    dS = np.diff(S, axis=1)
    carry = np.zeros(n_paths)                                # dJ/d delta_k through the network at date k + 1
    for k in range(n - 1, -1, -1):
        direct = -dS[:, k] + cost * (S[:, k] * sgn[:, k] - S[:, k + 1] * sgn[:, k + 1])
        g_out = gL * direct + carry                          # dJ / d delta_k
        h1, h2, x = h1s[k], h2s[k], xs[k]
        grad["W3"] += h2.T @ g_out[:, None]
        grad["b3"] += np.array([g_out.sum()])
        g2 = (g_out[:, None] @ p.W3.T) * (1 - h2 * h2)
        grad["W2"] += h1.T @ g2
        grad["b2"] += g2.sum(axis=0)
        g1 = (g2 @ p.W2.T) * (1 - h1 * h1)
        grad["W1"] += x.T @ g1
        grad["b1"] += g1.sum(axis=0)
        carry = (g1 @ p.W1.T)[:, 2]                          # input 2 of date k is delta_{k-1}
    return float(J), grad, L


@dataclass
class TrainLog:
    objective: list = field(default_factory=list)


def train(p, sampler, K, T, cost, alpha, iterations=2000, batch=4096, lr=1e-3, seed=0, lr_final_fraction=1.0,
          beta1=0.9, beta2=0.999, eps=1e-8, log_every=100):
    """Adam on fresh simulated batches; w is reset to each batch's alpha-quantile of losses before the gradient step.
    sampler(n, rng) -> (S, v or None). Returns the trained copy and a log of (iteration, J)."""
    p = p.copy()
    rng = np.random.default_rng(seed)
    m = {n: np.zeros_like(getattr(p, n)) for n in Policy.NAMES}
    s = {n: np.zeros_like(getattr(p, n)) for n in Policy.NAMES}
    log = TrainLog()
    for it in range(1, iterations + 1):
        S, v = sampler(batch, rng)
        p.w = var_threshold(p, S, K, T, cost, alpha, v)
        J, g, _ = objective_and_grad(p, S, K, T, cost, alpha, v)
        c1, c2 = 1 - beta1 ** it, 1 - beta2 ** it
        frac = (it - 1) / (iterations - 1) if iterations > 1 else 0.0
        step = lr * (1.0 - frac * (1.0 - lr_final_fraction))          # linear decay to lr * lr_final_fraction
        for n in Policy.NAMES:
            m[n] = beta1 * m[n] + (1 - beta1) * g[n]
            s[n] = beta2 * s[n] + (1 - beta2) * g[n] * g[n]
            setattr(p, n, getattr(p, n) - step * (m[n] / c1) / (np.sqrt(s[n] / c2) + eps))
        if it % log_every == 0 or it == 1:
            log.objective.append((it, J))
    return p, log


def var_threshold(p, S, K, T, cost, alpha, v=None):
    """alpha-quantile (lower, 'inverted cdf') of the policy's losses on these paths: the minimizing w."""
    L = hedge_loss(S, policy_deltas(p, S, K, T, v), K, cost)
    return float(np.quantile(L, alpha, method="inverted_cdf"))
