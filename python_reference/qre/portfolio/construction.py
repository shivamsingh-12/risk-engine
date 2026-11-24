"""Portfolio construction reference (Phase 21): weights from a covariance matrix (and expected returns).

Every method returns fully invested weights (sum to 1) within per-asset bounds lo <= w <= hi. Negative lower
bounds allow short positions (lo = -0.3 allows a 30% short in each asset); hi > 1 allows leverage in one asset.

    equal_weight      1/N
    inverse_vol       w_i proportional to 1/sigma_i (ignores correlations)
    min_variance      minimize w' S w
    mean_variance     maximize mu' w - (gamma/2) w' S w; gamma is relative risk aversion, and the optimum is the same
                      for daily or annual mu and S as long as both use the same unit
    risk_parity       equal risk contributions: w_i (S w)_i equal for every i (long-only, needs no bounds)

The quadratic programs min 0.5 w'Qw - c'w s.t. sum w = 1, lo <= w <= hi are solved by projected gradient descent
with Nesterov acceleration and adaptive restart (FISTA, O'Donoghue and Candes 2015). Each step projects onto the
constraint set exactly: clip(v - tau, lo, hi) summing to 1, with tau found by bisection. The objective is convex,
and strictly convex when S is positive definite, so the optimum is unique and every correct solver finds it.

Risk parity uses the cyclical coordinate descent of Griveau-Billion, Richard and Roncalli (2013) on Spinu's convex
formulation min 0.5 x'Sx - sum b_i ln x_i, whose minimizer rescaled to sum 1 has risk contributions in proportion
to the budgets b.

Matches cpp/include/qre/portfolio/construction.hpp.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class QpResult:
    weights: np.ndarray
    iterations: int
    converged: bool


def _check_cov(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.all(np.isfinite(cov)) or np.abs(cov - cov.T).max() > 1e-12 * max(1.0, np.abs(cov).max()):
        raise ValueError("covariance must be finite and symmetric")
    return cov


def bounds(n, lo=0.0, hi=1.0):
    """Per-asset bounds from scalars or arrays; checks that a fully invested portfolio exists."""
    lo = np.broadcast_to(np.asarray(lo, dtype=float), (n,)).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=float), (n,)).copy()
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(lo > hi):
        raise ValueError("bounds must be finite with lo <= hi")
    if lo.sum() > 1.0 + 1e-12 or hi.sum() < 1.0 - 1e-12:
        raise ValueError("no fully invested portfolio fits these bounds (need sum(lo) <= 1 <= sum(hi))")
    return lo, hi


def project(v, lo, hi, total=1.0):
    """Euclidean projection of v onto {w : sum w = total, lo <= w <= hi}: w = clip(v - tau, lo, hi).

    sum clip(v - tau, lo, hi) falls as tau rises, so tau is found by bisection between a value where every asset is
    at its upper bound and one where every asset is at its lower bound."""
    v = np.asarray(v, dtype=float)
    a = float(np.min(v - hi))              # tau <= a: every w_i = hi_i
    b = float(np.max(v - lo))              # tau >= b: every w_i = lo_i
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mid <= a or mid >= b:
            break
        if np.clip(v - mid, lo, hi).sum() > total:
            a = mid
        else:
            b = mid
    w = np.clip(v - 0.5 * (a + b), lo, hi)
    # remove the last rounding error from the free assets so the weights sum to total exactly
    free = (w > lo) & (w < hi)
    if free.any():
        w[free] += (total - w.sum()) / free.sum()
    return w


def solve_qp(q, c, lo, hi, tol=1e-12, max_iter=200_000):
    """min 0.5 w'Qw - c'w  s.t.  sum w = 1, lo <= w <= hi (Q positive semidefinite)."""
    q = _check_cov(q)
    n = q.shape[0]
    c = np.asarray(c, dtype=float)
    if c.shape != (n,):
        raise ValueError("c must have one entry per asset")
    lmax = float(np.linalg.eigvalsh(q)[-1])
    if lmax <= 0.0:
        raise ValueError("Q has no positive eigenvalue")
    step = 1.0 / lmax
    x = project(np.full(n, 1.0 / n), lo, hi)
    y, t = x.copy(), 1.0
    scale = max(1.0, float(np.abs(x).max()))
    for k in range(1, max_iter + 1):
        x_new = project(y - step * (q @ y - c), lo, hi)
        if float(np.abs(x_new - x).max()) <= tol * scale:
            return QpResult(x_new, k, True)
        if (y - x_new) @ (x_new - x) > 0.0:      # adaptive restart: momentum points uphill
            t = 1.0
            y = x_new.copy()
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new
    return QpResult(x, max_iter, False)


def equal_weight(n):
    if n < 1:
        raise ValueError("need at least one asset")
    return np.full(n, 1.0 / n)


def inverse_vol(cov):
    cov = _check_cov(cov)
    vol = np.sqrt(np.diag(cov))
    if np.any(vol <= 0):
        raise ValueError("every asset needs a positive variance")
    return (1.0 / vol) / (1.0 / vol).sum()


def min_variance(cov, lo=0.0, hi=1.0):
    cov = _check_cov(cov)
    lo, hi = bounds(cov.shape[0], lo, hi)
    return solve_qp(cov, np.zeros(cov.shape[0]), lo, hi)


def mean_variance(mu, cov, gamma, lo=0.0, hi=1.0):
    cov = _check_cov(cov)
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (cov.shape[0],) or not np.all(np.isfinite(mu)):
        raise ValueError("one finite expected return per asset")
    if not gamma > 0:
        raise ValueError("risk aversion gamma must be positive")
    lo, hi = bounds(cov.shape[0], lo, hi)
    return solve_qp(gamma * cov, mu, lo, hi)


def risk_contribution_shares(w, cov):
    """w_i (S w)_i / w'Sw: each asset's share of portfolio variance (sums to 1; negative for a hedge)."""
    w = np.asarray(w, dtype=float)
    cov = _check_cov(cov)
    m = cov @ w
    v = float(w @ m)
    if v <= 0:
        raise ValueError("portfolio variance must be positive")
    return w * m / v


def risk_parity(cov, budgets=None, tol=1e-12, max_iter=10_000):
    """Long-only weights whose risk contributions are proportional to budgets (equal by default)."""
    cov = _check_cov(cov)
    n = cov.shape[0]
    d = np.diag(cov)
    if np.any(d <= 0):
        raise ValueError("every asset needs a positive variance")
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if b.shape != (n,) or np.any(b <= 0) or not np.all(np.isfinite(b)):
        raise ValueError("one positive budget per asset")
    b = b / b.sum()
    x = 1.0 / np.sqrt(d)
    x = x / np.sqrt(x @ cov @ x)                          # start at inverse volatility, scaled to unit variance
    for k in range(1, max_iter + 1):
        for i in range(n):
            r = cov[i] @ x - d[i] * x[i]                  # (S x)_i without the i term
            x[i] = (-r + np.sqrt(r * r + 4.0 * d[i] * b[i])) / (2.0 * d[i])
        w = x / x.sum()
        if np.abs(risk_contribution_shares(w, cov) - b).max() <= tol:
            return QpResult(w, k, True)
    return QpResult(x / x.sum(), max_iter, False)
