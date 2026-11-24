"""Multi-day VaR and ES reference (Phase 21): risk over h trading days (Basel's market-risk horizon is 10).

Four ways to get an h-day figure from daily data:

sqrt_time
    Scale a 1-day VaR/ES by sqrt(h). Exact for i.i.d. normal returns; ignores volatility clustering (today's high
    volatility persists) and fat tails (sums of fat-tailed returns are less fat-tailed).
overlapping
    Historical simulation on overlapping h-day compounded returns, prod(1 + r) - 1, from the window. Uses real
    multi-day paths but has only T - h + 1 heavily dependent scenarios.
garch_normal
    GARCH(1,1) h-day variance: sum of the expected daily variances E[sigma^2_t+k], which revert from the next day's
    forecast towards the long-run variance v at the rate p = alpha + beta:
        V_h = h v + (sigma^2_1 - v) (1 - p^h) / (1 - p)
    then normal VaR/ES with standard deviation sqrt(V_h). Captures clustering, not fat tails.
fhs_garch
    Filtered historical simulation over h days: fit GARCH, standardize the window's returns by their fitted
    volatilities, and simulate paths that draw those standardized shocks at random (bootstrap), each day's
    volatility following the GARCH recursion from the last one. VaR/ES of the simulated compounded h-day losses.
    Captures clustering, the volatility feedback within the horizon and the empirical tail shape.

fhs_garch_paths takes the bootstrap draws (indices into the standardized residuals) as an argument, so the C++
and Python implementations can be checked against each other on the same draws; fhs_garch draws them itself.
Matches cpp/include/qre/risk/horizon.hpp.
"""
import math

import numpy as np

from qre.risk import garch as gh
from qre.risk.var_es import historical_var_es, parametric_var_es, var_es_from_losses


def _check(returns, h):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 20 or not np.all(np.isfinite(r)):
        raise ValueError("need a finite 1-D series of at least 20 daily returns")
    if int(h) != h or h < 1:
        raise ValueError("horizon must be a whole number of days, at least 1")
    return r, int(h)


def sqrt_time(returns, confidence, h):
    r, h = _check(returns, h)
    return historical_var_es(r, confidence, h)


def overlapping(returns, confidence, h):
    r, h = _check(returns, h)
    if r.size - h + 1 < 20:
        raise ValueError("window too short for overlapping multi-day returns")
    growth = np.cumprod(np.concatenate([[1.0], 1.0 + r]))
    multi = growth[h:] / growth[:-h] - 1.0
    return var_es_from_losses(-multi, confidence)


def garch_variance_sum(fit, sigma2_next, h):
    p = fit.alpha + fit.beta
    v = fit.long_run_var
    if p >= 1.0:
        return h * sigma2_next
    return h * v + (sigma2_next - v) * (1.0 - p ** h) / (1.0 - p)


def garch_normal(returns, confidence, h, fit=None):
    r, h = _check(returns, h)
    fit = fit or gh.fit_garch(r)
    s2 = gh.forecast_variance(fit, r)
    return parametric_var_es(0.0, math.sqrt(garch_variance_sum(fit, s2, h)), confidence)


def fhs_garch_paths(returns, h, draws, fit=None):
    """Simulated h-day compounded returns. draws: n_paths x h integer indices into the window's returns."""
    r, h = _check(returns, h)
    d = np.asarray(draws)
    if d.ndim != 2 or d.shape[1] != h or d.shape[0] < 1 or np.any(d < 0) or np.any(d >= r.size):
        raise ValueError("draws must be n_paths x h indices into the returns")
    fit = fit or gh.fit_garch(r)
    hv = gh.variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var)
    z = r / np.sqrt(hv[:-1])                            # standardized residuals
    s2 = np.full(d.shape[0], hv[-1])
    growth = np.ones(d.shape[0])
    for k in range(h):
        x = np.sqrt(s2) * z[d[:, k]]
        growth *= 1.0 + x
        s2 = fit.omega + fit.alpha * x * x + fit.beta * s2
    return growth - 1.0


def fhs_garch(returns, confidence, h, n_paths=10_000, seed=0, fit=None):
    r, h = _check(returns, h)
    draws = np.random.default_rng(seed).integers(0, r.size, size=(n_paths, h))
    return var_es_from_losses(-fhs_garch_paths(r, h, draws, fit), confidence)
