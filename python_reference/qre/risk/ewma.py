"""Adaptive volatility reference: EWMA (RiskMetrics) and filtered historical simulation.

Conventions (match cpp/include/qre/risk/ewma.hpp):
    - Zero-mean returns, as in RiskMetrics: at daily horizons the mean is tiny next to the volatility
      and poorly estimated.
    - EWMA weights on a window of n returns (oldest first) are
          w_i = (1 - lam) * lam^(n-1-i) / (1 - lam^n),   i = 0..n-1,
      so the newest return has the largest weight and the weights sum to exactly 1.
    - EWMA vol path for a single return series r_0..r_{n-1}:
          s2_0 = mean(r^2) over the window (seed),   s2_{i+1} = lam * s2_i + (1 - lam) * r_i^2
      s_i is the forecast for r_i made BEFORE r_i was seen; s_n is the forecast for the next day.
    - Filtered historical simulation (Hull-White): z_i = r_i / s_i, scenarios = z_i * s_n, then the
      empirical VaR/ES of those scenarios. It keeps the shape of the historical distribution but
      rescales it to today's volatility.
"""
import math

import numpy as np

from qre.risk import var_es as ve


def _check_lambda(lam):
    if not (0.0 < lam < 1.0):
        raise ValueError("lambda must be in (0, 1)")


def ewma_weights(n, lam=0.94):
    _check_lambda(lam)
    if n < 1:
        raise ValueError("need at least 1 observation")
    k = np.arange(n - 1, -1, -1, dtype=float)          # age: n-1 for the oldest, 0 for the newest
    return (1.0 - lam) * lam**k / (1.0 - lam**n)


def ewma_covariance(returns, lam=0.94):
    """Zero-mean EWMA covariance of a (T x N) return window; the newest row weighs most."""
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[0] < 1:
        raise ValueError("returns must be a non-empty 2-D array")
    w = ewma_weights(returns.shape[0], lam)
    return (returns * w[:, None]).T @ returns


def ewma_vol_path(returns, lam=0.94):
    """Volatility forecasts s_0..s_n for a 1-D return series (see module docstring)."""
    _check_lambda(lam)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("need a 1-D series with at least 2 returns")
    s2 = np.empty(r.size + 1)
    s2[0] = float(np.mean(r * r))
    if s2[0] <= 0.0:
        raise ValueError("returns are all zero; volatility is undefined")
    for i in range(r.size):
        s2[i + 1] = lam * s2[i] + (1.0 - lam) * r[i] * r[i]
    return np.sqrt(s2)


def fhs_var_es(returns, confidence, lam=0.94):
    """Filtered historical simulation VaR/ES of a 1-D portfolio return series."""
    r = np.asarray(returns, dtype=float)
    s = ewma_vol_path(r, lam)
    scenarios = (r / s[:-1]) * s[-1]
    return ve.var_es_from_losses(-scenarios, confidence)


def ewma_parametric_var_es(returns, weights, confidence, lam=0.94):
    """Normal VaR/ES with the portfolio volatility taken from the EWMA covariance."""
    cov = ewma_covariance(returns, lam)
    return ve.parametric_var_es(0.0, ve.portfolio_vol(weights, cov), confidence)
