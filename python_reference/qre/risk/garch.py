"""GARCH(1,1) reference: variance-targeted maximum likelihood, forecasts and VaR/ES.

Model (zero-mean returns, as in the EWMA module):
    r_t = sqrt(h_t) z_t,  z_t ~ N(0, 1)
    h_{t+1} = omega + alpha r_t^2 + beta h_t,   omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
    long-run variance v = omega / (1 - alpha - beta)
EWMA is the special case omega = 0, alpha = 1 - lambda, beta = lambda (persistence exactly 1, no mean reversion).

Estimation (matches cpp/include/qre/risk/garch.hpp):
    - Variance targeting: v is fixed to the sample second moment mean(r^2), so omega = v (1 - alpha - beta)
      and only (alpha, beta) are estimated. Fewer parameters -> much more stable on short windows.
    - Unconstrained reparameterization so any optimizer step is a valid model:
          p = P_MAX * logistic(a),  alpha = p * logistic(b),  beta = p - alpha
    - h_0 = v. Negative log-likelihood = 0.5 * sum(ln(2 pi) + ln h_t + r_t^2 / h_t), t = 0..n-1.
    - Nelder-Mead from three starting points, keep the best. This reference uses scipy's Nelder-Mead;
      the C++ uses its own implementation, and the tests check both find the same optimum.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from qre.risk import var_es as ve

P_MAX = 0.9999                       # persistence cap: alpha + beta < 1 keeps omega > 0
STARTS = ((0.05, 0.90), (0.10, 0.80), (0.02, 0.97))
LOG_2PI = math.log(2.0 * math.pi)


def _logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p):
    return math.log(p / (1.0 - p))


def params_from_theta(theta):
    p = P_MAX * _logistic(theta[0])
    alpha = p * _logistic(theta[1])
    return alpha, p - alpha


def theta_from_params(alpha, beta):
    p = alpha + beta
    return np.array([_logit(p / P_MAX), _logit(alpha / p)])


def variance_path(returns, omega, alpha, beta, h0):
    """h_0..h_n: h_t is the variance forecast for r_t made before seeing it; h_n is for the next day."""
    r = np.asarray(returns, dtype=float)
    h = np.empty(r.size + 1)
    h[0] = h0
    for t in range(r.size):
        h[t + 1] = omega + alpha * r[t] * r[t] + beta * h[t]
    return h


def neg_loglik(theta, returns, v):
    alpha, beta = params_from_theta(theta)
    r = np.asarray(returns, dtype=float)
    h = variance_path(r, v * (1.0 - alpha - beta), alpha, beta, v)[:-1]
    return 0.5 * float(np.sum(LOG_2PI + np.log(h) + r * r / h))


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    long_run_var: float
    loglik: float
    converged: bool

    @property
    def persistence(self):
        return self.alpha + self.beta


def fit_garch(returns):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 10:
        raise ValueError("need a 1-D series with at least 10 returns")
    v = float(np.mean(r * r))
    if v <= 0.0:
        raise ValueError("returns are all zero; volatility is undefined")
    best = None
    for a0, b0 in STARTS:
        res = minimize(neg_loglik, theta_from_params(a0, b0), args=(r, v), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000, "maxfev": 8000})
        if best is None or res.fun < best.fun:
            best = res
    alpha, beta = params_from_theta(best.x)
    return GarchFit(v * (1.0 - alpha - beta), alpha, beta, v, -float(best.fun), bool(best.success))


def forecast_variance(fit, returns):
    """One-day-ahead variance after the last observed return."""
    return float(variance_path(returns, fit.omega, fit.alpha, fit.beta, fit.long_run_var)[-1])


def garch_var_es(returns, confidence, filtered=False):
    """GARCH VaR/ES of a 1-D portfolio return series.
    filtered=False: normal with the GARCH volatility forecast.
    filtered=True : filtered historical simulation with GARCH volatilities (standardized residuals
                    rescaled to the forecast), which keeps the empirical tail shape."""
    r = np.asarray(returns, dtype=float)
    fit = fit_garch(r)
    h = variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var)
    sigma_next = math.sqrt(h[-1])
    if not filtered:
        return ve.parametric_var_es(0.0, sigma_next, confidence)
    scenarios = r / np.sqrt(h[:-1]) * sigma_next
    return ve.var_es_from_losses(-scenarios, confidence)


def simulate_garch(n, omega, alpha, beta, seed=0):
    """Simulated GARCH(1,1) returns, started at the long-run variance (used for testing)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    h = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        out[t] = math.sqrt(h) * z[t]
        h = omega + alpha * out[t] ** 2 + beta * h
    return out
