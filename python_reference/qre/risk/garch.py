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

Student-t innovations (dist="t"): z_t follows a STANDARDIZED Student-t with nu > 2 degrees of freedom
(scaled to unit variance, so h_t is still the variance). nu is estimated jointly with alpha and beta via
nu = 2 + exp(c). Small nu means fat tails; nu -> infinity recovers the normal model.
    log density of r_t:  lgamma((nu+1)/2) - lgamma(nu/2) - 0.5 ln(pi (nu-2)) - 0.5 ln h_t
                         - (nu+1)/2 ln(1 + r_t^2 / (h_t (nu-2)))
    VaR/ES at confidence c with q = t_nu^{-1}(c) and s = sigma sqrt((nu-2)/nu):
        VaR = s q,   ES = s f_nu(q) / (1-c) (nu + q^2) / (nu - 1)
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t as student_t

from qre.risk import var_es as ve

P_MAX = 0.9999                       # persistence cap: alpha + beta < 1 keeps omega > 0
STARTS = ((0.05, 0.90), (0.10, 0.80), (0.02, 0.97))
T_STARTS = ((0.05, 0.90, 8.0), (0.10, 0.80, 5.0), (0.02, 0.97, 20.0))
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


# Upper bound on nu. The likelihood is nearly flat as nu -> infinity, and without a bound the fit can drift to
# nu ~ 1e16, where gammaln differences lose all precision (in C++ the Student-t cdf breaks down too, giving negative
# VaR and ES ~ 1e32). At nu = 1000 the t is the normal for risk purposes, so reaching the bound means "normal shocks".
NU_MAX = 1000.0


def nu_from_theta(c):
    return 2.0 + math.exp(c)


def neg_loglik_t(theta, returns, v):
    alpha, beta = params_from_theta(theta[:2])
    nu = nu_from_theta(theta[2])
    if not nu <= NU_MAX:
        return math.inf
    r = np.asarray(returns, dtype=float)
    h = variance_path(r, v * (1.0 - alpha - beta), alpha, beta, v)[:-1]
    const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(math.pi * (nu - 2.0))
    ll = const - 0.5 * np.log(h) - 0.5 * (nu + 1.0) * np.log1p(r * r / (h * (nu - 2.0)))
    return -float(np.sum(ll))


NU_NORMAL_LIMIT = 1e6


def std_t_var_es(sigma, nu, confidence):
    """VaR/ES (positive losses) of sigma * z, z a standardized Student-t with nu > 2 degrees of freedom."""
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if sigma <= 0.0 or nu <= 2.0:
        raise ValueError("need sigma > 0 and nu > 2")
    if nu > NU_NORMAL_LIMIT:              # same normal limit as the C++ (the t is the normal to ~1e-6 here)
        return ve.parametric_var_es(0.0, sigma, confidence)
    q = float(student_t.ppf(confidence, nu))
    s = sigma * math.sqrt((nu - 2.0) / nu)
    es = s * float(student_t.pdf(q, nu)) / (1.0 - confidence) * (nu + q * q) / (nu - 1.0)
    return s * q, es


@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float
    long_run_var: float
    loglik: float
    converged: bool
    nu: float = math.inf       # Student-t degrees of freedom; inf for the normal model

    @property
    def persistence(self):
        return self.alpha + self.beta


def fit_garch(returns, dist="normal"):
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size < 10:
        raise ValueError("need a 1-D series with at least 10 returns")
    if dist not in ("normal", "t"):
        raise ValueError("dist must be 'normal' or 't'")
    v = float(np.mean(r * r))
    if v <= 0.0:
        raise ValueError("returns are all zero; volatility is undefined")
    if dist == "t":
        best = None
        for a0, b0, nu0 in T_STARTS:
            x0 = np.append(theta_from_params(a0, b0), math.log(nu0 - 2.0))
            res = minimize(neg_loglik_t, x0, args=(r, v), method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 6000, "maxfev": 12000})
            if best is None or res.fun < best.fun:
                best = res
        alpha, beta = params_from_theta(best.x[:2])
        return GarchFit(v * (1.0 - alpha - beta), alpha, beta, v, -float(best.fun), bool(best.success),
                        nu_from_theta(best.x[2]))
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


def garch_var_es(returns, confidence, filtered=False, dist="normal"):
    """GARCH VaR/ES of a 1-D portfolio return series.
    filtered=False: the model's own distribution (normal, or standardized t when dist="t") at the forecast vol.
    filtered=True : filtered historical simulation with GARCH volatilities (standardized residuals
                    rescaled to the forecast), which keeps the empirical tail shape."""
    r = np.asarray(returns, dtype=float)
    fit = fit_garch(r, dist)
    h = variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var)
    sigma_next = math.sqrt(h[-1])
    if not filtered:
        if dist == "t":
            return std_t_var_es(sigma_next, fit.nu, confidence)
        return ve.parametric_var_es(0.0, sigma_next, confidence)
    scenarios = r / np.sqrt(h[:-1]) * sigma_next
    return ve.var_es_from_losses(-scenarios, confidence)


def simulate_garch(n, omega, alpha, beta, seed=0, nu=None):
    """Simulated GARCH(1,1) returns, started at the long-run variance (used for testing).
    nu=None: normal shocks; otherwise standardized Student-t shocks with nu degrees of freedom."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) if nu is None else rng.standard_t(nu, n) * math.sqrt((nu - 2.0) / nu)
    h = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        out[t] = math.sqrt(h) * z[t]
        h = omega + alpha * out[t] ** 2 + beta * h
    return out
