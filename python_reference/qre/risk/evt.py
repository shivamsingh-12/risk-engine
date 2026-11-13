"""Phase 11 reference: extreme value theory for tail risk, and the uncertainty around risk estimates.

Peaks over threshold (POT). By the Pickands-Balkema-de Haan theorem, for a wide class of distributions the
excess over a high threshold u, Y = L - u given L > u, is approximately generalized Pareto:
    P(Y <= y) = 1 - (1 + xi y / beta)^(-1/xi)       (xi != 0),   1 - exp(-y / beta)   (xi = 0)
xi is the tail index: xi > 0 is a heavy (power-law) tail, the Student-t with nu degrees of freedom has xi = 1/nu;
xi = 0 is a thin exponential tail (normal, lognormal); xi < 0 has a finite upper end point.
With n losses, N_u of them above u, the tail quantile and expected shortfall at confidence p are
    VaR_p = u + beta / xi * ((n / N_u (1 - p))^(-xi) - 1)
    ES_p  = VaR_p / (1 - xi) + (beta - xi u) / (1 - xi)          (needs xi < 1)
so EVT extrapolates beyond the largest observed losses using the fitted tail shape.

GPD fit: maximum likelihood over (xi, ln beta) with Nelder-Mead (the C++ uses the engine's own Nelder-Mead),
started from the method-of-moments estimate. Tests check it against scipy.stats.genpareto.fit.
xi is restricted to (XI_MIN, XI_MAX) = (-0.5, 0.95): below -0.5 maximum likelihood is irregular, and at xi >= 1 the
mean (so ES) is infinite. With few exceedances (say under 30) the unrestricted MLE of xi is very unstable and can
land beyond 1; fit_gpd reports whether the bound was hit so callers can see how often that happens.

Hill estimator of xi from the k largest (positive) losses X_(1) >= ... >= X_(k+1):
    xi_H = (1/k) sum_{i=1}^{k} ln X_(i) - ln X_(k+1)
Only valid for heavy tails (xi > 0). Small k: high variance; large k: bias from the non-tail part.

Conditional EVT (McNeil and Frey 2000): filter returns with GARCH, apply POT to the standardized residuals'
losses, then scale the residual VaR/ES by the next-day GARCH volatility. Adapts to volatility like FHS-GARCH,
but extrapolates the tail with the GPD instead of using only observed residuals.

Bootstrap: resample the losses (moving blocks of length b keep short-range dependence; b = 1 is the iid
bootstrap), recompute historical VaR/ES each time, and read percentile confidence intervals.
"""
import math

import numpy as np
from scipy.optimize import minimize

from qre.risk import garch
from qre.risk import var_es as ve


XI_MIN, XI_MAX = -0.5, 0.95


def _gpd_nll(theta, y):
    xi, beta = theta[0], math.exp(theta[1])
    if not (XI_MIN < xi < XI_MAX):
        return math.inf
    if abs(xi) < 1e-9:
        return y.size * math.log(beta) + y.sum() / beta
    z = 1.0 + xi * y / beta
    if np.any(z <= 0.0):
        return math.inf
    return y.size * math.log(beta) + (1.0 + 1.0 / xi) * np.sum(np.log(z))


def fit_gpd(exceedances):
    """MLE of the generalized Pareto (xi, beta) for positive exceedances. Returns (xi, beta);
    gpd_at_bound(xi) tells whether the xi restriction was binding."""
    y = np.asarray(exceedances, dtype=float)
    if y.ndim != 1 or y.size < 5 or np.any(y <= 0.0):
        raise ValueError("need at least 5 positive exceedances")
    m, v = y.mean(), y.var(ddof=1)
    xi0 = float(np.clip(0.5 * (1.0 - m * m / v), -0.4, 0.8))
    beta0 = max(0.5 * m * (m * m / v + 1.0), 1e-12)
    best = None
    for x0 in ((xi0, math.log(beta0)), (0.1, math.log(m))):
        if not math.isfinite(_gpd_nll(np.array(x0), y)):
            continue                  # moment start outside the support; the xi = 0.1 start is always feasible
        res = minimize(_gpd_nll, np.array(x0), args=(y,), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(math.exp(best.x[1]))


def gpd_at_bound(xi, tol=1e-4):
    return xi <= XI_MIN + tol or xi >= XI_MAX - tol


def gpd_var_es(u, xi, beta, n, n_u, confidence):
    """Tail VaR/ES from a GPD fitted above threshold u with n_u of n observations exceeding it."""
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    if xi >= 1.0:
        raise ValueError("ES is infinite for xi >= 1")
    tail = n / n_u * (1.0 - confidence)
    if abs(xi) < 1e-9:
        var = u - beta * math.log(tail)
        return var, var + beta
    var = u + beta / xi * (tail ** (-xi) - 1.0)
    return var, var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi)


def pot_var_es(losses, confidence, threshold_quantile=0.9):
    """Peaks-over-threshold VaR/ES. Returns (var, es, xi, beta, u, n_u)."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or x.size < 20:
        raise ValueError("need at least 20 losses")
    if not (0.0 < threshold_quantile < 1.0) or confidence <= threshold_quantile:
        raise ValueError("need 0 < threshold_quantile < confidence < 1")
    u = float(np.quantile(x, threshold_quantile))
    exc = x[x > u] - u
    xi, beta = fit_gpd(exc)
    var, es = gpd_var_es(u, xi, beta, x.size, exc.size, confidence)
    return var, es, xi, beta, u, int(exc.size)


def hill(losses, k):
    """Hill estimate of the tail index xi from the k largest losses (which must be positive)."""
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    if not (1 <= k < x.size) or x[k] <= 0.0:
        raise ValueError("need 1 <= k < n and the (k+1)-th largest loss positive")
    return float(np.mean(np.log(x[:k])) - math.log(x[k]))


def garch_evt_var_es(returns, confidence, threshold_quantile=0.9):
    """Conditional EVT: GARCH(1,1) filter, POT on standardized residual losses, rescale by next-day vol."""
    r = np.asarray(returns, dtype=float)
    fit = garch.fit_garch(r)
    h = garch.variance_path(r, fit.omega, fit.alpha, fit.beta, fit.long_run_var)
    z_loss = -r / np.sqrt(h[:-1])
    var_z, es_z, *_ = pot_var_es(z_loss, confidence, threshold_quantile)
    s = math.sqrt(h[-1])
    return s * var_z, s * es_z


def bootstrap_var_es(losses, confidence, n_boot=2000, block=1, seed=42, level=0.90):
    """Percentile CIs for historical VaR and ES. Returns ((var_lo, var_hi), (es_lo, es_hi))."""
    x = np.asarray(losses, dtype=float)
    n = x.size
    if n < 20 or block < 1 or block > n or n_boot < 10:
        raise ValueError("need n >= 20, 1 <= block <= n and n_boot >= 10")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    vs, es = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        vs[b], es[b] = ve.var_es_from_losses(sample, confidence)
    a = (1.0 - level) / 2.0
    return (float(np.quantile(vs, a)), float(np.quantile(vs, 1 - a))), (float(np.quantile(es, a)), float(np.quantile(es, 1 - a)))
