"""Backtesting reference: rolling out-of-sample VaR/ES forecasts and the tests that judge them.

Conventions (match cpp/include/qre/risk/backtest.hpp):
    - losses are positive; a breach on day t means realized loss_t > VaR_t (strictly).
    - alpha = 1 - confidence is the breach probability a correct model should produce.
    - Forecast for day t uses only price rows <= t-1 (data.hardening.pit_window).
    - Weights are held constant every day (daily rebalanced to fixed weights).
    - Methods: historical, parametric, monte_carlo (equal-weight window) and the adaptive
      ewma (EWMA-covariance parametric) and fhs (filtered historical simulation), both with decay lam.

Tests:
    Kupiec POF        H0: breach rate == alpha                       LR ~ chi2(1)
    Christoffersen    H0: breaches independent (no clustering)       LR ~ chi2(1)
    Conditional cov.  both of the above                              LR ~ chi2(2)
    Traffic light     Basel zones from the binomial cumulative probability of the breach count
    Z2 (Acerbi-Szekely, loss convention): Z2 = sum(L_t 1[L_t > VaR_t] / ES_t) / (T alpha) - 1.
        A correct model has E[L 1(L > VaR)] = alpha ES, so E[Z2] = 0; Z2 > 0 means ES was too small.
        Reported as a descriptive statistic; no p-value (its distribution depends on the model).
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from qre.data.hardening import pit_window
from qre.risk import ewma
from qre.risk import var_es as ve


def _xlogy(x, y):
    """x * ln(y) with the convention 0 * ln(0) = 0."""
    return 0.0 if x == 0 else x * math.log(y)


def chi2_sf_df1(x):
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def chi2_sf_df2(x):
    return math.exp(-max(x, 0.0) / 2.0)


def _check_alpha(alpha):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")


def kupiec_pof(n_obs, n_breaches, alpha):
    """Proportion-of-failures likelihood ratio and p-value."""
    _check_alpha(alpha)
    if n_obs < 1 or not (0 <= n_breaches <= n_obs):
        raise ValueError("need n_obs >= 1 and 0 <= n_breaches <= n_obs")
    x, n = n_breaches, n_obs
    phat = x / n
    ll_null = _xlogy(n - x, 1.0 - alpha) + _xlogy(x, alpha)
    ll_alt = _xlogy(n - x, 1.0 - phat) + _xlogy(x, phat)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return lr, chi2_sf_df1(lr)


def christoffersen_independence(breaches):
    """First-order Markov test that a breach today does not change the chance of one tomorrow."""
    b = np.asarray(breaches, dtype=bool)
    if b.size < 2:
        raise ValueError("need at least 2 observations")
    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum(~prev & ~cur)); n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur)); n11 = int(np.sum(prev & cur))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return lr, chi2_sf_df1(lr)


def conditional_coverage(breaches, alpha):
    b = np.asarray(breaches, dtype=bool)
    lr_pof, _ = kupiec_pof(b.size, int(b.sum()), alpha)
    lr_ind, _ = christoffersen_independence(b)
    lr = lr_pof + lr_ind
    return lr, chi2_sf_df2(lr)


def traffic_light(n_breaches, n_obs=250, alpha=0.01):
    """Basel zone: green if P(X <= x) < 95%, yellow if < 99.99%, else red, X ~ Binomial(n_obs, alpha)."""
    _check_alpha(alpha)
    if n_obs < 1 or not (0 <= n_breaches <= n_obs):
        raise ValueError("need n_obs >= 1 and 0 <= n_breaches <= n_obs")
    cum = binom.cdf(n_breaches, n_obs, alpha)
    if cum < 0.95:
        return "green"
    if cum < 0.9999:
        return "yellow"
    return "red"


def es_z2(losses, var, es, alpha):
    _check_alpha(alpha)
    losses, var, es = (np.asarray(a, dtype=float) for a in (losses, var, es))
    if not (losses.shape == var.shape == es.shape) or losses.size < 1:
        raise ValueError("losses, var and es must be non-empty and the same length")
    if np.any(es <= 0.0):
        raise ValueError("ES forecasts must be positive")
    hit = losses > var
    return float(np.sum(losses[hit] / es[hit]) / (losses.size * alpha) - 1.0)


@dataclass
class BacktestSeries:
    rows: np.ndarray       # forecast day t (row index into prices)
    var: np.ndarray
    es: np.ndarray
    loss: np.ndarray       # realized portfolio loss on day t

    @property
    def breaches(self):
        return self.loss > self.var


def rolling_backtest(prices, weights, window=250, confidence=0.99, method="historical",
                     n_paths=20_000, seed=42, trading_days=252, lam=0.94):
    """One forecast per day t = window+1 .. T-1, each from the `window` returns ending at row t-1."""
    prices = np.asarray(prices, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if prices.ndim != 2 or prices.shape[1] != weights.shape[0]:
        raise ValueError("prices columns must match number of weights")
    if window < 2 or prices.shape[0] < window + 2:
        raise ValueError("need window >= 2 and at least window + 2 price rows")
    if method not in ("historical", "parametric", "monte_carlo", "ewma", "fhs"):
        raise ValueError("method must be historical, parametric, monte_carlo, ewma or fhs")
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
        raise ValueError("prices must be finite and positive; run data hardening first")

    rows = np.arange(window + 1, prices.shape[0])
    var = np.empty(rows.size); es = np.empty(rows.size); loss = np.empty(rows.size)
    for i, t in enumerate(rows):
        w = pit_window(prices, t, window)
        rets = w[1:] / w[:-1] - 1.0
        if method == "historical":
            var[i], es[i] = ve.historical_var_es(ve.portfolio_returns(rets, weights), confidence)
        elif method == "fhs":
            var[i], es[i] = ewma.fhs_var_es(ve.portfolio_returns(rets, weights), confidence, lam)
        elif method == "ewma":
            var[i], es[i] = ewma.ewma_parametric_var_es(rets, weights, confidence, lam)
        else:
            cov = np.cov(rets, rowvar=False, ddof=1)
            if method == "parametric":
                var[i], es[i] = ve.parametric_var_es(0.0, ve.portfolio_vol(weights, cov), confidence)
            else:
                var[i], es[i] = ve.monte_carlo_var_es(weights, cov, confidence, 1, n_paths, seed + int(t),
                                                      trading_days)
        loss[i] = -float(weights @ (prices[t] / prices[t - 1] - 1.0))
    return BacktestSeries(rows, var, es, loss)


@dataclass
class BacktestSummary:
    n_obs: int
    n_breaches: int
    breach_rate: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    cc_lr: float
    cc_p: float
    traffic_light: str        # on the last 250 forecasts (or all, if fewer)
    z2: float


def summarize(series, confidence):
    alpha = 1.0 - confidence
    b = series.breaches
    k_lr, k_p = kupiec_pof(b.size, int(b.sum()), alpha)
    c_lr, c_p = christoffersen_independence(b)
    cc_lr, cc_p = conditional_coverage(b, alpha)
    last = b[-250:]
    return BacktestSummary(
        n_obs=int(b.size), n_breaches=int(b.sum()), breach_rate=float(b.mean()),
        kupiec_lr=k_lr, kupiec_p=k_p, christoffersen_lr=c_lr, christoffersen_p=c_p,
        cc_lr=cc_lr, cc_p=cc_p,
        traffic_light=traffic_light(int(last.sum()), int(last.size), alpha),
        z2=es_z2(series.loss, series.var, series.es, alpha),
    )
