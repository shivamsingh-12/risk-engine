"""Component VaR and ES reference (Phase 21): how much of a portfolio's tail risk each holding is responsible for.

VaR and ES are homogeneous of degree one in the weights (double every position, double the risk), so by Euler's
theorem they split exactly into components C_i = w_i * dRisk/dw_i that sum to the total. A component is the
holding's marginal effect on the risk times its size. It can be negative: a hedge reduces the total.

Parametric (normal, zero mean), with portfolio volatility s = sqrt(w'Sw):
    VaR = z s                         component VaR_i = z     w_i (S w)_i / s
    ES  = phi(z) / (1 - c) s          component ES_i  = phi(z)/(1 - c) w_i (S w)_i / s

Historical (scenarios r_t of the assets, portfolio loss L_t = -w'r_t):
    ES  = mean of the losses at or beyond VaR (the definition in var_es.var_es_from_losses)
    component ES_i = mean of -w_i r_t,i over those same days. Exact: the components sum to the ES.
    component VaR_i = kernel-weighted mean of -w_i r_t,i around the VaR (Epperlein and Smillie 2006), with Gaussian
    weights K((L_t - VaR) / h) and Silverman's bandwidth h = 1.06 sd(L) T^(-1/5). The weighted losses sum to about
    the VaR, not exactly, so the components are rescaled to sum to it.

Matches cpp/include/qre/risk/component.hpp.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from qre.risk.var_es import var_es_from_losses


@dataclass
class ComponentRisk:
    var: float
    es: float
    component_var: np.ndarray       # sums to var
    component_es: np.ndarray        # sums to es
    tail_days: int = 0              # historical: number of scenarios in the ES average


def _check(weights, confidence):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a non-empty finite vector")
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1)")
    return w


def parametric_components(weights, cov, confidence):
    w = _check(weights, confidence)
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (w.size, w.size):
        raise ValueError("cov must be square and match the number of weights")
    m = cov @ w
    v = float(w @ m)
    if v <= 0:
        raise ValueError("portfolio variance must be positive")
    s = np.sqrt(v)
    z = float(norm.ppf(confidence))
    k_es = float(norm.pdf(z)) / (1.0 - confidence)
    share = w * m / s                                   # sums to s
    return ComponentRisk(var=z * s, es=k_es * s, component_var=z * share, component_es=k_es * share)


def historical_components(returns, weights, confidence):
    """returns: T x N scenarios of asset returns (e.g. the last 250 days)."""
    w = _check(weights, confidence)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[1] != w.size or r.shape[0] < 20 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite, T x N with T >= 20 and one column per weight")
    pos = -(r * w)                                      # each holding's loss on each day
    loss = pos.sum(axis=1)
    var, es = var_es_from_losses(loss, confidence)
    tail = loss >= var
    comp_es = pos[tail].mean(axis=0)
    sd = float(loss.std(ddof=1))
    h = 1.06 * sd * loss.size ** -0.2 if sd > 0 else 1.0
    k = np.exp(-0.5 * ((loss - var) / h) ** 2)
    raw = (k[:, None] * pos).sum(axis=0) / k.sum()
    total = float(raw.sum())
    comp_var = raw * (var / total) if total != 0.0 else raw
    return ComponentRisk(var=var, es=es, component_var=comp_var, component_es=comp_es,
                         tail_days=int(tail.sum()))
