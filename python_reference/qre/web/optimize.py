"""Phase 21: suggested weights for the app, built on qre.portfolio.construction.

The covariance is Ledoit-Wolf on the last `window` days of returns (at most 500). Expected returns, used only by
mean-variance, are the plain historical mean of the same window: Experiment 26 found that this makes mean-variance
weights swing and underperform, and the response says so. Each portfolio, current and suggested, is described by
the same in-sample numbers so they can be compared: volatility, 99% 1-day VaR (parametric and historical) and
each holding's share of risk.
"""
import math

import numpy as np

from qre.linalg.shrinkage import ledoit_wolf
from qre.portfolio import construction as pc
from qre.risk import var_es as ve

METHODS = ("min_variance", "risk_parity", "mean_variance", "inverse_vol", "equal")
ANN = math.sqrt(252)


def describe(w, rets, cov):
    rp = rets @ w
    var_h, es_h = ve.historical_var_es(rp, 0.99)
    sig = math.sqrt(max(float(w @ cov @ w), 0.0))
    var_p, es_p = ve.parametric_var_es(0.0, sig, 0.99)
    shares = pc.risk_contribution_shares(w, cov) if sig > 0 else np.zeros_like(w)
    return {"weights": w.tolist(), "vol": sig * ANN, "var99_parametric": var_p, "var99_historical": var_h,
            "es99_historical": es_h, "mean_return": float(rp.mean() * 252), "risk_shares": shares.tolist(),
            "gross": float(np.abs(w).sum())}


def suggest(rets, current, method, min_weight=0.0, max_weight=1.0, gamma=5.0, window=500):
    """rets: T x N daily returns (most recent last). Returns a JSON-ready dict."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    rets = np.asarray(rets, dtype=float)[-window:]
    n = rets.shape[1]
    if n < 2:
        raise ValueError("add at least two holdings to optimize")
    if rets.shape[0] < 100:
        raise ValueError("need at least 100 days of common history")
    if not -1.0 <= min_weight <= 0.0 or not 0.0 < max_weight <= 1.5:
        raise ValueError("minimum weight must be between -100% and 0%, maximum between 0% and 150%")
    cov, shrink = ledoit_wolf(rets)
    note = None
    if method == "equal":
        w = pc.equal_weight(n)
    elif method == "inverse_vol":
        w = pc.inverse_vol(cov)
    elif method == "risk_parity":
        w = pc.risk_parity(cov).weights
        if min_weight < 0 or max_weight < 1:
            note = "risk parity is long-only and ignores the weight limits"
    else:
        lo, hi = pc.bounds(n, min_weight, max_weight)
        if method == "min_variance":
            res = pc.min_variance(cov, lo, hi)
        else:
            res = pc.mean_variance(rets.mean(axis=0), cov, gamma, lo, hi)
            note = ("expected returns are the historical average of the window, which mostly measures luck: in "
                    "Experiment 26 mean-variance had the worst risk-adjusted results and 5 to 10 times the turnover")
        if not res.converged:
            raise ValueError("the optimizer did not converge")
        w = res.weights
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    cur = np.asarray(current, dtype=float)
    return {"method": method, "window_days": int(rets.shape[0]), "shrinkage": shrink, "note": note,
            "constraints": {"min_weight": min_weight, "max_weight": max_weight, "gamma": gamma},
            "current": describe(cur, rets, cov), "suggested": describe(w, rets, cov),
            "turnover": float(0.5 * np.abs(w - cur).sum())}
