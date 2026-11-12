"""Phase 13 reference: scoring risk forecasts, and testing whether one model is really better than another.

Backtests (Phase 7+) ask whether a model is acceptable: right breach rate, no clustering. To choose between models
that all pass, we need a score where lower is better and the truth wins on average. A scoring function S is
strictly consistent for a forecast if E[S(forecast, L)] is minimized exactly at the true value.

VaR alone (a quantile) has the quantile ("pinball") loss, with c the confidence level:
    QL(v, L) = (1{L <= v} - c) (v - L)
which charges c per unit of loss above v and (1 - c) per unit of room left below it.

ES alone is not elicitable (no such score exists), but (VaR, ES) jointly is (Fissler and Ziegel 2016).
The FZ0 loss of Patton, Ziegel and Chen (2019), written for losses L with VaR v > 0 and ES e > 0, alpha = 1 - c:
    FZ0(v, e, L) = 1{L > v} (L - v) / (alpha e) + v / e + ln(e) - 1
It is the member of the family that makes score differences depend only on the ratio of forecasts, so a period of
high volatility does not dominate the average.

Diebold-Mariano test: d_t = S_A,t - S_B,t. Under "equal predictive ability", E[d] = 0 and
    DM = mean(d) / sqrt(LRV / T),  LRV = gamma_0 + 2 sum_{k=1}^{L} (1 - k/(L+1)) gamma_k   (Newey-West, Bartlett)
is asymptotically standard normal. A negative DM means A has the lower (better) loss. The default lag is
floor(4 (T/100)^(2/9)), the usual Newey-West rule of thumb.
"""
import math

import numpy as np


def _check_confidence(confidence):
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")


def quantile_loss(loss, var, confidence):
    _check_confidence(confidence)
    loss, var = np.broadcast_arrays(np.asarray(loss, dtype=float), np.asarray(var, dtype=float))
    return ((loss <= var).astype(float) - confidence) * (var - loss)


def fz0_loss(loss, var, es, confidence):
    _check_confidence(confidence)
    loss, var, es = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (loss, var, es)))
    if np.any(es <= 0.0):
        raise ValueError("ES forecasts must be positive")
    alpha = 1.0 - confidence
    return (loss > var) * (loss - var) / (alpha * es) + var / es + np.log(es) - 1.0


def newey_west_lag(n):
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def diebold_mariano(loss_a, loss_b, lag=None):
    """Returns (DM statistic, two-sided p-value, mean difference A - B, lag used)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    if d.ndim != 1 or d.size < 10:
        raise ValueError("need two equal-length series of at least 10 scores")
    n = d.size
    lag = newey_west_lag(n) if lag is None else int(lag)
    if lag < 0 or lag >= n:
        raise ValueError("lag must be in [0, n)")
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0.0:
        return 0.0, 1.0, float(d.mean()), lag
    stat = d.mean() / math.sqrt(lrv / n)
    return float(stat), float(math.erfc(abs(stat) / math.sqrt(2.0))), float(d.mean()), lag
