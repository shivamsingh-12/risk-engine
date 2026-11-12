"""Phase 13 reference: an adaptive risk engine that scores its models every day and combines them.

Every earlier phase added a model; Experiment 16 found ten of them disagreeing by about +-9% on the same day,
with no model clearly best. The engine here runs all of them side by side and, each day, decides how much to trust
each one using only their past scores (FZ0 loss, see qre.risk.scoring). Three rules:
    equal   : average of all models' VaR and ES (no learning; the classic hard-to-beat forecast combination)
    select  : use the single model with the lowest mean FZ0 over the last `lookback` days
    exp     : weights w_k proportional to exp(-eta * S_k), S_k the summed FZ0 over the last `lookback` days
              (exponentially weighted forecasting, the "Hedge" rule). eta -> 0 gives equal weights,
              eta -> infinity gives select.
Combined forecasts are convex combinations of (VaR, ES) pairs, so ES >= VaR is preserved.
No look-ahead: the weights used on day t depend only on forecasts and losses of days before t.
The first `burn` days only build up scores, and evaluation starts after them.
"""
import numpy as np

from qre.risk import backtest as bt
from qre.risk import scoring


SCHEMES = ("equal", "select", "exp")


def run_models(prices, weights, methods, window=250, confidence=0.99):
    """Rolling forecasts from each method. Returns (var (T, K), es (T, K), loss (T,), rows (T,))."""
    series = [bt.rolling_backtest(prices, weights, window, confidence, m) for m in methods]
    rows = series[0].rows
    if any(not np.array_equal(s.rows, rows) for s in series):
        raise ValueError("models must forecast the same days")
    return (np.column_stack([s.var for s in series]), np.column_stack([s.es for s in series]),
            series[0].loss, rows)


def combine(var, es, loss, confidence, scheme="equal", lookback=100, eta=1.0, burn=100):
    """Combined forecasts for days burn .. T-1. Returns (var (T-burn,), es (T-burn,), weights (T-burn, K))."""
    var, es, loss = (np.asarray(a, dtype=float) for a in (var, es, loss))
    if var.ndim != 2 or es.shape != var.shape or loss.shape != (var.shape[0],):
        raise ValueError("var and es must be (T, K) and loss (T,)")
    if scheme not in SCHEMES:
        raise ValueError("scheme must be equal, select or exp")
    t_total, k = var.shape
    if not (1 <= burn < t_total) or lookback < 1 or eta < 0.0:
        raise ValueError("need 1 <= burn < T, lookback >= 1 and eta >= 0")
    scores = scoring.fz0_loss(loss[:, None], var, es, confidence)
    w = np.empty((t_total - burn, k))
    for i, t in enumerate(range(burn, t_total)):
        past = scores[max(0, t - lookback):t]
        if scheme == "equal":
            w[i] = 1.0 / k
        elif scheme == "select":
            w[i] = 0.0
            w[i, int(np.argmin(past.mean(axis=0)))] = 1.0
        else:
            s = past.sum(axis=0)
            e = np.exp(-eta * (s - s.min()))
            w[i] = e / e.sum()
    return (w * var[burn:]).sum(axis=1), (w * es[burn:]).sum(axis=1), w
