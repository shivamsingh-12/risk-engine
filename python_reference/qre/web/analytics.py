"""Phase 15: volatility and structure of a portfolio, for the app's charts and tables.

All on simple daily returns r_t = P_t / P_{t-1} - 1 with the given (daily-rebalanced) weights w, annualized with
sqrt(252) where stated. Risk-model numbers (VaR/ES, backtests) come from the engine; this module adds the
descriptive side:
  realized vol        std(r_p) sqrt(252) over the whole period and the last 20 / 60 / 250 days
  EWMA vol            RiskMetrics recursion (lambda = 0.94), today's value
  GARCH(1,1)          next-day and long-run vol, persistence alpha + beta and the half-life of a shock,
                      ln(0.5) / ln(alpha + beta) days
  risk contributions  RC_i = w_i (Sigma w)_i / sigma_p, which sum to sigma_p (Euler); shown as % of portfolio vol
  diversification     DR = sum w_i sigma_i / sigma_p >= 1; 1 means no diversification benefit
  drawdown            1 - V_t / max_{s<=t} V_s on the value index V
"""
import math

import numpy as np

from qre.risk import ewma as ew
from qre.risk import garch as gh
from qre.risk import var_es as ve

TRADING_DAYS = 252
ANN = math.sqrt(TRADING_DAYS)


def _vol(x):
    return float(np.std(x, ddof=1) * ANN) if len(x) > 1 else float("nan")


def rolling_vol(r, n):
    out = np.full(r.size, np.nan)
    if r.size >= n:
        c1 = np.concatenate([[0.0], np.cumsum(r)])
        c2 = np.concatenate([[0.0], np.cumsum(r * r)])
        s1, s2 = c1[n:] - c1[:-n], c2[n:] - c2[:-n]
        var = np.maximum((s2 - s1 * s1 / n) / (n - 1), 0.0)
        out[n - 1:] = np.sqrt(var) * ANN
    return out


def drawdown(value):
    peak = np.maximum.accumulate(value)
    return 1.0 - value / peak


def analyze(dates, tickers, prices, weights, window=250):
    prices = np.asarray(prices, dtype=float)
    w = np.asarray(weights, dtype=float)
    rets = prices[1:] / prices[:-1] - 1.0
    rp = ve.portfolio_returns(rets, w)
    rdates = list(dates[1:])
    value = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + rp)])
    dd = drawdown(value)
    trough = int(np.argmax(dd))
    peak = int(np.argmax(value[:trough + 1])) if trough > 0 else 0

    recent = rets[-window:]
    cov = np.cov(recent, rowvar=False, ddof=1).reshape(len(tickers), len(tickers))
    sig_p = ve.portfolio_vol(w, cov)
    rc = ve.risk_contributions(w, cov)
    asset_vol = np.sqrt(np.diag(cov))
    sd = np.where(asset_vol > 0, asset_vol, 1.0)
    corr = cov / np.outer(sd, sd)

    ewma_path = ew.ewma_vol_path(rp)
    g = gh.fit_garch(rp[-window:])
    h_next = gh.forecast_variance(g, rp[-window:])
    persistence = g.alpha + g.beta
    half_life = math.log(0.5) / math.log(persistence) if 0.0 < persistence < 1.0 else float("inf")

    order = np.argsort(rp)[:5]
    five = np.array([np.prod(1.0 + rp[i:i + 5]) - 1.0 for i in range(rp.size - 4)]) if rp.size >= 5 else np.array([0.0])
    k5 = int(np.argmin(five))
    return {
        "period": {"start": dates[0], "end": dates[-1], "days": len(rp)},
        "volatility": {
            "realized_full": _vol(rp), "realized_250": _vol(rp[-250:]), "realized_60": _vol(rp[-60:]),
            "realized_20": _vol(rp[-20:]), "ewma_now": float(ewma_path[-1] * ANN),
            "garch_next_day": float(math.sqrt(h_next) * ANN), "garch_long_run": float(math.sqrt(g.long_run_var) * ANN),
            "garch_alpha": g.alpha, "garch_beta": g.beta, "garch_persistence": persistence,
            "garch_half_life_days": half_life if math.isfinite(half_life) else None,
        },
        "returns": {
            "annualized_return": float(value[-1] / 100.0) ** (TRADING_DAYS / len(rp)) - 1.0,
            "mean_daily": float(rp.mean()), "skew": float(((rp - rp.mean()) ** 3).mean() / rp.std() ** 3),
            "excess_kurtosis": float(((rp - rp.mean()) ** 4).mean() / rp.var() ** 2 - 3.0),
            "max_drawdown": float(dd[trough]), "max_drawdown_peak": dates[peak], "max_drawdown_trough": dates[trough],
            "current_drawdown": float(dd[-1]),
            "worst_days": [{"date": rdates[i], "return": float(rp[i])} for i in order],
            "worst_week": {"start": rdates[k5], "return": float(five[k5])},
        },
        "assets": [{"ticker": t, "weight": float(w[j]), "vol": float(asset_vol[j] * ANN),
                    "risk_contribution": float(rc[j] / sig_p) if sig_p > 0 else 0.0}
                   for j, t in enumerate(tickers)],
        "portfolio_vol_250": float(sig_p * ANN),
        "diversification_ratio": float((w @ asset_vol) / sig_p) if sig_p > 0 else 1.0,
        "correlation": corr.tolist(),
        "series": {
            "dates": list(dates), "value": value.tolist(), "drawdown": dd.tolist(),
            "return_dates": rdates, "returns": rp.tolist(),
            "vol20": _nan_to_none(rolling_vol(rp, 20)), "vol60": _nan_to_none(rolling_vol(rp, 60)),
            "ewma": (ewma_path[1:] * ANN).tolist(),        # value after each day = forecast for the next
        },
    }


def _nan_to_none(a):
    return [None if not math.isfinite(x) else float(x) for x in a]
