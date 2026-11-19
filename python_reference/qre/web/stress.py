"""Phase 20: the app's stress tests, built on qre.risk.stress.

Prices from 2007 on for the holdings, the proxy (SPY) and any shocked tickers. Unlike the VaR analysis, holdings
need not share a history: a holding that did not trade during a scenario is replaced by beta x SPY on the days it
is missing, and the response says which holdings were proxied and for how many days.
"""
import numpy as np

from qre.risk import stress as st

PROXY = "SPY"
COV_DAYS = 500                     # covariance for hypothetical shocks: the most recent 500 common days
MIN_COV_DAYS = 120


def forward_fill_gaps(prices):
    """Carry the last price over internal gaps (other exchanges' holidays). Leading gaps (not yet listed) stay NaN."""
    p = np.array(prices, dtype=float)
    for j in range(p.shape[1]):
        ok = np.flatnonzero(np.isfinite(p[:, j]))
        if ok.size == 0:
            continue
        last = p[ok[0], j]
        for i in range(ok[0] + 1, p.shape[0]):
            if np.isfinite(p[i, j]):
                last = p[i, j]
            else:
                p[i, j] = last
    return p


def _returns(p):
    return p[1:] / p[:-1] - 1.0


def run_stress(dates, tickers, prices, weights, value, shocks=None, horizon_days=10, confidence=0.99,
               reverse_loss=0.2):
    """dates/tickers/prices: the full history, tickers = holdings first, then PROXY and shocked tickers not held.
    weights: one per holding (fractions of capital). shocks: [(ticker, return)]. Returns a JSON-ready dict."""
    dates = np.asarray(dates)
    p = forward_fill_gaps(prices)
    holdings = len(weights)
    w = np.asarray(weights, dtype=float)
    if PROXY not in tickers:
        raise ValueError("the proxy SPY must be in the price data")
    m = tickers.index(PROXY)
    keep = np.isfinite(p[:, m])                         # the proxy defines the trading calendar
    dates, p = dates[keep], p[keep]
    r = _returns(p)
    betas = np.array([st.beta(r[:, j], r[:, m]) if np.isfinite(r[:, j]).sum() >= 20 else np.nan
                      for j in range(holdings)])

    scenarios = []
    for sc in st.SCENARIOS:
        base = {"key": sc.key, "name": sc.name, "start": sc.start, "end": sc.end}
        try:
            i0, i1 = st.window_indices(dates, sc.start, sc.end)
        except ValueError as e:
            scenarios.append({**base, "available": False, "reason": str(e)})
            continue
        window = p[i0:i1 + 1, :holdings]
        missing = [tickers[j] for j in range(holdings) if not np.isfinite(window[:, j]).all() and not np.isfinite(betas[j])]
        if missing:
            scenarios.append({**base, "available": False,
                              "reason": f"not enough history to estimate a beta for {', '.join(missing)}"})
            continue
        rets, proxied, days = st.scenario_returns(window, p[i0:i1 + 1, m], np.where(np.isfinite(betas), betas, 0.0))
        rp = st.replay(rets, w)
        scenarios.append({
            **base, "available": True, "from": str(dates[i0]), "to": str(dates[i1]), "days": rp.days,
            "total_return": rp.total_return, "pnl": rp.total_return * value, "max_drawdown": rp.max_drawdown,
            "worst_day": rp.worst_day, "worst_day_date": str(dates[i0 + rp.worst_day_index]), "worst_10d": rp.worst_10d,
            "market_return": float(p[i1, m] / p[i0, m] - 1.0),
            "value": rp.value.tolist(), "path_dates": [str(d) for d in dates[i0:i1 + 1]],
            "holdings": [{"ticker": tickers[j], "return": float(rp.asset_returns[j]),
                          "contribution": float(rp.contributions[j]), "proxied_days": int(days[j])}
                         for j in range(holdings)],
            "proxied": [tickers[j] for j in range(holdings) if proxied[j]],
        })

    # covariance of the most recent common days, for shocks and the reverse stress test
    common = np.isfinite(p).all(axis=1)
    rows = np.flatnonzero(common)
    rows = rows[-(COV_DAYS + 1):]
    if rows.size - 1 < MIN_COV_DAYS:
        raise ValueError(f"only {max(rows.size - 1, 0)} days on which every ticker trades; need {MIN_COV_DAYS} for "
                         "the hypothetical shocks")
    pc = p[rows]
    contiguous = np.diff(rows) == 1                    # returns only between consecutive common days
    rc = _returns(pc)[contiguous]
    cov = np.cov(rc, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)
    wa = np.zeros(len(tickers))
    wa[:holdings] = w
    beta_p = st.portfolio_beta(wa, cov, m)
    rev = st.reverse_stress(wa, cov, m, reverse_loss)

    shock_out = None
    if shocks:
        idx = [tickers.index(t) for t, _ in shocks]
        sh = st.conditional_shock(cov, idx, [s for _, s in shocks], horizon_days)
        pnl = st.shock_pnl(wa, sh, confidence)
        shock_out = {
            "inputs": [{"ticker": t, "shock": s} for t, s in shocks], "horizon_days": horizon_days,
            "confidence": confidence, "expected_return": pnl.expected_return, "pnl": pnl.expected_return * value,
            "stdev": pnl.stdev, "loss_at_confidence": pnl.loss_at_confidence,
            "pnl_at_confidence": -pnl.loss_at_confidence * value,
            "holdings": [{"ticker": tickers[j], "expected_return": float(sh.mean[j]),
                          "contribution": float(pnl.contributions[j])} for j in range(holdings)],
        }
    return {
        "scenarios": scenarios, "shock": shock_out,
        "betas": {tickers[j]: (None if not np.isfinite(betas[j]) else float(betas[j])) for j in range(holdings)},
        "portfolio_beta": beta_p,
        "reverse": {"target_loss": reverse_loss, "market_return": rev, "proxy": PROXY},
        "data": {"start": str(dates[0]), "end": str(dates[-1]), "cov_days": int(rc.shape[0]),
                 "cov_from": str(dates[rows[0]]), "cov_to": str(dates[rows[-1]])},
    }
