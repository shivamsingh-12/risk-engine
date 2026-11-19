"""Stress testing reference (Phase 20): historical scenarios, hypothetical shocks and reverse stress tests.

VaR answers "how much could we lose on a normal bad day?". A stress test answers "what happens to this
portfolio if a particular crisis happens again?" or "if the market falls 20%?". Three tools:

Historical replay
    Apply each asset's actual daily returns over a past crisis window to today's portfolio, held buy-and-hold
    (fixed numbers of shares from the start of the window). An asset that did not trade then (a recent listing)
    is proxied by beta times a proxy's return each day: r_i,t = beta_i * r_proxy,t, with beta the OLS slope of
    the asset's daily returns on the proxy's over whatever history they share. The proxy keeps the market part
    of the move and drops the asset's own part; Experiment 25 measures what that costs.

Hypothetical shock (conditional Gaussian)
    Shock some assets (or a market index that need not be held) by chosen returns s. Under a multivariate normal
    model with covariance Sigma, every other asset's expected return given the shock is
        E[r_R | r_S = s] = Sigma_RS Sigma_SS^-1 s
    and the remaining uncertainty is Sigma_RR - Sigma_RS Sigma_SS^-1 Sigma_SR. The expected return does not depend
    on the scale (horizon) of Sigma; the spread around it does, so the covariance of daily returns is scaled by the
    shock's horizon in days. Returns are simple returns, combined linearly (no compounding), and an expected
    return below -100% is capped at -100%.

Reverse stress test
    The market move that produces a given expected portfolio loss: with the portfolio's beta to the market
    b = w' Sigma_:,m / Sigma_mm, a market return s gives an expected portfolio return b s, so a loss L needs
    s = -L / b. Undefined (None) when b <= 0: then the portfolio is expected to gain when the market falls.

Matches cpp/include/qre/risk/stress.hpp.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    start: str          # the close on (or the last trading day before) this date is the starting value
    end: str            # the close on (or the last trading day before) this date is the final value


# Market peak to trough (S&P 500) unless noted
SCENARIOS = (
    Scenario("gfc", "Global financial crisis", "2007-10-09", "2009-03-09"),
    Scenario("lehman", "Lehman collapse", "2008-09-12", "2008-11-20"),
    Scenario("downgrade_2011", "US downgrade and euro crisis", "2011-07-22", "2011-10-03"),
    Scenario("china_oil", "China devaluation and oil crash", "2015-08-17", "2016-02-11"),
    Scenario("volmageddon", "February 2018 volatility spike", "2018-01-26", "2018-02-08"),
    Scenario("q4_2018", "Q4 2018 sell-off", "2018-09-20", "2018-12-24"),
    Scenario("covid", "COVID crash", "2020-02-19", "2020-03-23"),
    Scenario("rates_2022", "2022 inflation and rate shock", "2022-01-03", "2022-10-12"),
    Scenario("banks_2023", "2023 regional bank failures", "2023-03-08", "2023-03-17"),   # bank stocks, not the index
    Scenario("tariffs_2025", "April 2025 tariff shock", "2025-04-02", "2025-04-08"),
)


def scenario(key):
    for s in SCENARIOS:
        if s.key == key:
            return s
    raise ValueError(f"unknown scenario '{key}'")


def window_indices(dates, start, end):
    """Rows (i0, i1) of the last dates on or before `start` and `end`. Dates are ISO strings, increasing."""
    if start >= end:
        raise ValueError("scenario start must be before its end")
    i0 = int(np.searchsorted(dates, start, side="right")) - 1
    i1 = int(np.searchsorted(dates, end, side="right")) - 1
    if i0 < 0:
        raise ValueError(f"data starts {dates[0]}, after the scenario start {start}")
    if i1 <= i0:
        raise ValueError(f"no trading days between {start} and {end} in the data")
    return i0, i1


def beta(asset_returns, proxy_returns):
    """OLS slope of asset on proxy over the days where both are finite. Needs at least 20 such days."""
    a = np.asarray(asset_returns, dtype=float)
    p = np.asarray(proxy_returns, dtype=float)
    if a.shape != p.shape:
        raise ValueError("asset and proxy returns must have the same length")
    ok = np.isfinite(a) & np.isfinite(p)
    if ok.sum() < 20:
        raise ValueError("need at least 20 days on which both the asset and the proxy trade to estimate a beta")
    x, y = p[ok] - p[ok].mean(), a[ok] - a[ok].mean()
    vx = float(x @ x)
    if vx <= 0.0:
        raise ValueError("proxy returns have no variance")
    return float(x @ y) / vx


def scenario_returns(prices, proxy, betas):
    """Daily simple returns over a scenario window, filling gaps from the proxy.

    prices: (n + 1) x N closes over the window (NaN where an asset did not trade); proxy: n + 1 closes, all finite;
    betas: N. Day t uses the asset's own return when both closes exist, otherwise beta * proxy return.
    Returns (returns n x N, proxied: bool per asset, proxied day count per asset)."""
    prices = np.asarray(prices, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if prices.ndim != 2 or prices.shape[0] < 2 or proxy.shape != (prices.shape[0],) or betas.shape != (prices.shape[1],):
        raise ValueError("need (n+1) x N prices, n+1 proxy prices and N betas, n >= 1")
    if not np.all(np.isfinite(proxy)) or np.any(proxy <= 0.0):
        raise ValueError("proxy prices must be finite and positive over the whole window")
    if np.any(prices[np.isfinite(prices)] <= 0.0):
        raise ValueError("prices must be positive")
    rp = proxy[1:] / proxy[:-1] - 1.0
    own = prices[1:] / prices[:-1] - 1.0                      # NaN where either close is missing
    have = np.isfinite(own)
    if np.any(~have & ~np.isfinite(betas)[None, :]):
        raise ValueError("an asset with missing prices needs a finite beta")
    out = np.where(have, own, betas[None, :] * rp[:, None])
    days = (~have).sum(axis=0)
    return out, days > 0, days


@dataclass
class Replay:
    total_return: float             # buy-and-hold portfolio return over the window
    contributions: np.ndarray       # w_i * (growth_i - 1); sums to total_return
    asset_returns: np.ndarray       # growth_i - 1
    value: np.ndarray               # portfolio value path, starts at 1 (n + 1 points)
    worst_day: float                # most negative daily portfolio return
    worst_day_index: int            # day (1..n) on which it happened
    worst_10d: float                # most negative 10-day return (the whole window if shorter)
    max_drawdown: float             # largest peak-to-trough fall of the value path, as a positive fraction
    days: int = field(default=0)


def replay(returns, weights):
    """Buy-and-hold replay of daily returns (n x N). Weights are fractions of the starting capital: negative is a
    short position, and whatever does not sum to 1 is cash earning nothing. The value path starts at 1."""
    r = np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if r.ndim != 2 or r.shape[0] < 1 or w.shape != (r.shape[1],):
        raise ValueError("returns must be n x N with n >= 1, and weights N")
    if not np.all(np.isfinite(r)) or np.any(r <= -1.0):
        raise ValueError("returns must be finite and above -100%")
    growth = np.vstack([np.ones(r.shape[1]), np.cumprod(1.0 + r, axis=0)])
    value = 1.0 - w.sum() + growth @ w                      # weights not summing to 1: the rest is cash
    daily = value[1:] / value[:-1] - 1.0
    k = min(10, r.shape[0])
    ten = value[k:] / value[:-k] - 1.0
    peak = np.maximum.accumulate(value)
    asset = growth[-1] - 1.0
    wd = int(np.argmin(daily))
    return Replay(total_return=float(value[-1] - 1.0), contributions=w * asset, asset_returns=asset, value=value,
                  worst_day=float(daily[wd]), worst_day_index=wd + 1, worst_10d=float(ten.min()),
                  max_drawdown=float((1.0 - value / peak).max()), days=int(r.shape[0]))


@dataclass
class Shock:
    mean: np.ndarray                # expected return of every asset given the shock (shocked assets: the shock)
    cond_cov: np.ndarray            # covariance left given the shock, at the shock's horizon
    shocked: np.ndarray             # indices of the shocked assets


def conditional_shock(cov, shocked, shocks, horizon_days=1.0):
    """Gaussian conditioning on r_S = s. cov is the covariance of daily simple returns."""
    cov = np.asarray(cov, dtype=float)
    idx = np.asarray(shocked, dtype=int)
    s = np.asarray(shocks, dtype=float)
    n = cov.shape[0]
    if cov.ndim != 2 or cov.shape != (n, n) or not np.allclose(cov, cov.T, rtol=0, atol=1e-14):
        raise ValueError("cov must be a symmetric square matrix")
    if idx.ndim != 1 or idx.size == 0 or idx.shape != s.shape or len(set(idx.tolist())) != idx.size:
        raise ValueError("give one shock per shocked asset, at least one, no repeats")
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError("shocked index out of range")
    if not np.all(np.isfinite(s)) or np.any(s <= -1.0):
        raise ValueError("shocks must be finite returns above -100%")
    if not horizon_days > 0:
        raise ValueError("horizon must be positive")
    sss = cov[np.ix_(idx, idx)]
    try:
        chol = np.linalg.cholesky(sss)
    except np.linalg.LinAlgError:
        raise ValueError("the shocked assets' covariance is singular (two shocked assets move identically?)") from None
    a = cov[:, idx]                                          # Sigma_:,S
    solve = lambda b: np.linalg.solve(chol.T, np.linalg.solve(chol, b))   # noqa: E731
    mean = a @ solve(s)
    mean[idx] = s                                            # exact, not up to rounding
    mean = np.maximum(mean, -1.0)
    cond = (cov - a @ solve(a.T)) * horizon_days
    cond = 0.5 * (cond + cond.T)
    cond[np.ix_(idx, idx)] = 0.0
    cond[idx, :] = 0.0
    cond[:, idx] = 0.0
    return Shock(mean=mean, cond_cov=cond, shocked=idx)


@dataclass
class ShockPnl:
    expected_return: float          # w' mean
    stdev: float                    # sqrt(w' C w): the spread left around it
    loss_at_confidence: float       # -expected + z * stdev: a bad outcome given the shock
    contributions: np.ndarray       # w_i * mean_i


def shock_pnl(weights, shock, confidence=0.99):
    w = np.asarray(weights, dtype=float)
    if w.shape != shock.mean.shape:
        raise ValueError("one weight per asset in the covariance")
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1)")
    er = float(w @ shock.mean)
    sd = float(np.sqrt(max(w @ shock.cond_cov @ w, 0.0)))
    return ShockPnl(expected_return=er, stdev=sd, loss_at_confidence=-er + float(norm.ppf(confidence)) * sd,
                    contributions=w * shock.mean)


def portfolio_beta(weights, cov, market):
    cov = np.asarray(cov, dtype=float)
    if not 0 <= market < cov.shape[0]:
        raise ValueError("market index out of range")
    if cov[market, market] <= 0.0:
        raise ValueError("market variance must be positive")
    return float(np.asarray(weights, dtype=float) @ cov[:, market]) / float(cov[market, market])


def reverse_stress(weights, cov, market, target_loss):
    """Market return that gives an expected portfolio loss of target_loss (a positive fraction). None if the
    portfolio's beta to the market is not positive."""
    if not target_loss > 0:
        raise ValueError("target loss must be positive")
    b = portfolio_beta(weights, cov, market)
    return None if b <= 0.0 else -target_loss / b
