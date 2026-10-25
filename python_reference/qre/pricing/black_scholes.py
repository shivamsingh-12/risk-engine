"""Phase 5 reference: Black-Scholes pricing, Greeks and implied volatility.

European options, no dividends. Conventions (match the C++ implementation):
    vega  : per 1.00 change in volatility
    theta : per year (calendar-time derivative, dV/dt)
    rho   : per 1.00 change in the rate
"""
import math

from scipy.optimize import brentq

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    return 0.5 * math.erfc(-x / SQRT_2)


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _validate(spot, strike, vol, time):
    if spot <= 0.0:
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if vol <= 0.0:
        raise ValueError("vol must be positive")
    if time <= 0.0:
        raise ValueError("time must be positive")


def _check_kind(kind):
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")


def d1_d2(spot, strike, rate, vol, time):
    sqrt_t = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def price(spot, strike, rate, vol, time, kind="call"):
    _validate(spot, strike, vol, time)
    _check_kind(kind)
    d1, d2 = d1_d2(spot, strike, rate, vol, time)
    disc = math.exp(-rate * time)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def delta(spot, strike, rate, vol, time, kind="call"):
    _validate(spot, strike, vol, time)
    _check_kind(kind)
    d1, _ = d1_d2(spot, strike, rate, vol, time)
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1.0


def gamma(spot, strike, rate, vol, time):
    _validate(spot, strike, vol, time)
    d1, _ = d1_d2(spot, strike, rate, vol, time)
    return _norm_pdf(d1) / (spot * vol * math.sqrt(time))


def vega(spot, strike, rate, vol, time):
    _validate(spot, strike, vol, time)
    d1, _ = d1_d2(spot, strike, rate, vol, time)
    return spot * _norm_pdf(d1) * math.sqrt(time)


def theta(spot, strike, rate, vol, time, kind="call"):
    _validate(spot, strike, vol, time)
    _check_kind(kind)
    d1, d2 = d1_d2(spot, strike, rate, vol, time)
    disc = math.exp(-rate * time)
    decay = -spot * _norm_pdf(d1) * vol / (2.0 * math.sqrt(time))
    if kind == "call":
        return decay - rate * strike * disc * _norm_cdf(d2)
    return decay + rate * strike * disc * _norm_cdf(-d2)


def rho(spot, strike, rate, vol, time, kind="call"):
    _validate(spot, strike, vol, time)
    _check_kind(kind)
    _, d2 = d1_d2(spot, strike, rate, vol, time)
    disc = math.exp(-rate * time)
    if kind == "call":
        return strike * time * disc * _norm_cdf(d2)
    return -strike * time * disc * _norm_cdf(-d2)


def implied_vol(market_price, spot, strike, rate, time, kind="call",
                lower=1e-6, upper=5.0, tol=1e-10):
    """Solve price(vol) = market_price for vol using Brent's method."""
    _validate(spot, strike, 1.0, time)
    _check_kind(kind)
    disc = math.exp(-rate * time)
    if kind == "call":
        lo_bound, hi_bound = max(spot - strike * disc, 0.0), spot
    else:
        lo_bound, hi_bound = max(strike * disc - spot, 0.0), strike * disc
    if not (lo_bound < market_price < hi_bound):
        raise ValueError("market price violates no-arbitrage bounds")

    def objective(v):
        return price(spot, strike, rate, v, time, kind) - market_price

    return brentq(objective, lower, upper, xtol=tol)
