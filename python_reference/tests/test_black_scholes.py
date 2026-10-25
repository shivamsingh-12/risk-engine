import math

import pytest

from qre.pricing import black_scholes as bs

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0


def test_known_values():
    assert bs.price(S, K, R, SIG, T, "call") == pytest.approx(10.450583572185565, abs=1e-8)
    assert bs.price(S, K, R, SIG, T, "put") == pytest.approx(5.573526022256971, abs=1e-8)


@pytest.mark.parametrize("spot,strike,rate,vol,time", [
    (100, 100, 0.05, 0.2, 1.0),
    (80, 100, 0.01, 0.4, 0.5),
    (120, 90, 0.03, 0.15, 2.0),
    (50, 55, 0.0, 0.3, 0.25),
])
def test_put_call_parity(spot, strike, rate, vol, time):
    c = bs.price(spot, strike, rate, vol, time, "call")
    p = bs.price(spot, strike, rate, vol, time, "put")
    assert c - p == pytest.approx(spot - strike * math.exp(-rate * time), abs=1e-10)


def test_deep_itm_and_otm_call():
    assert bs.price(500, 100, R, SIG, T, "call") == pytest.approx(500 - 100 * math.exp(-R * T), abs=1e-6)
    assert bs.price(20, 100, R, SIG, T, "call") == pytest.approx(0.0, abs=1e-10)


def test_short_expiry_approaches_intrinsic():
    assert bs.price(110, 100, R, SIG, 1e-8, "call") == pytest.approx(10.0, abs=1e-4)
    assert bs.price(90, 100, R, SIG, 1e-8, "call") == pytest.approx(0.0, abs=1e-6)


def _bump(fn, which, h, kind=None):
    args = dict(spot=S, strike=K, rate=R, vol=SIG, time=T)
    up, dn = dict(args), dict(args)
    up[which] += h
    dn[which] -= h
    if kind:
        return (fn(**up, kind=kind) - fn(**dn, kind=kind)) / (2 * h)
    return (fn(**up) - fn(**dn)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_finite_differences(kind):
    assert bs.delta(S, K, R, SIG, T, kind) == pytest.approx(_bump(bs.price, "spot", 1e-4, kind), abs=1e-6)
    assert bs.vega(S, K, R, SIG, T) == pytest.approx(_bump(bs.price, "vol", 1e-5, kind), abs=1e-5)
    assert bs.rho(S, K, R, SIG, T, kind) == pytest.approx(_bump(bs.price, "rate", 1e-5, kind), abs=1e-4)
    # theta is dV/dt = -dV/dT
    assert bs.theta(S, K, R, SIG, T, kind) == pytest.approx(-_bump(bs.price, "time", 1e-5, kind), abs=1e-4)


def test_gamma_matches_second_difference():
    h = 1e-3
    fd = (bs.price(S + h, K, R, SIG, T) - 2 * bs.price(S, K, R, SIG, T) + bs.price(S - h, K, R, SIG, T)) / h**2
    assert bs.gamma(S, K, R, SIG, T) == pytest.approx(fd, abs=1e-5)


def test_delta_put_call_relation():
    assert bs.delta(S, K, R, SIG, T, "call") - bs.delta(S, K, R, SIG, T, "put") == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
def test_implied_vol_round_trip(kind, vol):
    p = bs.price(S, K, R, vol, T, kind)
    assert bs.implied_vol(p, S, K, R, T, kind) == pytest.approx(vol, abs=1e-7)


def test_implied_vol_rejects_arbitrage_prices():
    with pytest.raises(ValueError):
        bs.implied_vol(0.0, S, K, R, T, "call")
    with pytest.raises(ValueError):
        bs.implied_vol(S + 1.0, S, K, R, T, "call")


@pytest.mark.parametrize("bad", [
    dict(spot=-1, strike=K, rate=R, vol=SIG, time=T),
    dict(spot=S, strike=0, rate=R, vol=SIG, time=T),
    dict(spot=S, strike=K, rate=R, vol=0, time=T),
    dict(spot=S, strike=K, rate=R, vol=SIG, time=0),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bs.price(**bad)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        bs.price(S, K, R, SIG, T, "straddle")
