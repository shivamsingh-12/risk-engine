import math

import numpy as np
import pytest
from scipy.integrate import quad

from qre.models import heston as h
from qre.pricing import black_scholes as bs

P = h.HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.05)          # Feller: 0.16 < 0.25 fails
P_FELLER = h.HestonParams(kappa=3.0, theta=0.05, sigma=0.3, rho=-0.6, v0=0.04)  # 0.30 >= 0.09 holds


def _heston_1993(p, S, K, r, T):
    """Original two-probability formula C = S P1 - K e^{-rT} P2, an independent implementation."""
    lnS = math.log(S)
    phi = lambda u: np.exp(1j * u * (lnS + r * T)) * h.char_func(u, p, T)     # CF of ln S_T
    phi_mi = phi(-1j)
    def prob(j):
        if j == 1:
            f = lambda u: (np.exp(-1j * u * math.log(K)) * phi(u - 1j) / (1j * u * phi_mi)).real
        else:
            f = lambda u: (np.exp(-1j * u * math.log(K)) * phi(u) / (1j * u)).real
        return 0.5 + quad(f, 1e-10, 300, limit=2000, epsabs=1e-12, epsrel=1e-11)[0] / math.pi
    return S * prob(1) - K * math.exp(-r * T) * prob(2)


def test_char_func_normalization_and_martingale():
    for p in (P, P_FELLER):
        assert abs(h.char_func(0.0, p, 1.0) - 1.0) < 1e-14
        assert abs(h.char_func(-1j, p, 1.0) - 1.0) < 1e-12      # E[S_T / F] = 1


def test_deterministic_variance_limit_is_black_scholes():
    k, th, v0, T = 1.5, 0.04, 0.09, 1.0
    p = h.HestonParams(k, th, 1e-3, 0.0, v0)
    vbar = th + (v0 - th) * (1 - math.exp(-k * T)) / (k * T)     # time-averaged deterministic variance
    for K in (80.0, 100.0, 120.0):
        assert h.heston_price(p, 100, K, 0.03, T) == pytest.approx(bs.price(100, K, 0.03, math.sqrt(vbar), T, "call"), abs=1e-5)


@pytest.mark.parametrize("p", [P, P_FELLER])
@pytest.mark.parametrize("K,T", [(80.0, 0.25), (100.0, 1.0), (125.0, 2.0)])
def test_lewis_matches_original_heston_formula(p, K, T):
    assert h.heston_price(p, 100.0, K, 0.02, T) == pytest.approx(_heston_1993(p, 100.0, K, 0.02, T), abs=1e-7)


def test_put_call_parity():
    c = h.heston_price(P, 100, 90, 0.03, 0.75)
    p = h.heston_price(P, 100, 90, 0.03, 0.75, "put")
    assert c - p == pytest.approx(100 - 90 * math.exp(-0.03 * 0.75), abs=1e-10)


@pytest.mark.parametrize("p", [P, P_FELLER])
def test_monte_carlo_converges_to_semi_closed_form(p):
    for K in (85.0, 100.0, 115.0):
        price, se = h.heston_mc_price(p, 100.0, K, 0.02, 1.0, steps=250, n_paths=200_000, seed=int(K))
        assert abs(price - h.heston_price(p, 100.0, K, 0.02, 1.0)) < 4 * se + 0.02   # + small discretization bias


def test_rho_controls_skew_and_sigma_controls_curvature():
    strikes = [80.0, 100.0, 125.0]
    neg = h.implied_vol_surface(h.HestonParams(2.0, 0.04, 0.5, -0.7, 0.04), 100, 0.0, strikes, [1.0])[0]
    pos = h.implied_vol_surface(h.HestonParams(2.0, 0.04, 0.5, 0.7, 0.04), 100, 0.0, strikes, [1.0])[0]
    assert neg[0] > neg[1] > neg[2]                       # equity-like downside skew
    assert pos[0] < pos[1] < pos[2]
    flat = h.implied_vol_surface(h.HestonParams(2.0, 0.04, 0.05, 0.0, 0.04), 100, 0.0, strikes, [1.0])[0]
    curved = h.implied_vol_surface(h.HestonParams(2.0, 0.04, 0.8, 0.0, 0.04), 100, 0.0, strikes, [1.0])[0]
    smile = lambda iv: 0.5 * (iv[0] + iv[2]) - iv[1]
    assert smile(curved) > smile(flat) + 0.005             # vol of vol makes the smile convex


def test_calibration_recovers_a_noiseless_surface():
    true = h.HestonParams(1.5, 0.05, 0.6, -0.65, 0.035)
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    mats = [0.25, 1.0, 2.0]
    ivs = h.implied_vol_surface(true, 100.0, 0.02, strikes, mats)
    quotes = h.make_quotes(100.0, 0.02, strikes, mats, ivs)
    fit, rmse = h.calibrate(100.0, 0.02, quotes)
    assert rmse < 1e-4                                    # under 0.01 vol points
    assert fit.rho == pytest.approx(true.rho, abs=0.05)
    assert fit.v0 == pytest.approx(true.v0, rel=0.05)
    assert fit.sigma == pytest.approx(true.sigma, rel=0.2)


def test_params_round_trip():
    x = h.x_from_params(P)
    q = h.params_from_x(x)
    assert (q.kappa, q.theta, q.sigma, q.rho, q.v0) == pytest.approx((P.kappa, P.theta, P.sigma, P.rho, P.v0))


def test_feller_flag_and_invalid_inputs():
    assert P_FELLER.feller() and not P.feller()
    with pytest.raises(ValueError):
        h.heston_price(h.HestonParams(2.0, 0.04, 0.5, -1.0, 0.04), 100, 100, 0.0, 1.0)
    with pytest.raises(ValueError):
        h.heston_price(h.HestonParams(-2.0, 0.04, 0.5, -0.5, 0.04), 100, 100, 0.0, 1.0)
    with pytest.raises(ValueError):
        h.heston_price(P, 100, 100, 0.0, 0.0)
