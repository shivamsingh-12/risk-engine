import math

import numpy as np
import pytest
from scipy.stats import norm

from qre.credit import merton as m
from qre.pricing import black_scholes as bs

V, D, R, S, T = 100.0, 70.0, 0.03, 0.25, 2.0


def test_equity_is_a_call_and_debt_is_riskless_minus_a_put():
    e = m.equity_value(V, D, R, S, T)
    assert e == pytest.approx(bs.price(V, D, R, S, T, "call"), rel=1e-14)
    b = m.debt_value(V, D, R, S, T)
    assert b == pytest.approx(D * math.exp(-R * T) - bs.price(V, D, R, S, T, "put"), rel=1e-13)
    assert e + b == pytest.approx(V, rel=1e-15)


def test_spread_pd_and_recovery_identities():
    s = m.credit_spread(V, D, R, S, T)
    b = m.debt_value(V, D, R, S, T)
    assert s == pytest.approx(-math.log(b / D) / T - R, rel=1e-12)
    pd, rec = m.risk_neutral_pd(V, D, R, S, T), m.recovery_rate(V, D, R, S, T)
    # B = e^{-rT} (D (1 - PD) + PD * recovery * D)
    assert b == pytest.approx(math.exp(-R * T) * D * (1 - pd + pd * rec), rel=1e-13)
    assert 0 < rec < 1 and s > 0
    assert m.physical_pd(V, D, R, S, T) == pytest.approx(pd, rel=1e-14)          # mu = r gives the Q measure


def test_pd_and_spread_by_monte_carlo():
    z = np.random.default_rng(1).standard_normal(1_000_000)
    vt = V * np.exp((R - 0.5 * S * S) * T + S * math.sqrt(T) * z)
    assert np.mean(vt < D) == pytest.approx(m.risk_neutral_pd(V, D, R, S, T), abs=1e-3)
    assert math.exp(-R * T) * np.mean(np.minimum(vt, D)) == pytest.approx(m.debt_value(V, D, R, S, T), rel=2e-3)


def test_spread_term_structure_shapes():
    low = [m.credit_spread(100, 40, R, 0.2, t) for t in (0.05, 1, 5, 20)]
    high = [m.credit_spread(100, 90, R, 0.3, t) for t in (0.05, 0.5, 2, 10, 30)]
    assert low[0] < 1e-8 and low[0] < low[1] < low[2]            # safe firm: ~0 at the short end, rising
    assert np.argmax(high) not in (0, 4)                         # risky firm: humped
    assert m.credit_spread(100, 120, R, 0.3, 0.05) > m.credit_spread(100, 120, R, 0.3, 5)   # distressed: falling


def test_solve_assets_recovers_v_and_sigma():
    for v, d, s, t in ((100, 70, 0.25, 2), (100, 30, 0.15, 1), (100, 95, 0.4, 0.5), (50, 80, 0.3, 1)):
        e, se = m.equity_value(v, d, R, s, t), m.equity_volatility(v, d, R, s, t)
        vv, ss = m.solve_assets(e, se, d, R, t)
        assert vv == pytest.approx(v, rel=1e-10) and ss == pytest.approx(s, rel=1e-9)
    with pytest.raises(ValueError):
        m.solve_assets(-1, 0.3, 50, R, 1)


def _firm_path(n=251, mu=0.08, sigma=0.25, d=60.0, seed=1):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    v = 100 * np.exp(np.cumsum(np.r_[0, (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n - 1)]))
    return v, np.array([m.equity_value(x, d, R, sigma, 1.0) for x in v])


def test_kmv_iteration_is_a_fixed_point_near_the_truth():
    v, e = _firm_path()
    assert np.array([m.asset_value(x, 60.0, R, 0.25, 1.0) for x in e]) == pytest.approx(v, rel=1e-10)
    k = m.kmv_iterative(e, 60.0, R)
    assert k.sigma == pytest.approx(np.std(np.diff(np.log(k.assets)), ddof=1) * math.sqrt(252), rel=1e-8)
    assert k.sigma == pytest.approx(0.25, abs=0.035)               # one year of data: sampling error only
    assert 2 <= k.iterations < 50
    assert k.pd == pytest.approx(m.physical_pd(k.assets[-1], 60.0, k.mu, k.sigma, 1.0))
    with pytest.raises(ValueError):
        m.kmv_iterative(e[:5], 60.0, R)


def test_kmv_beats_the_static_two_equation_estimate_after_a_drawdown():
    """When leverage changed over the window, a historical sigma_E is out of date; the iteration is not fooled."""
    errs_kmv, errs_naive = [], []
    for seed in range(20):
        v, e = _firm_path(mu=-0.3, seed=100 + seed)
        se = np.std(np.diff(np.log(e)), ddof=1) * math.sqrt(252)
        errs_naive.append(m.solve_assets(e[-1], se, 60.0, R, 1.0)[1] - 0.25)
        errs_kmv.append(m.kmv_iterative(e, 60.0, R).sigma - 0.25)
    assert np.sqrt(np.mean(np.square(errs_kmv))) < 0.5 * np.sqrt(np.mean(np.square(errs_naive)))


def test_black_cox_formula_and_ordering():
    for v, k, mu, s, t in ((100, 70, 0.08, 0.25, 2), (100, 90, -0.02, 0.3, 1)):
        p = m.black_cox_pd(v, k, mu, s, t)
        mc, se = m.first_passage_mc(v, k, mu, s, t, n_paths=200_000, seed=3)
        assert p == pytest.approx(mc, abs=4 * se + 1e-3)
        assert p >= m.physical_pd(v, k, mu, s, t)
    # zero drift: reflection principle, P = 2 N(b / (sigma sqrt T)) adjusted by the -sigma^2/2 drift term
    nu = -0.5 * 0.2 ** 2
    b = math.log(0.8)
    exact = norm.cdf((b - nu) / 0.2) + 0.8 ** (2 * nu / 0.04) * norm.cdf((b + nu) / 0.2)
    assert m.black_cox_pd(1.0, 0.8, 0.0, 0.2, 1.0) == pytest.approx(exact, rel=1e-14)
    with pytest.raises(ValueError):
        m.black_cox_pd(100, 120, 0.05, 0.2, 1)
