"""Phase 21: portfolio construction, component VaR/ES and multi-day VaR. The fixtures at the bottom print the values
hard-coded in cpp/tests/test_portfolio.cpp."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import norm

from qre.portfolio import construction as pc
from qre.risk import component as cr
from qre.risk import garch as gh
from qre.risk import horizon as hz


def fixture_cov():
    """Five assets: two correlated equities, a bond that hedges them, gold and a high-volatility asset."""
    vol = np.array([0.012, 0.015, 0.005, 0.010, 0.025])
    corr = np.array([[1.0, 0.8, -0.3, 0.1, 0.6],
                     [0.8, 1.0, -0.2, 0.1, 0.7],
                     [-0.3, -0.2, 1.0, 0.2, -0.2],
                     [0.1, 0.1, 0.2, 1.0, 0.1],
                     [0.6, 0.7, -0.2, 0.1, 1.0]])
    return corr * np.outer(vol, vol)


def fixture_returns(t=300, n=5):
    """Deterministic, non-normal daily returns with a few large losses (no RNG, so both languages build the same)."""
    i = np.arange(t)[:, None]
    j = np.arange(n)[None, :]
    r = 0.01 * np.sin(0.37 * i * (j + 1) + j) + 0.004 * np.cos(1.3 * i + 2.0 * j)
    r[::37] -= 0.03 * (1 + np.arange(n)) / n
    return r


def slsqp(q, c, lo, hi):
    n = q.shape[0]
    s = 1e4
    res = minimize(lambda w: s * (0.5 * w @ q @ w - c @ w), np.full(n, 1 / n), jac=lambda w: s * (q @ w - c),
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   bounds=[(lo, hi)] * n, method="SLSQP", options={"ftol": 1e-20, "maxiter": 2000})
    return res.x


def test_projection_is_the_nearest_feasible_point():
    lo, hi = pc.bounds(4, -0.2, 0.6)
    v = np.array([0.9, 0.5, -0.7, 0.1])
    w = pc.project(v, lo, hi)
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.all(w >= lo - 1e-15) and np.all(w <= hi + 1e-15)
    res = minimize(lambda x: ((x - v) ** 2).sum(), np.full(4, 0.25), method="SLSQP", bounds=[(-0.2, 0.6)] * 4,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-16})
    assert w == pytest.approx(res.x, abs=1e-7)
    assert pc.project(np.array([0.1, 0.2, 0.3, 0.4]), lo, hi) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        pc.bounds(3, 0.0, 0.3)                          # 3 x 30% cannot reach 100%
    with pytest.raises(ValueError):
        pc.bounds(3, 0.5, 1.0)                          # 3 x 50% minimum exceeds 100%


def test_unconstrained_min_variance_matches_closed_form():
    s = fixture_cov()
    one = np.linalg.solve(s, np.ones(5))
    closed = one / one.sum()                            # has a short in asset 5
    q = pc.min_variance(s, -10.0, 10.0)
    assert q.converged and q.weights == pytest.approx(closed, abs=1e-10)


@pytest.mark.parametrize("lo,hi", [(0.0, 1.0), (-0.3, 1.0), (0.05, 0.4)])
def test_min_and_mean_variance_match_slsqp(lo, hi):
    s = fixture_cov()
    mu = np.array([0.0006, 0.0008, 0.0002, 0.0003, 0.0012])
    q = pc.min_variance(s, lo, hi)
    assert q.converged and q.weights == pytest.approx(slsqp(s, np.zeros(5), lo, hi), abs=1e-7)
    m = pc.mean_variance(mu, s, 4.0, lo, hi)
    assert m.converged and m.weights == pytest.approx(slsqp(4.0 * s, mu, lo, hi), abs=1e-7)
    for w in (q.weights, m.weights):
        assert w.sum() == pytest.approx(1.0, abs=1e-13)
        assert np.all(w >= lo - 1e-13) and np.all(w <= hi + 1e-13)


def test_mean_variance_tends_to_min_variance_as_risk_aversion_grows():
    s = fixture_cov()
    mu = np.array([0.0006, 0.0008, 0.0002, 0.0003, 0.0012])
    big = pc.mean_variance(mu, s, 1e7).weights
    assert big == pytest.approx(pc.min_variance(s).weights, abs=1e-3)
    low = pc.mean_variance(mu, s, 0.1).weights               # almost risk-neutral: everything in the best asset
    assert low[4] == pytest.approx(1.0, abs=1e-9)


def test_risk_parity_equalizes_and_respects_budgets():
    s = fixture_cov()
    rp = pc.risk_parity(s)
    assert rp.converged and np.all(rp.weights > 0)
    assert pc.risk_contribution_shares(rp.weights, s) == pytest.approx(np.full(5, 0.2), abs=1e-11)
    b = np.array([0.4, 0.1, 0.2, 0.2, 0.1])
    rb = pc.risk_parity(s, b)
    assert pc.risk_contribution_shares(rb.weights, s) == pytest.approx(b, abs=1e-11)
    diag = np.diag(np.diag(s))                              # uncorrelated: risk parity = inverse volatility
    assert pc.risk_parity(diag).weights == pytest.approx(pc.inverse_vol(diag), abs=1e-12)


def test_parametric_components_are_euler_and_match_finite_differences():
    s = fixture_cov()
    w = np.array([0.5, 0.4, 0.1, 0.0, 0.2])                  # equity-heavy, with a small bond position
    c = cr.parametric_components(w, s, 0.99)
    sig = np.sqrt(w @ s @ w)
    assert c.var == pytest.approx(norm.ppf(0.99) * sig)
    assert c.component_var.sum() == pytest.approx(c.var, rel=1e-14)
    assert c.component_es.sum() == pytest.approx(c.es, rel=1e-14)
    eps = 1e-7
    for i in range(5):
        dw = np.zeros(5)
        dw[i] = eps
        fd = (cr.parametric_components(w + dw, s, 0.99).var - cr.parametric_components(w - dw, s, 0.99).var) / (2 * eps)
        assert w[i] * fd == pytest.approx(c.component_var[i], abs=1e-9)
    assert c.component_var[2] < 0                            # the bond hedges the equities
    assert c.component_var[3] == 0.0


def test_historical_components_sum_exactly_and_follow_the_tail():
    r = fixture_returns()
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    c = cr.historical_components(r, w, 0.975)
    loss = -(r @ w)
    tail = loss >= np.quantile(loss, 0.975)
    assert c.tail_days == tail.sum()
    assert c.es == pytest.approx(loss[tail].mean())
    assert c.component_es.sum() == pytest.approx(c.es, rel=1e-13)
    assert c.component_var.sum() == pytest.approx(c.var, rel=1e-13)
    assert c.component_es == pytest.approx((-(r * w))[tail].mean(axis=0))


def test_overlapping_and_sqrt_time_by_hand():
    r = fixture_returns()[:, 0]
    g = np.cumprod(1 + r)
    g = np.concatenate([[1.0], g])
    ten = g[10:] / g[:-10] - 1
    assert hz.overlapping(r, 0.99, 10) == pytest.approx((np.quantile(-ten, 0.99), (-ten)[-ten >= np.quantile(-ten, 0.99)].mean()))
    one = hz.sqrt_time(r, 0.99, 1)
    assert hz.sqrt_time(r, 0.99, 10) == pytest.approx((one[0] * np.sqrt(10), one[1] * np.sqrt(10)))


def test_garch_variance_sum_and_fhs_against_monte_carlo():
    fit = gh.GarchFit(omega=2e-6, alpha=0.08, beta=0.9, long_run_var=1e-4, loglik=0.0, converged=True)
    p = 0.98
    s1 = 4e-4
    v = sum(1e-4 + (s1 - 1e-4) * p ** k for k in range(10))   # expected daily variances, summed
    assert hz.garch_variance_sum(fit, s1, 10) == pytest.approx(v, rel=1e-13)
    # FHS with normal shocks on a long simulated series approaches the GARCH-normal answer
    r = gh.simulate_garch(4000, 2e-6, 0.08, 0.9, seed=3)
    f = gh.fit_garch(r)
    a = hz.garch_normal(r, 0.99, 10, fit=f)
    b = hz.fhs_garch(r, 0.99, 10, n_paths=40_000, seed=1, fit=f)
    assert b[0] == pytest.approx(a[0], rel=0.08)
    with pytest.raises(ValueError):
        hz.fhs_garch_paths(r, 10, np.zeros((5, 9), dtype=int), fit=f)


def test_cross_language_fixture():
    """Values printed here are hard-coded in cpp/tests/test_portfolio.cpp (same inputs)."""
    s = fixture_cov()
    mu = np.array([0.0006, 0.0008, 0.0002, 0.0003, 0.0012])
    mv = pc.min_variance(s, -0.3, 1.0).weights
    mean_v = pc.mean_variance(mu, s, 4.0, 0.0, 0.4).weights
    rp = pc.risk_parity(s).weights
    r = fixture_returns()
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    hc = cr.historical_components(r, w, 0.975)
    pr = r @ w
    fit = gh.GarchFit(omega=1e-6, alpha=0.1, beta=0.85, long_run_var=2e-5, loglik=0.0, converged=True)
    draws = (np.arange(200)[:, None] * 7 + np.arange(10)[None, :] * 13) % 300
    paths = hz.fhs_garch_paths(pr, 10, draws, fit=fit)
    gn = hz.garch_normal(pr, 0.99, 10, fit=fit)
    ov = hz.overlapping(pr, 0.99, 10)
    vals = dict(mv=mv, mean_v=mean_v, rp=rp, hc_var=hc.component_var, hc_es=hc.component_es,
                paths=paths[[0, 57, 199]], gn=gn, ov=ov)
    for k, x in vals.items():
        print(k, repr(np.asarray(x).tolist()))
    assert mv == pytest.approx([0.2262963262582542, -0.029221372460156807, 0.7447291080105434, 0.05872227742542246,
                                -0.000526339234063241], abs=1e-10)
    assert mean_v == pytest.approx([0.039268697038675165, 0.3579024719954555, 0.3000822992127668,
                                    0.06121932074202628, 0.2415272110110763], abs=1e-10)
    assert rp == pytest.approx([0.13483964337452994, 0.09597423755809534, 0.5265434533894444, 0.17992690553453836,
                                0.06271576014339207], abs=1e-12)
    assert hc.component_var == pytest.approx([0.0013628920217820958, 0.0017305886874910519, 0.00306617267018189,
                                              0.003898189967704644, 0.002424805224873086], rel=1e-12)
    assert hc.component_es == pytest.approx([0.0012643583704369536, 0.002507691747119058, 0.0037314219780306065,
                                             0.004632851697403739, 0.002856859006729886], rel=1e-12)
    assert paths[[0, 57, 199]] == pytest.approx([-0.03385657136014031, -0.002744770095467697, 0.009462224924436935],
                                                rel=1e-12)
    assert gn == pytest.approx((0.034724226930752604, 0.039782314777197375), rel=1e-12)
    assert ov == pytest.approx((0.031032780714151154, 0.03114560415747854), rel=1e-12)
