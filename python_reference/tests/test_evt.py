import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import genpareto, norm, t as student_t

from qre.risk import backtest as bt
from qre.risk import evt
from qre.risk import garch


# ---------------------------------------------------------------- GPD fit
@pytest.mark.parametrize("xi,beta", [(0.3, 1.0), (0.0, 2.0), (-0.2, 0.5)])
def test_gpd_fit_matches_scipy_likelihood(xi, beta):
    y = genpareto.rvs(xi, scale=beta, size=3000, random_state=int(10 * (xi + 1)))
    ours = evt.fit_gpd(y)
    c, _, scale = genpareto.fit(y, floc=0)
    ll = lambda a, b: genpareto.logpdf(y, a, scale=b).sum()
    assert ll(*ours) >= ll(c, scale) - 1e-6          # at least as good an optimum as scipy's
    assert ours[0] == pytest.approx(c, abs=5e-3)
    assert ours[1] == pytest.approx(scale, rel=5e-3)


def test_gpd_fit_recovers_parameters():
    y = genpareto.rvs(0.25, scale=1.5, size=20000, random_state=1)
    xi, beta = evt.fit_gpd(y)
    assert xi == pytest.approx(0.25, abs=0.03)
    assert beta == pytest.approx(1.5, rel=0.04)


def test_gpd_var_es_is_exact_for_a_gpd():
    """If the whole distribution is GPD (u = 0, every observation exceeds), the formulas are exact."""
    xi, beta, p = 0.25, 1.0, 0.99
    var, es = evt.gpd_var_es(0.0, xi, beta, 100, 100, p)
    assert var == pytest.approx(genpareto.ppf(p, xi, scale=beta), rel=1e-12)
    tail = quad(lambda x: x * genpareto.pdf(x, xi, scale=beta), var, np.inf)[0] / (1 - p)
    assert es == pytest.approx(tail, rel=1e-8)
    v0, e0 = evt.gpd_var_es(0.0, 0.0, 2.0, 100, 100, p)             # exponential
    assert v0 == pytest.approx(-2.0 * math.log(1 - p))
    assert e0 == pytest.approx(v0 + 2.0)
    with pytest.raises(ValueError):
        evt.gpd_var_es(0.0, 1.2, 1.0, 100, 100, p)


def test_pot_extrapolates_student_t_tail():
    nu = 4.0
    x = student_t.rvs(nu, size=20000, random_state=2)
    var, es, xi, beta, u, n_u = evt.pot_var_es(x, 0.999, threshold_quantile=0.95)
    assert xi == pytest.approx(1 / nu, abs=0.12)
    assert var == pytest.approx(student_t.ppf(0.999, nu), rel=0.1)
    assert n_u == pytest.approx(1000, abs=1)
    assert es > var


def test_gpd_fit_respects_xi_bounds_on_tiny_samples():
    """15 exceedances from a moderately heavy tail can push the unrestricted MLE past xi = 1."""
    hits = 0
    for seed in range(40):
        y = genpareto.rvs(0.4, scale=1.0, size=15, random_state=seed)
        xi, beta = evt.fit_gpd(y)
        assert evt.XI_MIN < xi < evt.XI_MAX and beta > 0
        hits += evt.gpd_at_bound(xi)
    assert hits >= 1                                  # the bound genuinely binds sometimes
    assert not evt.gpd_at_bound(evt.fit_gpd(genpareto.rvs(0.2, size=5000, random_state=1))[0])


def test_pot_validation():
    with pytest.raises(ValueError):
        evt.pot_var_es(np.arange(10.0), 0.99)
    with pytest.raises(ValueError):
        evt.pot_var_es(np.arange(100.0), 0.9, threshold_quantile=0.95)
    with pytest.raises(ValueError):
        evt.fit_gpd([1.0, 2.0, -1.0, 3.0, 4.0])


# ---------------------------------------------------------------- Hill
def test_hill_on_pareto_tail():
    alpha = 3.0
    x = (1.0 - np.random.default_rng(3).random(50000)) ** (-1.0 / alpha)   # Pareto, xi = 1/3
    assert evt.hill(x, 2000) == pytest.approx(1 / alpha, rel=0.06)
    with pytest.raises(ValueError):
        evt.hill(x, 0)
    with pytest.raises(ValueError):
        evt.hill(-np.abs(x), 10)


# ---------------------------------------------------------------- conditional EVT
def test_garch_evt_reacts_to_volatility():
    r = garch.simulate_garch(1000, 2e-6, 0.08, 0.90, seed=4, nu=5.0)
    v, e = evt.garch_evt_var_es(r, 0.99)
    assert e > v > 0
    shocked = r.copy()
    shocked[-1] = -0.08
    v2, _ = evt.garch_evt_var_es(shocked, 0.99)
    assert v2 > 1.5 * v


def test_garch_evt_backtest_has_no_look_ahead():
    rng = np.random.default_rng(5)
    rets = rng.standard_normal((200, 2)) * 0.01
    p = 100.0 * np.vstack([np.ones(2), np.cumprod(1.0 + rets, axis=0)])
    w = np.array([0.6, 0.4])
    s = bt.rolling_backtest(p, w, window=150, method="garch_evt")
    future = p.copy()
    future[181:] *= np.linspace(0.5, 3.0, future.shape[0] - 181)[:, None]
    s2 = bt.rolling_backtest(future, w, window=150, method="garch_evt")
    i = int(np.flatnonzero(s.rows == 181)[0])
    assert np.array_equal(s.var[: i + 1], s2.var[: i + 1])


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_coverage_rate_is_near_nominal():
    """A 90% interval should miss about 10% of the time, so test the rate over many samples, not one sample.
    The percentile bootstrap under-covers somewhat in the far tail (about 85% here), hence the lower bound."""
    z = norm.ppf(0.99)
    es_true = norm.pdf(z) / 0.01
    cover_v = cover_e = 0
    for s in range(100):
        x = np.random.default_rng(1000 + s).standard_normal(2000)
        (vlo, vhi), (elo, ehi) = evt.bootstrap_var_es(x, 0.99, n_boot=300, seed=s)
        assert vlo < vhi and elo < ehi
        cover_v += vlo < z < vhi
        cover_e += elo < es_true < ehi
    assert 0.75 <= cover_v / 100 <= 0.97
    assert 0.72 <= cover_e / 100 <= 0.97


def test_block_bootstrap_is_wider_for_clustered_volatility():
    r = garch.simulate_garch(1500, 1e-6, 0.10, 0.89, seed=8)
    iid = evt.bootstrap_var_es(-r, 0.99, n_boot=800, block=1, seed=9)[0]
    blk = evt.bootstrap_var_es(-r, 0.99, n_boot=800, block=25, seed=9)[0]
    assert (blk[1] - blk[0]) > (iid[1] - iid[0])


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        evt.bootstrap_var_es(np.arange(10.0), 0.99)
    with pytest.raises(ValueError):
        evt.bootstrap_var_es(np.arange(100.0), 0.99, block=0)
