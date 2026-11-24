"""Phase 22: incremental risk. The fixture at the bottom prints values hard-coded in cpp/tests/test_stream.cpp."""
import numpy as np
import pytest
from scipy.stats import norm

from qre.risk.var_es import var_es_from_losses
from qre.stream.incremental import IncrementalRisk, quantile_linear


def fixture():
    t, n = 250, 4
    i = np.arange(t)[:, None]
    j = np.arange(n)[None, :]
    r = 0.01 * np.sin(0.37 * i * (j + 1) + j) + 0.004 * np.cos(1.3 * i + 2.0 * j)
    r[::37] -= 0.03 * (1 + np.arange(n)) / n
    shares = np.array([100.0, -40.0, 250.0, 60.0])             # one short position
    prices = np.array([180.0, 95.0, 42.0, 310.0])
    return shares, prices, np.cov(r, rowvar=False), r


def ticks(prices, k_max=500):
    """Deterministic tick sequence: asset k mod N, price moved by a few percent."""
    n = prices.size
    return [(k % n, prices[k % n] * (1.0 + 0.03 * np.sin(0.7 * k + k % n))) for k in range(k_max)]


def full(shares, prices, cov, r, c):
    x = shares * prices
    sig = np.sqrt(x @ cov @ x)
    var_h, es_h = var_es_from_losses(-(r @ x), c)
    return x.sum(), sig, norm.ppf(c) * sig, var_h, es_h


def test_quantile_by_selection_matches_numpy():
    x = np.random.default_rng(0).standard_t(4, size=257)
    for q in (0.5, 0.95, 0.975, 0.99, 0.999):
        assert quantile_linear(x, q) == np.quantile(x, q)


def test_incremental_equals_full_recompute_after_every_tick():
    shares, prices, cov, r = fixture()
    risk = IncrementalRisk(shares, prices, cov, r, 0.99)
    p = prices.copy()
    for k, (i, price) in enumerate(ticks(prices)):
        risk.on_tick(i, price)
        p[i] = price
        if k % 50 == 0:
            s = risk.snapshot()
            value, sig, var_p, var_h, es_h = full(shares, p, cov, r, 0.99)
            assert s.value == pytest.approx(value, rel=1e-13)
            assert s.sigma == pytest.approx(sig, rel=1e-12)
            assert s.var_parametric == pytest.approx(var_p, rel=1e-12)
            assert s.var_historical == pytest.approx(var_h, rel=1e-12)
            assert s.es_historical == pytest.approx(es_h, rel=1e-12)
            assert s.component_var.sum() == pytest.approx(s.var_parametric, rel=1e-12)
    d = risk.drift()
    assert max(d.values()) < 1e-12
    risk.resync()
    assert max(risk.drift().values()) == 0.0


def test_pnl_and_validation():
    shares, prices, cov, r = fixture()
    risk = IncrementalRisk(shares, prices, cov, r)
    risk.on_tick(1, prices[1] * 1.10)                             # the short loses when its price rises
    assert risk.snapshot().pnl == pytest.approx(-40.0 * prices[1] * 0.10)
    with pytest.raises(ValueError):
        risk.on_tick(0, -1.0)
    with pytest.raises(ValueError):
        IncrementalRisk(shares, prices, cov, r[:10])
    with pytest.raises(ValueError):
        IncrementalRisk(shares[:3], prices, cov, r)


def test_cross_language_fixture():
    shares, prices, cov, r = fixture()
    risk = IncrementalRisk(shares, prices, cov, r, 0.975)
    for i, price in ticks(prices):
        risk.on_tick(i, price)
    s = risk.snapshot()
    print(repr([s.value, s.pnl, s.sigma, s.var_parametric, s.es_parametric, s.var_historical, s.es_historical]),
          repr(s.component_var.tolist()))
    assert [s.value, s.pnl, s.sigma, s.var_parametric, s.es_parametric, s.var_historical, s.es_historical] == pytest.approx(
        [43798.72387759365, 498.7238775936494, 266.01768306576565, 521.3850780596913, 621.8968822460976,
         633.9301400698679, 852.8840476556697], rel=1e-12)
    assert s.component_var == pytest.approx([181.877293533477, 3.462184199621236, 73.20629523668005,
                                             262.8393050899131], rel=1e-12)
