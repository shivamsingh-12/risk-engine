import math

import numpy as np
import pytest
from scipy.stats import binom, chi2, norm

from qre.risk import backtest as bt

COV = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, -0.01], [0.0, -0.01, 0.0225]]) / 252.0
W = np.array([0.5, 0.3, 0.2])


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n, 3)) @ np.linalg.cholesky(COV).T
    return 100.0 * np.vstack([np.ones(3), np.cumprod(1.0 + rets, axis=0)])


# ---------------------------------------------------------------- chi-square helpers
@pytest.mark.parametrize("x", [0.0, 0.3, 1.0, 3.841458820694124, 10.0])
def test_closed_form_chi2_matches_scipy(x):
    assert bt.chi2_sf_df1(x) == pytest.approx(chi2.sf(x, 1), rel=1e-12, abs=1e-15)
    assert bt.chi2_sf_df2(x) == pytest.approx(chi2.sf(x, 2), rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------- Kupiec
@pytest.mark.parametrize("n,x", [(250, 0), (250, 2), (250, 7), (500, 5), (500, 13), (10, 10)])
def test_kupiec_matches_binomial_likelihood_ratio(n, x):
    alpha = 0.01
    lr, p = bt.kupiec_pof(n, x, alpha)
    expected = 2.0 * (binom.logpmf(x, n, x / n) - binom.logpmf(x, n, alpha))
    assert lr == pytest.approx(expected, rel=1e-10, abs=1e-12)
    assert p == pytest.approx(chi2.sf(expected, 1), rel=1e-9, abs=1e-15)


def test_kupiec_exact_rate_is_not_rejected():
    lr, p = bt.kupiec_pof(500, 5, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_kupiec_known_value():
    lr, p = bt.kupiec_pof(250, 10, 0.01)
    assert lr == pytest.approx(12.955491062356, abs=1e-9)       # many breaches -> reject
    assert p < 0.001


@pytest.mark.parametrize("args", [(0, 0, 0.01), (10, 11, 0.01), (10, -1, 0.01), (10, 1, 0.0), (10, 1, 1.0)])
def test_kupiec_invalid_inputs(args):
    with pytest.raises(ValueError):
        bt.kupiec_pof(*args)


# ---------------------------------------------------------------- Christoffersen
def _manual_ind(seq):
    seq = list(seq)
    n = {(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for a, b in zip(seq[:-1], seq[1:]):
        n[(a, b)] += 1
    pi01 = n[(0, 1)] / (n[(0, 0)] + n[(0, 1)])
    pi11 = n[(1, 1)] / (n[(1, 0)] + n[(1, 1)])
    pi = (n[(0, 1)] + n[(1, 1)]) / sum(n.values())
    l0 = (1 - pi) ** (n[(0, 0)] + n[(1, 0)]) * pi ** (n[(0, 1)] + n[(1, 1)])
    l1 = (1 - pi01) ** n[(0, 0)] * pi01 ** n[(0, 1)] * (1 - pi11) ** n[(1, 0)] * pi11 ** n[(1, 1)]
    return -2.0 * math.log(l0 / l1)


def test_christoffersen_matches_manual_likelihoods():
    seq = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    lr, p = bt.christoffersen_independence(seq)
    assert lr == pytest.approx(_manual_ind(seq), rel=1e-12)
    assert p == pytest.approx(chi2.sf(lr, 1))


def test_christoffersen_clustered_breaches_are_rejected():
    seq = np.zeros(500, dtype=bool)
    seq[100:105] = True                                 # five breaches in a row
    _, p = bt.christoffersen_independence(seq)
    assert p < 0.001


def test_christoffersen_no_breaches_and_isolated_breaches():
    lr0, p0 = bt.christoffersen_independence(np.zeros(100, dtype=bool))
    assert lr0 == 0.0 and p0 == 1.0
    seq = np.zeros(500, dtype=bool)
    seq[[50, 150, 250, 350, 450]] = True               # evenly spread: no clustering evidence
    _, p = bt.christoffersen_independence(seq)
    assert p > 0.5


def test_conditional_coverage_is_sum_of_parts():
    seq = np.zeros(300, dtype=bool)
    seq[[10, 11, 80, 200, 201, 202]] = True
    lr_pof, _ = bt.kupiec_pof(300, 6, 0.01)
    lr_ind, _ = bt.christoffersen_independence(seq)
    lr, p = bt.conditional_coverage(seq, 0.01)
    assert lr == pytest.approx(lr_pof + lr_ind)
    assert p == pytest.approx(chi2.sf(lr, 2))


# ---------------------------------------------------------------- traffic light
def test_traffic_light_reproduces_basel_table():
    zones = [bt.traffic_light(x, 250, 0.01) for x in range(0, 13)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 3


# ---------------------------------------------------------------- ES Z2
def test_z2_is_zero_on_average_for_a_correct_model():
    rng = np.random.default_rng(1)
    alpha = 0.025
    losses = rng.standard_normal(400_000)
    z = norm.ppf(1 - alpha)
    var = np.full(losses.size, z)
    es = np.full(losses.size, norm.pdf(z) / alpha)
    assert abs(bt.es_z2(losses, var, es, alpha)) < 0.03


def test_z2_positive_when_es_too_small():
    rng = np.random.default_rng(2)
    alpha = 0.025
    losses = rng.standard_normal(400_000)
    z = norm.ppf(1 - alpha)
    es_true = norm.pdf(z) / alpha
    z2 = bt.es_z2(losses, np.full(losses.size, z), np.full(losses.size, 0.8 * es_true), alpha)
    assert z2 == pytest.approx(0.25, abs=0.04)          # 1/0.8 - 1


def test_z2_invalid_inputs():
    with pytest.raises(ValueError):
        bt.es_z2([1.0, 2.0], [1.0], [1.0, 1.0], 0.01)
    with pytest.raises(ValueError):
        bt.es_z2([1.0], [0.5], [0.0], 0.01)


# ---------------------------------------------------------------- size and power of the test itself
def test_kupiec_size_and_power():
    rng = np.random.default_rng(3)
    n, alpha, reps = 500, 0.01, 3000

    def reject_rate(p_true):
        counts = rng.binomial(n, p_true, size=reps)
        return np.mean([bt.kupiec_pof(n, int(x), alpha)[1] < 0.05 for x in counts])

    size, power = reject_rate(alpha), reject_rate(2 * alpha)
    assert 0.02 < size < 0.09          # discrete test: close to, not exactly, 5%
    assert power > 3 * size


# ---------------------------------------------------------------- rolling engine
@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_rolling_backtest_shapes_and_realized_losses(method):
    p = _prices()
    s = bt.rolling_backtest(p, W, window=100, confidence=0.99, method=method, n_paths=4000)
    assert s.rows[0] == 101 and s.rows[-1] == p.shape[0] - 1
    assert s.var.size == p.shape[0] - 101
    t = s.rows[10]
    assert s.loss[10] == pytest.approx(-W @ (p[t] / p[t - 1] - 1.0))
    assert np.all(s.es >= s.var)


@pytest.mark.parametrize("method", ["historical", "parametric", "monte_carlo"])
def test_rolling_backtest_has_no_look_ahead(method):
    p = _prices()
    s = bt.rolling_backtest(p, W, window=100, method=method, n_paths=4000)
    future = p.copy()
    future[201:] *= np.linspace(0.5, 3.0, future.shape[0] - 201)[:, None]   # rewrite rows >= 201
    s2 = bt.rolling_backtest(future, W, window=100, method=method, n_paths=4000)
    i = int(np.flatnonzero(s.rows == 201)[0])
    assert np.array_equal(s.var[: i + 1], s2.var[: i + 1])   # forecasts up to day 201 unchanged
    assert np.array_equal(s.es[: i + 1], s2.es[: i + 1])
    assert s.loss[i] != s2.loss[i]                           # but day 201's realized loss did change


def test_parametric_backtest_on_gaussian_data_passes():
    p = _prices(1500, seed=4)
    s = bt.rolling_backtest(p, W, window=250, confidence=0.99, method="parametric")
    summ = bt.summarize(s, 0.99)
    assert summ.n_obs == 1250
    assert summ.kupiec_p > 0.01
    assert summ.traffic_light in ("green", "yellow")


def test_rolling_backtest_rejects_bad_input():
    p = _prices(50)
    with pytest.raises(ValueError):
        bt.rolling_backtest(p, W, window=60)
    with pytest.raises(ValueError):
        bt.rolling_backtest(p, np.ones(2) / 2, window=10)
    bad = p.copy()
    bad[20, 1] = np.nan
    with pytest.raises(ValueError):
        bt.rolling_backtest(bad, W, window=10)
    with pytest.raises(ValueError):
        bt.rolling_backtest(p, W, window=10, method="not_a_method")
