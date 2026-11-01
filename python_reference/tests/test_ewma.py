import math

import numpy as np
import pytest
from scipy.stats import norm

from qre.risk import backtest as bt
from qre.risk import ewma
from qre.risk import var_es as ve

W = np.array([0.5, 0.3, 0.2])
COV = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, -0.01], [0.0, -0.01, 0.0225]]) / 252.0


def _prices(n=400, seed=0, vol_scale=None):
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal((n, 3)) @ np.linalg.cholesky(COV).T
    if vol_scale is not None:
        rets *= vol_scale[:, None]
    return 100.0 * np.vstack([np.ones(3), np.cumprod(1.0 + rets, axis=0)])


# ---------------------------------------------------------------- weights and covariance
def test_weights_sum_to_one_and_favor_recent():
    w = ewma.ewma_weights(250, 0.94)
    assert w.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.diff(w) > 0)                                  # newest (last) is largest
    assert w[-1] / w[-2] == pytest.approx(1 / 0.94)


def test_weights_known_values():
    w = ewma.ewma_weights(3, 0.5)                                  # (1-l) l^k / (1-l^3), k = 2,1,0
    assert w == pytest.approx([0.125 / 0.875, 0.25 / 0.875, 0.5 / 0.875])


def test_covariance_matches_manual_weighted_sum():
    rng = np.random.default_rng(1)
    r = rng.standard_normal((40, 3)) * 0.01
    w = ewma.ewma_weights(40, 0.9)
    manual = sum(w[i] * np.outer(r[i], r[i]) for i in range(40))
    assert ewma.ewma_covariance(r, 0.9) == pytest.approx(manual, rel=1e-12)


def test_covariance_approaches_zero_mean_sample_moment_as_lambda_to_one():
    rng = np.random.default_rng(2)
    r = rng.standard_normal((300, 2)) * 0.01
    assert ewma.ewma_covariance(r, 0.999999) == pytest.approx(r.T @ r / 300, abs=1e-3 * np.abs(r.T @ r / 300).max())


def test_covariance_reacts_to_recent_shock():
    r = np.full((100, 1), 0.01)
    r[-1, 0] = 0.05
    assert ewma.ewma_covariance(r, 0.94)[0, 0] > ewma.ewma_covariance(r, 0.99)[0, 0]


# ---------------------------------------------------------------- vol path and FHS
def test_vol_path_recursion():
    r = np.array([0.01, -0.02, 0.015, 0.0, -0.005])
    s = ewma.ewma_vol_path(r, 0.9)
    s2 = [np.mean(r * r)]
    for x in r:
        s2.append(0.9 * s2[-1] + 0.1 * x * x)
    assert s == pytest.approx(np.sqrt(s2), rel=1e-14)
    assert s.size == r.size + 1


def test_fhs_rescales_history_to_current_volatility():
    rng = np.random.default_rng(3)
    calm = rng.standard_normal(200) * 0.01
    stressed = np.concatenate([calm, rng.standard_normal(50) * 0.03])   # vol triples at the end
    v_calm, _ = ewma.fhs_var_es(calm, 0.99)
    v_stress, _ = ewma.fhs_var_es(stressed, 0.99)
    v_hist, _ = ve.historical_var_es(stressed, 0.99)
    assert v_stress > 2.0 * v_calm
    assert v_stress > v_hist                         # plain historical is still anchored to the calm days


def test_fhs_is_close_to_historical_on_constant_vol_data():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(5000) * 0.01
    v_fhs, e_fhs = ewma.fhs_var_es(r, 0.99, lam=0.99)
    v_hist, e_hist = ve.historical_var_es(r, 0.99)
    assert v_fhs == pytest.approx(v_hist, rel=0.15)
    assert e_fhs >= v_fhs


def test_ewma_parametric_uses_ewma_vol():
    rng = np.random.default_rng(5)
    r = rng.standard_normal((250, 3)) @ np.linalg.cholesky(COV).T
    v, e = ewma.ewma_parametric_var_es(r, W, 0.99, 0.94)
    sigma = math.sqrt(W @ ewma.ewma_covariance(r, 0.94) @ W)
    assert v == pytest.approx(sigma * norm.ppf(0.99), rel=1e-12)
    assert e == pytest.approx(sigma * norm.pdf(norm.ppf(0.99)) / 0.01, rel=1e-12)


@pytest.mark.parametrize("call", [
    lambda: ewma.ewma_weights(10, 1.0),
    lambda: ewma.ewma_weights(10, 0.0),
    lambda: ewma.ewma_weights(0, 0.9),
    lambda: ewma.ewma_vol_path([0.01], 0.9),
    lambda: ewma.ewma_vol_path([0.0, 0.0, 0.0], 0.9),
    lambda: ewma.ewma_covariance(np.zeros((0, 2)), 0.9),
])
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()


# ---------------------------------------------------------------- backtest integration
@pytest.mark.parametrize("method", ["ewma", "fhs"])
def test_adaptive_backtest_has_no_look_ahead(method):
    p = _prices()
    s = bt.rolling_backtest(p, W, window=100, method=method)
    future = p.copy()
    future[201:] *= np.linspace(0.5, 3.0, future.shape[0] - 201)[:, None]
    s2 = bt.rolling_backtest(future, W, window=100, method=method)
    i = int(np.flatnonzero(s.rows == 201)[0])
    assert np.array_equal(s.var[: i + 1], s2.var[: i + 1])
    assert np.array_equal(s.es[: i + 1], s2.es[: i + 1])


def test_adaptive_methods_cut_breaches_after_a_volatility_jump():
    """Vol triples halfway through: the equal-weight window lags, EWMA and FHS catch up quickly."""
    n = 900
    scale = np.where(np.arange(n) < 600, 1.0, 3.0)
    p = _prices(n, seed=6, vol_scale=scale)
    counts = {}
    for m in ("parametric", "historical", "ewma", "fhs"):
        s = bt.rolling_backtest(p, W, window=250, confidence=0.99, method=m)
        after = s.rows >= 601
        counts[m] = int(s.breaches[after].sum())
    assert counts["ewma"] < counts["parametric"]
    assert counts["fhs"] < counts["historical"]
