import math

import numpy as np
import pytest
from scipy.stats import norm

from qre.risk import pipeline as pl
from qre.risk import var_es as ve

COV = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, -0.01], [0.0, -0.01, 0.0225]]) / 252.0
W = np.array([0.5, 0.3, 0.2])


def test_historical_known_values():
    returns = -np.arange(1, 101) / 100.0                  # losses 0.01 .. 1.00
    var, es = ve.historical_var_es(returns, 0.95)
    assert var == pytest.approx(0.9505, abs=1e-12)        # linear quantile: 94.05 -> between 95 and 96
    assert es == pytest.approx(0.98, abs=1e-12)           # mean of losses 0.96 .. 1.00


def test_historical_es_at_least_var():
    rng = np.random.default_rng(1)
    var, es = ve.historical_var_es(rng.standard_normal(1000) * 0.01, 0.975)
    assert es >= var


def test_parametric_known_values():
    var, es = ve.parametric_var_es(0.0, 0.01, 0.99)
    assert var == pytest.approx(0.01 * 2.3263478740408408, rel=1e-12)
    assert es == pytest.approx(0.01 * 2.665214220345808, rel=1e-9)


def test_parametric_mean_and_horizon_scaling():
    v1, e1 = ve.parametric_var_es(0.0, 0.01, 0.99, horizon_days=1)
    v10, e10 = ve.parametric_var_es(0.0, 0.01, 0.99, horizon_days=10)
    assert v10 == pytest.approx(v1 * math.sqrt(10))
    assert e10 == pytest.approx(e1 * math.sqrt(10))
    vm, _ = ve.parametric_var_es(0.001, 0.01, 0.99)
    assert vm == pytest.approx(v1 - 0.001)


def test_risk_contributions_sum_to_portfolio_vol():
    rc = ve.risk_contributions(W, COV)
    assert rc.sum() == pytest.approx(ve.portfolio_vol(W, COV), rel=1e-12)


def test_risk_contributions_independent_equal_weights():
    cov = np.diag([0.01, 0.04])
    rc = ve.risk_contributions(np.array([0.5, 0.5]), cov)
    assert rc[1] / rc[0] == pytest.approx(4.0)             # contribution proportional to variance


def test_single_asset_contribution_is_its_vol():
    assert ve.risk_contributions(np.array([1.0]), np.array([[0.0004]]))[0] == pytest.approx(0.02)


def test_monte_carlo_matches_parametric_one_day():
    sigma = ve.portfolio_vol(W, COV)
    pv, pe = ve.parametric_var_es(0.0, sigma, 0.99)
    mv, me = ve.monte_carlo_var_es(W, COV, 0.99, n_paths=400_000, seed=3)
    assert mv == pytest.approx(pv, rel=0.03)
    assert me == pytest.approx(pe, rel=0.03)


def test_monte_carlo_horizon_scales_like_sqrt_time():
    v1, _ = ve.monte_carlo_var_es(W, COV, 0.99, horizon_days=1, n_paths=400_000, seed=4)
    v10, _ = ve.monte_carlo_var_es(W, COV, 0.99, horizon_days=10, n_paths=400_000, seed=4)
    assert v10 / v1 == pytest.approx(math.sqrt(10), rel=0.05)


def test_monte_carlo_reproducible():
    a = ve.monte_carlo_var_es(W, COV, 0.99, n_paths=10_000, seed=9)
    assert a == ve.monte_carlo_var_es(W, COV, 0.99, n_paths=10_000, seed=9)


def test_higher_confidence_means_higher_var_and_es():
    low = ve.parametric_var_es(0.0, 0.01, 0.95)
    high = ve.parametric_var_es(0.0, 0.01, 0.99)
    assert high[0] > low[0] and high[1] > low[1]


def test_portfolio_returns_matches_manual():
    r = np.array([[0.01, -0.02], [0.03, 0.01]])
    assert ve.portfolio_returns(r, np.array([0.5, 0.5])) == pytest.approx([-0.005, 0.02])


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_invalid_confidence_raises(bad):
    with pytest.raises(ValueError):
        ve.parametric_var_es(0.0, 0.01, bad)
    with pytest.raises(ValueError):
        ve.historical_var_es(np.arange(10) / 100.0, bad)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        ve.parametric_var_es(0.0, 0.0, 0.99)
    with pytest.raises(ValueError):
        ve.historical_var_es([0.01], 0.99)
    with pytest.raises(ValueError):
        ve.portfolio_returns(np.zeros((5, 3)), np.ones(2))
    with pytest.raises(ValueError):
        ve.parametric_var_es(0.0, 0.01, 0.99, horizon_days=0)


def _prices(n=750, seed=0):
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(COV)
    rets = rng.standard_normal((n, 3)) @ chol.T
    return 100.0 * np.cumprod(1.0 + rets, axis=0)


def test_pipeline_report_is_consistent():
    prices = _prices()
    rep = pl.run_pipeline(prices, W, confidence=0.99, n_paths=200_000, seed=5)
    returns = prices[1:] / prices[:-1] - 1.0
    cov = np.cov(returns, rowvar=False, ddof=1)
    assert rep.n_returns == 749
    assert rep.portfolio_vol == pytest.approx(math.sqrt(W @ cov @ W))
    assert rep.risk_contributions.sum() == pytest.approx(rep.portfolio_vol)
    assert rep.parametric[0] == pytest.approx(rep.portfolio_vol * norm.ppf(0.99))
    # On Gaussian data all three methods should be in the same ballpark
    assert rep.historical[0] == pytest.approx(rep.parametric[0], rel=0.2)
    assert rep.monte_carlo[0] == pytest.approx(rep.parametric[0], rel=0.03)
    for var, es in (rep.historical, rep.parametric, rep.monte_carlo):
        assert es >= var > 0.0


def test_pipeline_forward_fills_short_gap_but_rejects_long_gap():
    prices = _prices()
    short = prices.copy()
    short[100:103, 1] = np.nan
    assert pl.run_pipeline(short, W, n_paths=20_000).n_returns == 749
    long_gap = prices.copy()
    long_gap[100:110, 1] = np.nan
    with pytest.raises(ValueError):
        pl.run_pipeline(long_gap, W, n_paths=20_000)


def test_pipeline_input_validation():
    prices = _prices(50)
    with pytest.raises(ValueError):
        pl.run_pipeline(prices, np.array([0.5, 0.5]))
    bad = prices.copy()
    bad[5, 0] = -1.0
    with pytest.raises(ValueError):
        pl.run_pipeline(bad, W)
    with pytest.raises(ValueError):
        pl.run_pipeline(prices[:2], W)
