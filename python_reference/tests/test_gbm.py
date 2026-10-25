import math

import numpy as np
import pytest

from qre.montecarlo import gbm
from qre.pricing import black_scholes as bs

S, K, R, SIG, T = 100.0, 100.0, 0.05, 0.2, 1.0
MODES = [(False, False), (True, False), (False, True), (True, True)]


def test_paths_shape_start_and_positive():
    p = gbm.simulate_gbm_paths(S, R, SIG, T, 50, 1000, seed=1)
    assert p.shape == (1000, 51)
    assert np.all(p[:, 0] == S)
    assert np.all(p > 0.0)


def test_terminal_log_return_moments():
    p = gbm.simulate_gbm_paths(S, R, SIG, T, 10, 200_000, seed=2)
    lr = np.log(p[:, -1] / S)
    se = SIG * math.sqrt(T) / math.sqrt(len(lr))
    assert abs(lr.mean() - (R - 0.5 * SIG**2) * T) < 4 * se
    assert lr.std(ddof=1) == pytest.approx(SIG * math.sqrt(T), rel=0.01)


def test_terminal_mean_is_forward():
    p = gbm.simulate_gbm_paths(S, R, SIG, T, 5, 200_000, seed=3)
    assert p[:, -1].mean() == pytest.approx(S * math.exp(R * T), rel=0.01)


def test_step_count_does_not_change_terminal_distribution():
    a = np.log(gbm.simulate_gbm_paths(S, R, SIG, T, 1, 200_000, seed=4)[:, -1] / S)
    b = np.log(gbm.simulate_gbm_paths(S, R, SIG, T, 100, 200_000, seed=5)[:, -1] / S)
    assert a.mean() == pytest.approx(b.mean(), abs=0.01)
    assert a.std() == pytest.approx(b.std(), rel=0.01)


def test_same_seed_reproducible_different_seed_differs():
    a = gbm.mc_european_price(S, K, R, SIG, T, 10_000, seed=7)
    b = gbm.mc_european_price(S, K, R, SIG, T, 10_000, seed=7)
    c = gbm.mc_european_price(S, K, R, SIG, T, 10_000, seed=8)
    assert a == b
    assert a != c


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("anti,cv", MODES)
def test_mc_price_within_four_standard_errors_of_black_scholes(kind, anti, cv):
    price, se = gbm.mc_european_price(S, K, R, SIG, T, 200_000, kind=kind, seed=11, antithetic=anti, control_variate=cv)
    exact = bs.price(S, K, R, SIG, T, kind)
    assert abs(price - exact) < 4 * se


def test_variance_reduction_lowers_standard_error():
    base = gbm.mc_european_price(S, K, R, SIG, T, 100_000, seed=21)[1]
    anti = gbm.mc_european_price(S, K, R, SIG, T, 100_000, seed=21, antithetic=True)[1]
    cv = gbm.mc_european_price(S, K, R, SIG, T, 100_000, seed=21, control_variate=True)[1]
    both = gbm.mc_european_price(S, K, R, SIG, T, 100_000, seed=21, antithetic=True, control_variate=True)[1]
    assert anti < base
    assert cv < base
    assert both < base


def test_error_scales_as_inverse_sqrt_n():
    exact = bs.price(S, K, R, SIG, T, "call")

    def rmse(n):
        errs = [gbm.mc_european_price(S, K, R, SIG, T, n, seed=1000 + i)[0] - exact for i in range(200)]
        return math.sqrt(np.mean(np.square(errs)))

    ratio = rmse(1_000) / rmse(10_000)          # expect sqrt(10) ~ 3.16
    assert ratio == pytest.approx(math.sqrt(10), rel=0.25)


def test_reported_standard_error_matches_empirical_spread():
    prices = [gbm.mc_european_price(S, K, R, SIG, T, 5_000, seed=2000 + i, antithetic=True)[0] for i in range(300)]
    se = gbm.mc_european_price(S, K, R, SIG, T, 5_000, seed=1, antithetic=True)[1]
    assert np.std(prices, ddof=1) == pytest.approx(se, rel=0.2)


def test_correlated_paths_reproduce_target_correlation():
    corr = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.1], [-0.3, 0.1, 1.0]])
    paths = gbm.simulate_correlated_gbm([100, 50, 80], R, [0.2, 0.3, 0.25], corr, T, 4, 200_000, seed=31)
    assert paths.shape == (3, 200_000, 5)
    lr = np.log(paths[:, :, -1] / paths[:, :, 0])
    assert np.corrcoef(lr) == pytest.approx(corr, abs=0.01)


def test_correlated_marginals_match_single_asset_moments():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    paths = gbm.simulate_correlated_gbm([100, 100], R, [0.2, 0.4], corr, T, 3, 200_000, seed=32)
    for i, vol in enumerate([0.2, 0.4]):
        lr = np.log(paths[i, :, -1] / 100)
        assert lr.std(ddof=1) == pytest.approx(vol * math.sqrt(T), rel=0.01)


def test_antithetic_halves_are_mirrored():
    p = gbm.simulate_gbm_paths(S, R, SIG, T, 3, 10, seed=5, antithetic=True)
    log_inc = np.diff(np.log(p), axis=1) - (R - 0.5 * SIG**2) * T / 3
    assert log_inc[:5] == pytest.approx(-log_inc[5:])


@pytest.mark.parametrize("bad", [
    dict(spot=-1, strike=K, rate=R, vol=SIG, time=T, n_paths=100),
    dict(spot=S, strike=0, rate=R, vol=SIG, time=T, n_paths=100),
    dict(spot=S, strike=K, rate=R, vol=0, time=T, n_paths=100),
    dict(spot=S, strike=K, rate=R, vol=SIG, time=0, n_paths=100),
    dict(spot=S, strike=K, rate=R, vol=SIG, time=T, n_paths=1),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        gbm.mc_european_price(**bad)


def test_odd_path_count_with_antithetic_raises():
    with pytest.raises(ValueError):
        gbm.mc_european_price(S, K, R, SIG, T, 101, antithetic=True)


def test_non_positive_definite_correlation_raises():
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with pytest.raises(ValueError):
        gbm.simulate_correlated_gbm([100, 100, 100], R, [0.2] * 3, bad, T, 2, 100)
