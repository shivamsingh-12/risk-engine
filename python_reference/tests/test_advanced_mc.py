import math

import numpy as np
import pytest
from scipy.stats import qmc as sqmc

from qre.autodiff import aad
from qre.montecarlo import asian, greeks, qmc
from qre.pricing import black_scholes as bs


# ---------------------------------------------------------------- Sobol
@pytest.mark.parametrize("dim", [1, 2, 7, 32, 64])
def test_sobol_matches_scipy(dim):
    ours = qmc.sobol_points(8192, dim)
    theirs = sqmc.Sobol(dim, scramble=False).random(8192)
    assert np.array_equal(ours, theirs)


def test_sobol_is_stratified():
    """Every block of 2^k consecutive points puts exactly one point in each 1/2^k interval of every coordinate."""
    u = qmc.sobol_points(1024, 16)
    for d in range(16):
        assert np.array_equal(np.sort(np.floor(u[:, d] * 1024)), np.arange(1024.0))


def test_digital_shift_keeps_uniformity_and_stratification():
    u = (qmc.sobol_ints(256, 3) ^ qmc.random_shift(3, seed=7)).astype(float) / 2 ** 32
    for d in range(3):
        assert np.array_equal(np.sort(np.floor(u[:, d] * 256)), np.arange(256.0))
    means = [qmc.sobol_normals(64, 2, qmc.random_shift(2, s)).mean() for s in range(400)]
    assert abs(np.mean(means)) < 3 * np.std(means) / math.sqrt(400)        # unbiased over shifts


def test_sobol_validation():
    with pytest.raises(ValueError):
        qmc.direction_numbers(qmc.MAX_DIM + 1)
    with pytest.raises(ValueError):
        qmc.sobol_normals(8, 3, shift=[1, 2])


# ---------------------------------------------------------------- Brownian bridge
@pytest.mark.parametrize("m", [1, 2, 5, 8, 13])
def test_bridge_and_incremental_have_brownian_covariance(m):
    t = 1.5 * np.arange(1, m + 1) / m
    for build in (qmc.brownian_bridge, qmc.incremental_path):
        a = build(np.eye(m), 1.5)                    # row i = response to unit shock in coordinate i
        assert a.T @ a == pytest.approx(np.minimum.outer(t, t), abs=1e-14)


def test_bridge_first_coordinate_sets_the_endpoint():
    w = qmc.brownian_bridge(np.array([[2.0] + [0.0] * 7]), 4.0)
    assert w[0, -1] == pytest.approx(2.0 * 2.0)            # sqrt(T) z_1
    assert w[0] == pytest.approx(np.arange(1, 9) / 8 * 4.0)   # the rest interpolate linearly


# ---------------------------------------------------------------- Asian options
def test_geometric_asian_closed_form():
    assert asian.geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 1) == pytest.approx(bs.price(100, 100, 0.05, 0.2, 1.0))
    exact = asian.geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 12)
    z = np.random.default_rng(1).standard_normal((400_000, 12))
    pay = asian.asian_call_payoffs(100, 100, 0.05, 0.2, 1.0, qmc.incremental_path(z, 1.0), "geometric")
    assert pay.mean() == pytest.approx(exact, abs=3.5 * pay.std() / math.sqrt(pay.size))


def test_qmc_beats_mc_and_bridge_helps():
    exact = asian.geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 32)
    rmse = {}
    for method in ("mc", "sobol", "sobol_bridge"):
        e = [asian.asian_call_estimate(100, 100, 0.05, 0.2, 1.0, 32, 4096, method, "geometric", seed=s) - exact
             for s in range(30)]
        rmse[method] = math.sqrt(np.mean(np.square(e)))
    assert rmse["sobol_bridge"] < rmse["sobol"] < rmse["mc"]
    assert rmse["sobol_bridge"] < rmse["mc"] / 10


def test_asian_validation():
    with pytest.raises(ValueError):
        asian.asian_call_estimate(100, 100, 0.05, 0.2, 1.0, 4, 16, method="halton")
    with pytest.raises(ValueError):
        asian.geometric_asian_call(100, 100, 0.05, 0.2, 1.0, 0)


# ---------------------------------------------------------------- Greeks
def test_pathwise_and_lr_greeks_are_unbiased_but_lr_is_noisier():
    s, k, r, v, t = 100, 105, 0.03, 0.25, 0.8
    z = np.random.default_rng(2).standard_normal(1_000_000)
    pw, lr = greeks.european_call_pathwise(s, k, r, v, t, z), greeks.european_call_lr(s, k, r, v, t, z)
    for name, exact in (("delta", bs.delta(s, k, r, v, t)), ("vega", bs.vega(s, k, r, v, t)),
                        ("rho", bs.rho(s, k, r, v, t))):
        est, se = pw[name]
        assert est == pytest.approx(exact, abs=4 * se)
        if name in lr:
            assert lr[name][0] == pytest.approx(exact, abs=4 * lr[name][1])
            assert lr[name][1] > 2 * se


def test_digital_delta_pathwise_fails_lr_works():
    s, k, r, v, t = 100, 100, 0.02, 0.2, 0.5
    exact = greeks.digital_call_delta(s, k, r, v, t)
    z = np.random.default_rng(3).standard_normal(1_000_000)
    assert greeks.digital_call_delta_pathwise(s, k, r, v, t, z)[0] == 0.0
    est, se = greeks.digital_call_delta_lr(s, k, r, v, t, z)
    assert est == pytest.approx(exact, abs=4 * se)
    small, large = (greeks.digital_call_delta_bump(s, k, r, v, t, z, h) for h in (0.01, 1.0))
    assert small[1] > 5 * large[1]                      # a small bump on a jump is very noisy


def test_basket_pathwise_reduces_to_black_scholes():
    z = np.random.default_rng(4).standard_normal((1_000_000, 1))
    price, g = greeks.basket_call_pathwise([100.0], [0.2], [[1.0]], [1.0], 100.0, 0.05, 1.0, z)
    assert price == pytest.approx(bs.price(100, 100, 0.05, 0.2, 1.0), rel=0.01)
    assert g["spots"][0] == pytest.approx(bs.delta(100, 100, 0.05, 0.2, 1.0), rel=0.01)
    assert g["vols"][0] == pytest.approx(bs.vega(100, 100, 0.05, 0.2, 1.0), rel=0.01)
    assert g["rate"] == pytest.approx(bs.rho(100, 100, 0.05, 0.2, 1.0), rel=0.01)


# ---------------------------------------------------------------- AAD
def test_aad_black_scholes_greeks_are_exact():
    tape = aad.Tape()
    s, k, r, v, t = (tape.variable(a) for a in (100.0, 95.0, 0.03, 0.25, 0.7))
    sq = aad.sqrt(t)
    d1 = (aad.log(s / k) + (r + 0.5 * v * v) * t) / (v * sq)
    d2 = d1 - v * sq
    price = s * aad.ncdf(d1) - k * aad.exp(-r * t) * aad.ncdf(d2)
    ds, dv, dr, dt = tape.gradient(price, [s, v, r, t])
    args = (100.0, 95.0, 0.03, 0.25, 0.7)
    assert price.value == pytest.approx(bs.price(*args), rel=1e-14)
    assert ds == pytest.approx(bs.delta(*args), rel=1e-13)
    assert dv == pytest.approx(bs.vega(*args), rel=1e-13)
    assert dr == pytest.approx(bs.rho(*args), rel=1e-13)
    assert dt == pytest.approx(-bs.theta(*args), rel=1e-13)       # dV/dT = -dV/dt


def test_aad_matches_finite_differences_on_a_messy_function():
    f = lambda x, y, z: (x * y + 3.0) / (1.0 + z * z) - aad.exp(x / y) + 2.0 / x if isinstance(x, aad.Var) else \
        (x * y + 3.0) / (1.0 + z * z) - math.exp(x / y) + 2.0 / x
    tape = aad.Tape()
    xs = [tape.variable(a) for a in (0.7, 1.9, -0.4)]
    grad = tape.gradient(f(*xs), xs)
    for i in range(3):
        p = [0.7, 1.9, -0.4]
        h = 1e-6
        up, dn = list(p), list(p)
        up[i] += h
        dn[i] -= h
        assert grad[i] == pytest.approx((f(*up) - f(*dn)) / (2 * h), rel=1e-8)


def test_aad_basket_equals_pathwise_formula():
    """Recording the basket pricer on the tape reproduces the hand-derived pathwise gradient exactly."""
    spots, vols, w = [100.0, 90.0, 110.0], [0.2, 0.3, 0.25], [0.5, 0.3, 0.2]
    corr = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    k, r, t = 100.0, 0.04, 0.9
    z = qmc.sobol_normals(256, 3, qmc.random_shift(3, 1))
    price, g = greeks.basket_call_pathwise(spots, vols, corr, w, k, r, t, z)
    x = z @ np.linalg.cholesky(corr).T
    tape = aad.Tape()
    s_v, v_v = [tape.variable(a) for a in spots], [tape.variable(a) for a in vols]
    r_v, k_v = tape.variable(r), tape.variable(k)
    total = 0.0
    for row in x:
        b = sum(wi * si * aad.exp((r_v - 0.5 * vi * vi) * t + vi * math.sqrt(t) * xi)
                for wi, si, vi, xi in zip(w, s_v, v_v, row))
        total = total + aad.max0(b - k_v)
    v_out = aad.exp(-r_v * t) * total / len(x)
    grad = tape.gradient(v_out, s_v + v_v + [r_v, k_v])
    assert v_out.value == pytest.approx(price, rel=1e-13)
    assert grad[:3] == pytest.approx(g["spots"], rel=1e-12)
    assert grad[3:6] == pytest.approx(g["vols"], rel=1e-12)
    assert grad[6] == pytest.approx(g["rate"], rel=1e-12)
    assert grad[7] == pytest.approx(g["strike"], rel=1e-12)


def test_aad_rejects_foreign_output():
    a, b = aad.Tape(), aad.Tape()
    x = a.variable(1.0)
    with pytest.raises(ValueError):
        b.gradient(x * 2.0, [x])
