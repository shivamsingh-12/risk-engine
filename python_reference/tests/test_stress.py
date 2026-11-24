import math
import os

import numpy as np
import pytest

from qre.risk import stress as st

LONG = os.path.join(os.path.dirname(__file__), "..", "..", "data", "market_data", "long_history_close.csv")


def test_window_indices_use_last_trading_day_on_or_before():
    dates = np.array(["2020-03-05", "2020-03-06", "2020-03-09", "2020-03-10", "2020-03-11"])
    assert st.window_indices(dates, "2020-03-07", "2020-03-10") == (1, 3)      # Saturday -> Friday's close
    assert st.window_indices(dates, "2020-03-05", "2020-12-31") == (0, 4)
    with pytest.raises(ValueError):
        st.window_indices(dates, "2020-03-01", "2020-03-10")                    # before the data
    with pytest.raises(ValueError):
        st.window_indices(dates, "2020-03-09", "2020-03-09")
    with pytest.raises(ValueError):
        st.window_indices(dates, "2020-03-07", "2020-03-08")                    # no trading day in between


def test_scenarios_are_ordered_and_unique():
    keys = [s.key for s in st.SCENARIOS]
    assert len(set(keys)) == len(keys)
    assert all(s.start < s.end for s in st.SCENARIOS)
    assert st.scenario("covid").start == "2020-02-19"
    with pytest.raises(ValueError):
        st.scenario("nope")


def test_beta_recovers_slope_and_skips_missing_days():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, 2000)
    a = 1.3 * m + rng.normal(0, 0.002, 2000)
    a[:500] = np.nan                                                            # listed later
    assert st.beta(a, m) == pytest.approx(1.3, abs=0.01)
    with pytest.raises(ValueError):
        st.beta(a[:510], m[:510])                                               # only 10 shared days


def test_scenario_returns_fill_from_proxy():
    proxy = np.array([100.0, 98.0, 99.0, 90.0])
    prices = np.array([[50.0, np.nan], [51.0, np.nan], [49.0, 20.0], [48.0, 19.0]])
    r, proxied, days = st.scenario_returns(prices, proxy, np.array([np.nan, 1.5]))
    assert r[:, 0] == pytest.approx([0.02, 49 / 51 - 1, 48 / 49 - 1])
    assert r[0, 1] == pytest.approx(1.5 * (98 / 100 - 1))
    assert r[1, 1] == pytest.approx(1.5 * (99 / 98 - 1))
    assert r[2, 1] == pytest.approx(19 / 20 - 1)                                # own return once both closes exist
    assert proxied.tolist() == [False, True] and days.tolist() == [0, 2]
    with pytest.raises(ValueError):
        st.scenario_returns(prices, proxy, np.array([np.nan, np.nan]))          # missing data needs a beta
    with pytest.raises(ValueError):
        st.scenario_returns(prices, np.array([100.0, np.nan, 99.0, 90.0]), np.array([1.0, 1.0]))


def test_replay_buy_and_hold_by_hand():
    r = np.array([[0.10, -0.20], [-0.50, 0.10], [0.20, 0.0]])
    w = np.array([0.6, 0.4])
    rp = st.replay(r, w)
    g = np.array([1.1 * 0.5 * 1.2, 0.8 * 1.1])
    v = [1.0, 0.6 * 1.1 + 0.4 * 0.8, 0.6 * 0.55 + 0.4 * 0.88, 0.6 * 0.66 + 0.4 * 0.88]
    assert rp.value == pytest.approx(v)
    assert rp.total_return == pytest.approx(v[-1] - 1)
    assert rp.contributions.sum() == pytest.approx(rp.total_return)
    assert rp.asset_returns == pytest.approx(g - 1)
    assert rp.worst_day == pytest.approx(v[2] / v[1] - 1) and rp.worst_day_index == 2
    assert rp.worst_10d == pytest.approx(v[3] / v[0] - 1)                       # window shorter than 10 days
    assert rp.max_drawdown == pytest.approx(1 - v[2] / v[0])                    # the first point is the peak


def test_replay_cash_and_shorts():
    r = np.array([[-0.10], [-0.10]])
    half = st.replay(r, np.array([0.5]))                                        # half in cash
    assert half.total_return == pytest.approx(0.5 * (0.81 - 1))
    short = st.replay(r, np.array([-1.0]))                                      # 200% cash, short 100%
    assert short.total_return == pytest.approx(0.19)
    assert short.max_drawdown == 0.0
    with pytest.raises(ValueError):
        st.replay(np.array([[-1.0]]), np.array([1.0]))


def bivariate(s1, s2, rho):
    return np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])


def test_conditional_shock_matches_bivariate_formula():
    cov = bivariate(0.01, 0.02, 0.6)
    sh = st.conditional_shock(cov, [0], [-0.2], horizon_days=10)
    assert sh.mean[0] == -0.2
    assert sh.mean[1] == pytest.approx(0.6 * 0.02 / 0.01 * -0.2)
    assert sh.cond_cov[1, 1] == pytest.approx(0.02 ** 2 * (1 - 0.36) * 10)
    assert sh.cond_cov[0, 0] == 0.0 and sh.cond_cov[0, 1] == 0.0
    # the expected move does not depend on the scale (horizon) of the covariance
    assert st.conditional_shock(cov * 25, [0], [-0.2]).mean == pytest.approx(sh.mean)


def test_conditional_shock_caps_at_minus_100_and_validates():
    cov = bivariate(0.01, 0.05, 0.9)
    sh = st.conditional_shock(cov, [0], [-0.5])                                 # implies -225% for asset 1
    assert sh.mean[1] == -1.0
    with pytest.raises(ValueError):
        st.conditional_shock(np.ones((2, 2)) * 1e-4, [0, 1], [-0.1, -0.1])      # singular
    with pytest.raises(ValueError):
        st.conditional_shock(cov, [0, 0], [-0.1, -0.1])
    with pytest.raises(ValueError):
        st.conditional_shock(cov, [2], [-0.1])
    with pytest.raises(ValueError):
        st.conditional_shock(cov, [0], [-1.0])
    with pytest.raises(ValueError):
        st.conditional_shock(np.array([[1.0, 0.5], [0.4, 1.0]]), [0], [-0.1])


def three_asset_cov():
    vol = np.array([0.012, 0.018, 0.006])
    corr = np.array([[1.0, 0.7, -0.3], [0.7, 1.0, -0.2], [-0.3, -0.2, 1.0]])
    return corr * np.outer(vol, vol)


def test_shock_pnl_and_reverse_stress_agree():
    cov = three_asset_cov()
    w = np.array([0.5, 0.3, 0.2])
    s = st.reverse_stress(w, cov, 0, 0.15)
    pnl = st.shock_pnl(w, st.conditional_shock(cov, [0], [s]), 0.99)
    assert pnl.expected_return == pytest.approx(-0.15)
    assert pnl.contributions.sum() == pytest.approx(pnl.expected_return)
    assert pnl.loss_at_confidence == pytest.approx(0.15 + 2.3263478740408408 * pnl.stdev)
    assert st.reverse_stress(np.array([0.0, 0.0, 1.0]), cov, 0, 0.1) is None    # bonds gain when stocks fall


def test_cross_language_fixture():
    """Printed values are hard-coded in cpp/tests/test_stress.cpp."""
    cov = three_asset_cov()
    w = np.array([0.5, 0.3, 0.2])
    sh = st.conditional_shock(cov, [0, 2], [-0.12, 0.03], horizon_days=21)
    pnl = st.shock_pnl(w, sh, 0.975)
    r = np.array([[0.01, -0.02, 0.003], [-0.04, -0.06, 0.01], [0.02, 0.03, -0.002], [-0.01, 0.005, 0.0]])
    rp = st.replay(r, np.array([0.6, -0.2, 0.5]))
    rev = st.reverse_stress(w, cov, 0, 0.2)
    print(repr(sh.mean[1]), repr(sh.cond_cov[1, 1]), repr(pnl.loss_at_confidence), repr(rp.total_return),
          repr(rp.max_drawdown), repr(rp.worst_day), repr(rev))
    assert sh.mean[1] == pytest.approx(-0.1256043956043956, rel=1e-12)
    assert sh.cond_cov[1, 1] == pytest.approx(0.003469292307692308, rel=1e-12)
    assert pnl.loss_at_confidence == pytest.approx(0.12631429311790987, rel=1e-12)
    assert rp.total_return == pytest.approx(0.0022471820000000697, rel=1e-12)
    assert rp.max_drawdown == pytest.approx(0.009147620365793419, rel=1e-12)
    assert rp.worst_day == pytest.approx(-0.007380128521997098, rel=1e-12)
    assert rev == pytest.approx(-0.25477707006369427, rel=1e-12)


@pytest.mark.skipif(not os.path.exists(LONG), reason="long price history not present")
def test_covid_replay_on_real_data():
    from qre.data.hardening import load_price_csv
    t = load_price_csv(LONG)
    d = np.array(t.dates)
    i0, i1 = st.window_indices(d, "2020-02-19", "2020-03-23")
    spy = t.prices[i0:i1 + 1, t.tickers.index("SPY")]
    rp = st.replay((spy[1:] / spy[:-1] - 1)[:, None], np.array([1.0]))
    assert -0.36 < rp.total_return < -0.32                                      # S&P 500 fell about 34%
    assert rp.max_drawdown == pytest.approx(-rp.total_return)                   # peak to trough by construction
