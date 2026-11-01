import math

import numpy as np
import pytest

from qre.data import hardening as h

NAN = math.nan


def _prices():
    """Clean deterministic 3-asset history, 60 rows, small smooth moves."""
    rng = np.random.default_rng(0)
    rets = rng.standard_normal((59, 3)) * 0.01
    p = np.vstack([np.full(3, 100.0), 100.0 * np.cumprod(1.0 + rets, axis=0)])
    return p


# ---------------------------------------------------------------- loading
def test_load_price_csv(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("Date,AAA,BBB\n2024-01-02,10.5,20\n2024-01-03,,21\n2024-01-04,11,22.5\n")
    t = h.load_price_csv(f)
    assert t.tickers == ["AAA", "BBB"]
    assert t.dates == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert t.prices.shape == (3, 2)
    assert math.isnan(t.prices[1, 0]) and t.prices[2, 1] == 22.5


@pytest.mark.parametrize("body", [
    "Date,AAA\n2024-01-02,1\n2024-01-02,2\n",          # duplicate date
    "Date,AAA\n2024-01-03,1\n2024-01-02,2\n",          # out of order
    "Date,AAA\n2024-01-02,1,5\n",                       # too many cells
    "Date,AAA\n2024-01-02,abc\n",                       # non-numeric
    "Date,AAA\n01/02/2024,1\n",                         # bad date format
    "Date,AAA,AAA\n2024-01-02,1,2\n",                   # duplicate ticker
    "Date,AAA\n",                                       # no rows
])
def test_load_price_csv_rejects_malformed(tmp_path, body):
    f = tmp_path / "p.csv"
    f.write_text(body)
    with pytest.raises(ValueError):
        h.load_price_csv(f)


# ---------------------------------------------------------------- corporate actions
def test_split_back_adjustment_removes_the_jump():
    raw = np.array([[100.0], [102.0], [51.5], [52.0]])          # 2-for-1 at row 2
    adj = h.back_adjust(raw, [h.CorporateAction(0, 2, "split", 2.0)])
    assert adj[:, 0] == pytest.approx([50.0, 51.0, 51.5, 52.0])
    rets = adj[1:, 0] / adj[:-1, 0] - 1.0
    assert np.all(np.abs(rets) < 0.03)


def test_reverse_split():
    raw = np.array([[10.0], [10.2], [20.6]])                     # 1-for-2 reverse split at row 2
    adj = h.back_adjust(raw, [h.CorporateAction(0, 2, "split", 0.5)])
    assert adj[:, 0] == pytest.approx([20.0, 20.4, 20.6])


def test_dividend_adjustment_gives_total_return():
    raw = np.array([[100.0], [101.0], [99.0], [100.0]])          # $2 dividend, ex-date row 2
    adj = h.back_adjust(raw, [h.CorporateAction(0, 2, "dividend", 2.0)])
    f = 1.0 - 2.0 / 101.0
    assert adj[:, 0] == pytest.approx([100.0 * f, 101.0 * f, 99.0, 100.0])
    # total return on the ex-date: (99 + 2) / 101 - 1
    assert adj[2, 0] / adj[1, 0] - 1.0 == pytest.approx((99.0 + 2.0) / 101.0 - 1.0)


def test_multiple_actions_compound_and_only_touch_their_asset():
    raw = np.array([[100.0, 50.0], [100.0, 50.0], [50.0, 50.0], [25.0, 50.0]])
    acts = [h.CorporateAction(0, 2, "split", 2.0), h.CorporateAction(0, 3, "split", 2.0)]
    adj = h.back_adjust(raw, acts)
    assert adj[:, 0] == pytest.approx([25.0, 25.0, 25.0, 25.0])
    assert adj[:, 1] == pytest.approx(raw[:, 1])


@pytest.mark.parametrize("act", [
    h.CorporateAction(5, 2, "split", 2.0),
    h.CorporateAction(0, 0, "split", 2.0),
    h.CorporateAction(0, 2, "split", 0.0),
    h.CorporateAction(0, 2, "dividend", 500.0),
    h.CorporateAction(0, 2, "merger", 1.0),
])
def test_invalid_actions_raise(act):
    with pytest.raises(ValueError):
        h.back_adjust(np.full((4, 1), 100.0), [act])


def test_detect_split_candidates_finds_splits_not_normal_moves():
    p = _prices()
    p[30:, 1] /= 4.0                                   # unadjusted 4-for-1 in asset 1
    p[45:, 2] *= 2.0                                   # unadjusted 1-for-2 reverse split in asset 2
    cands = h.detect_split_candidates(p)
    assert [(c.asset, c.row, c.ratio) for c in cands] == [(1, 30, 4.0), (2, 45, 0.5)]
    assert h.detect_split_candidates(_prices()) == []


def test_crash_that_looks_like_split_is_only_flagged():
    p = _prices()
    p[20:, 0] *= 0.5                                   # a genuine -50% crash
    clean, rep = h.harden(p)
    assert [(c.asset, c.row, c.ratio) for c in rep.split_candidates] == [(0, 20, 2.0)]
    assert clean[20, 0] == pytest.approx(p[20, 0])     # never auto-corrected


# ---------------------------------------------------------------- quality checks
def test_stale_runs():
    p = _prices()
    p[10:18, 0] = p[9, 0]                              # 8 repeated rows
    p[40:42, 1] = p[39, 1]                             # 2 repeated rows: below threshold
    runs = h.stale_runs(p, min_run=5)
    assert runs == [h.StaleRun(0, 10, 8)]


def test_stale_run_at_end_of_series_is_reported():
    p = _prices()
    p[54:, 2] = p[53, 2]
    assert h.stale_runs(p, min_run=5) == [h.StaleRun(2, 54, 6)]


def test_bad_tick_detected_and_repaired():
    p = _prices()
    p[25, 0] *= 1.4                                    # one-day spike that reverts
    p[33, 2] *= 0.7                                    # one-day drop that reverts
    clean, rep = h.harden(p)
    assert rep.bad_ticks == [(0, 25), (2, 33)]
    assert clean[25, 0] == pytest.approx(p[24, 0])     # removed, then forward filled
    assert clean[33, 2] == pytest.approx(p[32, 2])


def test_split_is_not_a_bad_tick():
    p = _prices()
    p[30:, 1] /= 2.0
    assert h.bad_ticks(p) == []


def test_forward_fill_within_life_respects_listing_and_delisting():
    p = _prices()
    p[:5, 0] = NAN          # listed late
    p[50:, 1] = NAN         # delisted
    p[20:23, 2] = NAN       # short gap -> filled
    p[30:40, 2] = NAN       # long gap -> left
    out = h.forward_fill_within_life(p, 5)
    assert np.isnan(out[:5, 0]).all()
    assert np.isnan(out[50:, 1]).all()                 # no fabricated prices after delisting
    assert out[20:23, 2] == pytest.approx([p[19, 2]] * 3)
    assert np.isnan(out[30:40, 2]).all()


def test_quality_report_missing_breakdown():
    p = _prices()
    p[:5, 0] = NAN
    p[50:, 1] = NAN
    p[20:23, 2] = NAN
    p[30:40, 2] = NAN
    _, rep = h.harden(p)
    assert rep.leading_missing == [5, 0, 0]
    assert rep.trailing_missing == [0, 10, 0]
    assert rep.internal_missing == [0, 0, 13]
    assert rep.remaining_internal_missing == [0, 0, 10]


def test_non_positive_prices_become_missing():
    p = _prices()
    p[12, 1] = 0.0
    p[13, 2] = -5.0
    clean, rep = h.harden(p)
    assert rep.non_positive == [(1, 12), (2, 13)]
    assert clean[12, 1] == pytest.approx(p[11, 1])


def test_clean_data_passes_untouched():
    p = _prices()
    clean, rep = h.harden(p)
    assert np.array_equal(clean, p)
    assert not (rep.non_positive or rep.stale or rep.bad_ticks or rep.split_candidates)


# ---------------------------------------------------------------- point in time
def test_pit_window_uses_only_past_rows():
    p = _prices()
    w = h.pit_window(p, 30, 10)
    assert w.shape == (11, 3)
    assert np.array_equal(w, p[19:30])                 # rows 19..29, never row 30
    assert h.pit_window(p, 60, 10).shape == (11, 3)    # forecasting the unseen next day
    for t, win in [(5, 10), (61, 5), (30, 0)]:
        with pytest.raises(ValueError):
            h.pit_window(p, t, win)


def test_universe_at_excludes_assets_not_alive_in_window():
    p = _prices()
    p[:25, 0] = NAN         # listed at row 25
    p[40:, 1] = NAN         # delisted after row 39
    assert h.universe_at(p, 20, 10) == [1, 2]
    assert h.universe_at(p, 37, 10) == [0, 1, 2]
    assert h.universe_at(p, 55, 10) == [0, 2]


def test_returns_from_full_back_adjustment_are_point_in_time_but_levels_are_not():
    raw = _prices()
    raw[40:, 0] /= 2.0                                 # future split (row 40)
    raw[20, 1] -= 1.0                                  # past dividend (ex-date row 20, $1)
    acts = [h.CorporateAction(0, 40, "split", 2.0), h.CorporateAction(1, 20, "dividend", 1.0)]
    full = h.back_adjust(raw, acts)
    t = 30
    pit = h.adjust_as_of(raw, acts, t)
    r_full = full[1:t] / full[: t - 1] - 1.0
    r_pit = pit[1:t] / pit[: t - 1] - 1.0
    assert r_full == pytest.approx(r_pit, rel=1e-12)   # returns before t are identical
    assert not np.allclose(full[:t, 0], pit[:t, 0])    # but levels before t used future info
