import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from qre.risk import var_es as ve
from qre.web import analytics, data, runner, server


def _gbm(n=520, k=3, seed=3):
    rng = np.random.default_rng(seed)
    rets = 0.0003 + 0.012 * rng.standard_normal((n - 1, k)) @ np.linalg.cholesky(0.4 + 0.6 * np.eye(k)).T
    prices = 100.0 * np.vstack([np.ones(k), np.cumprod(1.0 + rets, axis=0)])
    dates = [str(np.datetime64("2019-01-01") + np.timedelta64(i, "D")) for i in range(n)]
    return dates, prices


def fake_fetcher(tickers, start, end):
    dates, p = _gbm(k=len(tickers))
    return dates, list(tickers), p


# ---------------------------------------------------------------- data
def test_normalize_tickers():
    assert data.normalize_tickers([" aapl", "brk.b", "^GSPC", ""]) == ["AAPL", "BRK.B", "^GSPC"]
    for bad, msg in ((["AAPL", "aapl"], "twice"), (["DROP TABLE"], "not a valid"), ([], "at least one"),
                     ([f"T{i}" for i in range(30)], "at most")):
        with pytest.raises(data.DataError, match=msg):
            data.normalize_tickers(bad)


def test_prepare_aligns_listings_and_holidays():
    dates, p = _gbm(n=420)
    p = p.copy()
    p[:30, 2] = np.nan            # third ticker lists 30 days later
    p[100, 0] = np.nan            # a holiday on one exchange only
    d2, p2, notes = data.prepare(dates, ["A", "B", "C"], p, min_rows=300)
    assert d2[0] == dates[30] and len(d2) == 420 - 30 - 1
    assert np.isfinite(p2).all()
    assert any("starts" in n and "C" in n for n in notes) and any("Dropped 1 day" in n for n in notes)


def test_prepare_repairs_bad_ticks_and_rejects_bad_data():
    dates, p = _gbm(n=420)
    p = p.copy()
    p[200, 1] *= 1.6              # spike and revert
    d2, p2, notes = data.prepare(dates, ["A", "B", "C"], p, min_rows=300)
    assert any("Repaired 1" in n for n in notes) and p2[200, 1] == p2[199, 1]
    stale = p.copy()
    stale[300:310, 0] = stale[299, 0]
    with pytest.raises(data.DataError, match="stale"):
        data.prepare(dates, ["A", "B", "C"], stale, min_rows=300)
    with pytest.raises(data.DataError, match="no prices found for B"):
        data.prepare(dates, ["A", "B"], np.column_stack([p[:, 0], np.full(420, np.nan)]), min_rows=300)
    with pytest.raises(data.DataError, match="only 420 days"):
        data.prepare(dates, ["A", "B", "C"], _gbm(n=420)[1], min_rows=500)


def test_cache_reuses_fresh_files_and_refetches_stale(tmp_path):
    calls = []

    def fetch(t, s, e):
        calls.append(1)
        return fake_fetcher(t, s, e)
    c = data.PriceCache(str(tmp_path), max_age_hours=1)
    a = c.get("x", ["A", "B"], "2019-01-01", "2020-06-01", fetch)
    b = c.get("x", ["A", "B"], "2019-01-01", "2020-06-01", fetch)
    assert len(calls) == 1 and not a[3] and b[3]
    assert np.allclose(a[2], b[2], rtol=0, atol=0) and a[0] == b[0]
    c.get("x", ["A", "B"], "2019-01-01", "2020-06-01", fetch, now=os.path.getmtime(
        c.path("x", ["A", "B"], "2019-01-01", "2020-06-01")) + 7200)
    assert len(calls) == 2


def test_bundled_fetcher_and_missing_yfinance(monkeypatch):
    d, t, p = data.bundled_fetcher(["XOM", "AAPL"], "2023-01-01", "2024-01-01")
    assert t == ["XOM", "AAPL"] and d[0].startswith("2023-01") and p.shape == (len(d), 2)
    with pytest.raises(data.DataError, match="MSFT not in the bundled data"):
        data.bundled_fetcher(["MSFT"], "2021-01-01", "2024-01-01")
    monkeypatch.setitem(__import__("sys").modules, "yfinance", None)
    with pytest.raises(data.DataError, match="pip install yfinance"):
        data.yfinance_fetcher(["AAPL"], "2021-01-01", "2024-01-01")
    assert data.bundled_last_date().isoformat() == "2023-12-29"


# ---------------------------------------------------------------- analytics
def test_analytics_identities():
    dates, p = _gbm()
    w = np.array([0.5, 0.3, 0.2])
    a = analytics.analyze(dates, ["A", "B", "C"], p, w)
    rc = [x["risk_contribution"] for x in a["assets"]]
    assert sum(rc) == pytest.approx(1.0)
    assert a["diversification_ratio"] > 1.0
    rp = ve.portfolio_returns(p[1:] / p[:-1] - 1.0, w)
    assert a["volatility"]["realized_full"] == pytest.approx(np.std(rp, ddof=1) * np.sqrt(252))
    assert a["series"]["vol20"][-1] == pytest.approx(np.std(rp[-20:], ddof=1) * np.sqrt(252))
    assert a["series"]["vol20"][18] is None and a["series"]["vol20"][19] is not None
    v = np.array(a["series"]["value"])
    assert v[0] == 100.0 and a["returns"]["max_drawdown"] == pytest.approx(np.max(1 - v / np.maximum.accumulate(v)))
    assert len(a["series"]["ewma"]) == len(a["series"]["return_dates"])
    assert a["returns"]["worst_days"][0]["return"] == pytest.approx(rp.min())


# ---------------------------------------------------------------- engine runner and app
@pytest.fixture
def python_engine(monkeypatch):
    monkeypatch.setattr(runner, "find_qre_run", lambda path=None: None)


def _request(**kw):
    req = {"holdings": [{"ticker": "aaa", "weight": 2}, {"ticker": "BBB", "weight": 1}, {"ticker": "CCC", "weight": 1}],
           "source": "fake", "start": "2019-01-01", "end": "2020-06-01", "value": 50000, "confidence": [0.99, 0.95],
           "models": "fast", "combination": "exp"}
    req.update(kw)
    return req


def test_app_analyze_end_to_end(tmp_path, python_engine):
    app = server.App(str(tmp_path), fetchers={"fake": fake_fetcher})
    r = app.analyze(_request())
    assert r["input"]["tickers"] == ["AAA", "BBB", "CCC"] and r["input"]["weights"] == [0.5, 0.25, 0.25]
    assert r["risk"]["engine"] == "Python reference" and r["risk"]["models"] == runner.FAST_MODELS
    combined = [x for x in r["risk"]["next_day"] if x["model"] == "combined"]
    assert len(combined) == 2 and all(x["es"] >= x["var"] > 0 for x in combined)
    assert set(r["risk"]["forecasts"]) == {"99", "95"}
    assert len(r["risk"]["backtest"]) == 2 * 5
    assert json.dumps(r)                                   # serializable, no NaN
    fc = r["risk"]["forecasts"]["99"]
    assert fc["combined_var"][99] is None and fc["combined_var"][100] is not None


def test_app_validation(tmp_path, python_engine):
    app = server.App(str(tmp_path), fetchers={"fake": fake_fetcher})
    for kw, msg in (({"holdings": [{"ticker": "AAA", "weight": -1}]}, "non-negative"),
                    ({"holdings": [{"ticker": "AAA", "weight": "x"}]}, "numbers"),
                    ({"years": 1, "start": None}, "between 2 and 15"),
                    ({"value": -5}, "positive"), ({"confidence": [0.5]}, "between 0.90"),
                    ({"combination": "vote"}, "combination"), ({"models": "some"}, "'all' or 'fast'"),
                    ({"source": "bloomberg"}, "unknown data source")):
        with pytest.raises(data.DataError, match=msg):
            app.analyze(_request(**kw))


def test_cpp_and_python_engines_agree_in_the_app(tmp_path):
    exe = runner.find_qre_run()
    if exe is None:
        pytest.skip("qre_run not built")
    dates, p = _gbm()
    a = runner.run_risk(dates, ["A", "B", "C"], p, [0.5, 0.3, 0.2], (0.99,), runner.FAST_MODELS, qre_run=exe)
    orig = runner.find_qre_run
    runner.find_qre_run = lambda path=None: None
    try:
        b = runner.run_risk(dates, ["A", "B", "C"], p, [0.5, 0.3, 0.2], (0.99,), runner.FAST_MODELS)
    finally:
        runner.find_qre_run = orig
    assert a["engine"].startswith("C++") and b["engine"] == "Python reference"
    for x, y in zip(a["next_day"], b["next_day"]):
        assert x["var"] == pytest.approx(y["var"], rel=1e-9) and x["es"] == pytest.approx(y["es"], rel=1e-9)


def test_http_server(tmp_path, python_engine):
    app = server.App(str(tmp_path), fetchers={"fake": fake_fetcher})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(app))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "app.js" in page and "react@18.3.1" in page
        assert urllib.request.urlopen(base + "/app.js").headers["Content-Type"].endswith("javascript")
        assert json.load(urllib.request.urlopen(base + "/api/status"))["engine"] == "Python reference"
        assert len(json.load(urllib.request.urlopen(base + "/api/examples"))) >= 4
        post = lambda body: urllib.request.urlopen(urllib.request.Request(
            base + "/api/analyze", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}))
        assert json.load(post(_request()))["data"]["rows"] == 520
        with pytest.raises(urllib.error.HTTPError) as e:
            post(_request(holdings=[{"ticker": "bad ticker!", "weight": 1}]))
        assert e.value.code == 400 and "valid ticker" in json.load(e.value)["error"]
        for path in ("/../server.py", "/%2e%2e/server.py", "/nope"):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + path)
            assert e.value.code == 404
    finally:
        httpd.shutdown()


def test_yfinance_fetcher_reads_both_column_layouts(monkeypatch):
    pd = pytest.importorskip("pandas")
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    multi = pd.DataFrame([[1.0, 10.0, 1.1, 11.0], [2.0, 20.0, 2.1, np.nan], [3.0, 30.0, 3.1, 31.0]], index=idx,
                         columns=pd.MultiIndex.from_tuples([("Close", "AAA"), ("Close", "BBB"), ("Open", "AAA"),
                                                            ("Open", "BBB")]))
    flat = pd.DataFrame({"Open": [1.0, 2.0, 3.0], "Close": [5.0, 6.0, 7.0]}, index=idx)
    fake = type("yf", (), {})()
    for frame, ticks, expected in ((multi, ["AAA", "BBB"], [[1, 10], [2, 20], [3, 30]]),
                                   (flat, ["ZZZ"], [[5], [6], [7]])):
        fake.download = lambda *a, f=frame, **k: f
        monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)
        d, t, p = data.yfinance_fetcher(ticks, "2024-01-01", "2024-01-05")
        assert d == ["2024-01-02", "2024-01-03", "2024-01-04"] and t == ticks
        assert np.array_equal(p, np.array(expected, dtype=float))
    fake.download = lambda *a, **k: pd.DataFrame()
    with pytest.raises(data.DataError, match="no data"):
        data.yfinance_fetcher(["AAA"], "2024-01-01", "2024-01-05")


# ---------------------------------------------------------------- regressions found while building the app
GPD_FIXTURE = [0.629533, 0.922216, 0.200356, 0.996887, 0.721952, 0.189688, 1.264614, 0.794235, 1.525641, 1.147761,
               0.711705, 0.321565, 0.152588, 0.790589, 0.421305, 0.482209, 0.076213, 0.508952, 0.51812, 0.48874,
               0.652854, 2.431878, 0.680863, 0.053342, 0.581883]


def test_gpd_fit_skips_an_infeasible_moment_start():
    """From a min-variance portfolio's conditional-EVT window: the moment start (xi ~ -0.4) puts the largest
    exceedance outside the GPD support, which made the C++ optimizer stop the whole run."""
    import math
    from qre.risk import evt
    y = np.array(GPD_FIXTURE)
    m, v = y.mean(), y.var(ddof=1)
    xi0 = float(np.clip(0.5 * (1 - m * m / v), -0.4, 0.8))
    assert not math.isfinite(evt._gpd_nll(np.array([xi0, math.log(0.5 * m * (m * m / v + 1))]), y))
    xi, beta = evt.fit_gpd(y)
    assert xi == pytest.approx(-0.2505940805, abs=1e-6) and beta == pytest.approx(0.8551165871, rel=1e-6)


def test_garch_t_nu_is_clamped_not_walled():
    from qre.risk import garch
    r = 0.008 * np.random.default_rng(1313).standard_normal(250)
    fit = garch.fit_garch(r, "t")
    assert fit.nu == garch.NU_MAX
    assert np.isfinite(garch.neg_loglik_t(np.array([0.0, 0.0, 50.0]), r, float(np.mean(r * r))))
