import os

import numpy as np
import pytest

from qre.engine import compare
from qre.engine import config as cf
from qre.engine import engine as en
from qre.engine import report as rp
from qre.risk import backtest as bt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------- config
def test_config_parses_and_round_trips():
    c = cf.parse_config("""
        # comment
        data = prices.csv
        weights = AAA=0.75, BBB = 0.25    # trailing comment
        confidence = 0.99, 0.975
        models = historical, garch_t
        combination = exp
        eta = 0.5
    """)
    assert c.data == "prices.csv" and c.weights == [("AAA", 0.75), ("BBB", 0.25)]
    assert c.confidence == [0.99, 0.975] and c.models == ["historical", "garch_t"]
    assert c.combination == "exp" and c.eta == 0.5 and c.window == 250 and c.burn == 100
    again = cf.parse_config(cf.canonical(c))
    assert cf.canonical(again) == cf.canonical(c)


@pytest.mark.parametrize("text,msg", [
    ("data = x\nwindwo = 250", "line 2: unknown key 'windwo'"),
    ("data = x\nwindow = 2.5", "line 2: 'window' needs an integer"),
    ("data = x\nmodels = historical, magic", "line 2: unknown model 'magic'"),
    ("data = x\ndata = y", "line 2: 'data' given twice"),
    ("data = x\nconfidence = 0.99, , 0.95", "line 2: empty item"),
    ("data = x\nconfidence = 1.5", "line 2: confidence must be in (0.5, 1)"),
    ("data = x\nweights", "line 2: expected key = value"),
    ("window = 250", "config needs 'data'"),
    ("data = x\nmodels = ewma, ewma", "listed twice"),
])
def test_config_errors_name_the_line(text, msg):
    with pytest.raises(ValueError, match=msg.replace("(", r"\(").replace(")", r"\)")):
        cf.parse_config(text)


def test_shipped_configs_parse():
    for name in ("daily_risk.conf", "ci_smoke.conf"):
        cf.load_config(os.path.join(ROOT, "configs", name))


# ---------------------------------------------------------------- engine pieces
def test_fnv1a64_known_values():
    assert en.fnv1a64(b"") == 0xcbf29ce484222325
    assert en.fnv1a64(b"a") == 0xaf63dc4c8601ec8c
    assert en.fnv1a64(b"foobar") == 0x85944171f73967e8


def test_resolve_weights():
    c = cf.parse_config("data = x\nweights = B=0.6, A=0.4")
    assert en.resolve_weights(c, ["A", "B", "C"]) == pytest.approx([0.4, 0.6, 0.0])
    with pytest.raises(ValueError, match="unknown ticker"):
        en.resolve_weights(cf.parse_config("data = x\nweights = Z=1"), ["A"])
    with pytest.raises(ValueError, match="sum to 1"):
        en.resolve_weights(cf.parse_config("data = x\nweights = A=0.5"), ["A", "B"])


def test_forecast_var_es_is_the_backtest_day():
    rng = np.random.default_rng(1)
    p = 100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal((80, 3)), axis=0)
    w = np.array([0.5, 0.3, 0.2])
    s = bt.rolling_backtest(p, w, window=50, confidence=0.975, method="ewma")
    for i, t in enumerate(s.rows):
        assert bt.forecast_var_es(p, w, t, 50, 0.975, "ewma") == (s.var[i], s.es[i])
    nxt = bt.forecast_var_es(p, w, 80, 50, 0.975, "ewma")         # the next, unseen day
    assert np.all(np.isfinite(nxt))


def _small_run(tmp_path, models="historical, parametric, ewma", workers=1, combination="select"):
    rng = np.random.default_rng(7)
    p = 100.0 * np.cumprod(1.0 + 0.01 * rng.standard_normal((230, 2)), axis=0)
    data = tmp_path / "prices.csv"
    with open(data, "w") as fh:
        fh.write("Date,AAA,BBB\n")
        for i, row in enumerate(p):
            fh.write(f"2020-{1 + i // 28:02d}-{1 + i % 28:02d},{float(row[0])!r},{float(row[1])!r}\n")
    cfg = cf.parse_config(f"data = {data}\nconfidence = 0.99, 0.95\nwindow = 100\nmodels = {models}\n"
                          f"combination = {combination}\nlookback = 30\nburn = 40\noutput = {tmp_path / 'out'}")
    return cfg, en.run_engine(cfg, workers)


def test_engine_matches_rolling_backtest_and_combination(tmp_path):
    cfg, r = _small_run(tmp_path)
    assert r.rows[0] == 101 and r.rows[-1] == 229
    for cr in r.results:
        for j, m in enumerate(cfg.models):
            s = bt.rolling_backtest(r.prices, r.weights, 100, cr.confidence, m)
            assert np.array_equal(cr.var[:, j], s.var) and np.array_equal(cr.es[:, j], s.es)
            assert (cr.next_var[j], cr.next_es[j]) == bt.forecast_var_es(r.prices, r.weights, 230, 100, cr.confidence, m)
        assert cr.next_weights.sum() == pytest.approx(1.0)
        assert cr.next_combined[0] == pytest.approx(cr.next_weights @ cr.next_var)
        assert len(cr.summaries) == len(cfg.models) + 1
        assert cr.summaries[0].n_obs == r.rows.size - cfg.burn


def test_workers_do_not_change_results(tmp_path):
    _, a = _small_run(tmp_path, workers=1)
    _, b = _small_run(tmp_path, workers=3)
    for x, y in zip(a.results, b.results):
        assert np.array_equal(x.var, y.var) and np.array_equal(x.es, y.es)
        assert np.array_equal(x.combined_var, y.combined_var)


def test_report_files_and_self_comparison(tmp_path):
    cfg, r = _small_run(tmp_path)
    rp.write_report(r, cfg.output)
    files = sorted(os.listdir(cfg.output))
    assert files == ["backtest_summary.csv", "forecasts_95.csv", "forecasts_99.csv", "manifest.txt", "next_day.csv",
                     "run_log.txt"]
    with open(os.path.join(cfg.output, "forecasts_99.csv")) as fh:
        lines = fh.read().splitlines()
    assert lines[0].split(",")[:4] == ["date", "loss", "historical_var", "historical_es"]
    assert lines[1].endswith(",,") and not lines[41].endswith(",,")     # combination starts after burn
    with open(os.path.join(cfg.output, "manifest.txt")) as fh:
        man = fh.read()
    assert f"fnv1a64 = {r.data_hash:016x}" in man and "evaluation_days = 89" in man
    assert compare.main([cfg.output, cfg.output]) == 0


def test_engine_refuses_dirty_data(tmp_path):
    data = tmp_path / "bad.csv"
    rows = ["Date,AAA"] + [f"2020-01-{d:02d},{100 + d}" for d in range(1, 29)] * 1
    rows[10] = "2020-01-10,-5"
    data.write_text("\n".join(rows) + "\n")
    cfg = cf.parse_config(f"data = {data}\nwindow = 5\nburn = 2\nmodels = historical")
    with pytest.raises(ValueError, match="quality report"):
        en.run_engine(cfg)
