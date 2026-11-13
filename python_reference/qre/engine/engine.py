"""Phase 14 reference: the production engine, mirroring cpp/include/qre/engine/engine.hpp.

config -> read and fingerprint the price file -> harden (refuse dirty data) -> VaR/ES forecasts for every
confidence x model x day (each backtest day plus the next, unseen day) -> per confidence: daily FZ0 scores,
combination, backtest summaries over the evaluation days (after `burn`), next-day combined forecast.
Forecasts are independent per (confidence, model), so they can run in separate processes (workers > 1) with
identical results. The C++ engine parallelizes the same work over threads.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from qre.data import hardening as hd
from qre.engine.config import RunConfig
from qre.risk import adaptive
from qre.risk import backtest as bt
from qre.risk import scoring

VERSION = "1.0.0"


def fnv1a64(data):
    """64-bit FNV-1a of a bytes object: the fingerprint that ties a report to its exact input file."""
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


@dataclass
class ConfidenceResult:
    confidence: float
    var: np.ndarray            # (T, K) backtest days
    es: np.ndarray
    scores: np.ndarray         # FZ0, (T, K)
    combined_var: np.ndarray   # evaluation days
    combined_es: np.ndarray
    summaries: list            # K models then the combination (BacktestSummary)
    mean_fz0: list
    next_var: np.ndarray       # (K,) day after the last price
    next_es: np.ndarray
    next_weights: np.ndarray
    next_combined: tuple


@dataclass
class EngineResult:
    config: RunConfig
    dates: list
    tickers: list
    prices: np.ndarray
    weights: np.ndarray
    quality: object
    data_hash: int
    data_bytes: int
    rows: np.ndarray
    loss: np.ndarray
    results: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def is_clean(q):
    """Same rule as QualityReport::clean() in C++: no gaps inside listed lives and nothing flagged."""
    return not any(q.internal_missing) and not (q.non_positive or q.stale or q.bad_ticks or q.split_candidates)


def resolve_weights(cfg, tickers):
    n = len(tickers)
    if cfg.weights == "equal":
        return np.full(n, 1.0 / n)
    w = np.zeros(n)
    for name, x in cfg.weights:
        if name not in tickers:
            raise ValueError(f"weight for unknown ticker '{name}'")
        j = tickers.index(name)
        if w[j] != 0.0:
            raise ValueError(f"ticker '{name}' weighted twice")
        w[j] = x
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    return w


def _forecast_series(args):
    """All days (backtest days and the next day) for one (confidence, model)."""
    prices, weights, first, window, confidence, model, mc_paths, seed, lam = args
    out = np.empty((prices.shape[0] - first + 1, 2))
    for i, t in enumerate(range(first, prices.shape[0] + 1)):
        out[i] = bt.forecast_var_es(prices, weights, t, window, confidence, model, mc_paths, seed, 252, lam)
    return out


def run_engine(cfg, workers=1):
    t0 = time.perf_counter()
    with open(cfg.data, "rb") as fh:
        raw = fh.read()
    table = hd.load_price_csv(cfg.data)
    dates, tickers = table.dates, table.tickers
    prices, quality = hd.harden(table.prices)
    if not is_clean(quality):
        raise ValueError("price data failed the quality report; fix the data before running the engine")
    weights = resolve_weights(cfg, list(tickers))
    bt.check_backtest_inputs(prices, weights, cfg.window)
    first = cfg.window + 1
    rows = np.arange(first, prices.shape[0])
    if rows.size <= cfg.burn:
        raise ValueError("burn must leave at least one evaluation day")
    loss = np.array([bt.realized_loss(prices, weights, t) for t in rows])
    t1 = time.perf_counter()

    jobs = [(prices, weights, first, cfg.window, c, m, cfg.mc_paths, cfg.seed, cfg.lam)
            for c in cfg.confidence for m in cfg.models]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            series = list(pool.map(_forecast_series, jobs))
    else:
        series = [_forecast_series(j) for j in jobs]
    t2 = time.perf_counter()

    r = EngineResult(cfg, list(dates), list(tickers), prices, weights, quality, fnv1a64(raw), len(raw), rows, loss)
    k = len(cfg.models)
    for ci, c in enumerate(cfg.confidence):
        block = series[ci * k:(ci + 1) * k]
        var = np.column_stack([s[:-1, 0] for s in block])
        es = np.column_stack([s[:-1, 1] for s in block])
        scores = scoring.fz0_loss(loss[:, None], var, es, c)
        cv, ce, _ = adaptive.combine(var, es, loss, c, cfg.combination, cfg.lookback, cfg.eta, cfg.burn)
        summaries, mean_fz0 = [], []
        ev = slice(cfg.burn, None)
        for v_, e_ in [(var[ev, j], es[ev, j]) for j in range(k)] + [(cv, ce)]:
            s = bt.BacktestSeries(rows[ev], v_, e_, loss[ev])
            summaries.append(bt.summarize(s, c))
            mean_fz0.append(float(scoring.fz0_loss(loss[ev], v_, e_, c).mean()))
        nv = np.array([s[-1, 0] for s in block])
        ne = np.array([s[-1, 1] for s in block])
        nw = adaptive.combination_weights(scores, scores.shape[0], cfg.combination, cfg.lookback, cfg.eta)
        r.results.append(ConfidenceResult(c, var, es, scores, cv, ce, summaries, mean_fz0, nv, ne, nw,
                                          (float(nw @ nv), float(nw @ ne))))
    t3 = time.perf_counter()
    r.timings = {"workers": workers, "load_ms": 1e3 * (t1 - t0), "forecast_ms": 1e3 * (t2 - t1),
                 "combine_ms": 1e3 * (t3 - t2)}
    return r
