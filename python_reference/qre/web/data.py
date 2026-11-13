"""Phase 15: price data for an arbitrary portfolio.

Prices come from Yahoo Finance through yfinance (dividend- and split-adjusted closes), cached on disk so repeated
analyses of the same portfolio and period do not download again, or from the repo's bundled 6-stock file (offline).
The fetcher is a plain function (tickers, start, end) -> (dates, tickers, prices with NaN for missing), so tests and
other data sources can be plugged in without touching the rest.

Downloaded data is messier than the curated file: tickers start trading on different dates, foreign listings have
different holidays, and Yahoo occasionally returns a bad print. prepare() therefore
  1. drops every row before the date on which all tickers have a price (common start),
  2. drops the remaining rows where any ticker is missing (different holidays) and reports how many,
  3. runs the Phase 3.5 hardening (repairs isolated bad ticks) and refuses data that still fails the quality report.
"""
import csv
import datetime as dt
import hashlib
import math
import os
import re

import numpy as np

from qre.data import hardening as hd

TICKER_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,14}$")    # e.g. AAPL, BRK-B, BRK.B, ^GSPC, EURUSD=X
MAX_TICKERS = 25
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BUNDLED = os.path.join(REPO_ROOT, "data", "market_data", "adjusted_close.csv")


class DataError(ValueError):
    """Bad tickers or unusable data; the message is shown to the user."""


def normalize_tickers(tickers):
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t:
            continue
        if not TICKER_RE.match(t):
            raise DataError(f"'{t}' is not a valid ticker symbol")
        if t in out:
            raise DataError(f"{t} is listed twice")
        out.append(t)
    if not out:
        raise DataError("add at least one ticker")
    if len(out) > MAX_TICKERS:
        raise DataError(f"at most {MAX_TICKERS} tickers")
    return out


def yfinance_fetcher(tickers, start, end):
    """Adjusted daily closes from Yahoo Finance. Needs `pip install yfinance` and internet access."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataError("yfinance is not installed: run  pip install yfinance") from e
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True,
                      group_by="column")
    if raw is None or len(raw) == 0:
        raise DataError("Yahoo Finance returned no data (check the tickers and your internet connection)")
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if hasattr(close, "to_frame") and not hasattr(close, "columns"):
        close = close.to_frame(name=tickers[0])
    cols = [str(c) for c in close.columns]
    prices = np.full((len(close), len(tickers)), np.nan)
    for j, t in enumerate(tickers):
        if t in cols:
            prices[:, j] = close[t].to_numpy(dtype=float)
    dates = [d.strftime("%Y-%m-%d") for d in close.index]
    return dates, list(tickers), prices


def bundled_fetcher(tickers, start, end):
    """The repo's curated 6-stock file; works offline."""
    table = hd.load_price_csv(BUNDLED)
    missing = [t for t in tickers if t not in table.tickers]
    if missing:
        raise DataError(f"{', '.join(missing)} not in the bundled data (it has {', '.join(table.tickers)}); "
                        "switch the data source to Yahoo Finance")
    keep = [i for i, d in enumerate(table.dates) if start <= d < end]
    cols = [table.tickers.index(t) for t in tickers]
    return [table.dates[i] for i in keep], list(tickers), table.prices[np.ix_(keep, cols)]


def bundled_last_date():
    with open(BUNDLED) as fh:
        last = fh.read().strip().splitlines()[-1]
    return dt.date.fromisoformat(last.split(",", 1)[0])


class PriceCache:
    """CSV per (fetcher, tickers, start, end) under cache_dir. Today's data is re-fetched after `max_age_hours`."""

    def __init__(self, cache_dir, max_age_hours=12.0):
        self.dir = cache_dir
        self.max_age = max_age_hours * 3600.0
        os.makedirs(cache_dir, exist_ok=True)

    def path(self, source, tickers, start, end):
        key = hashlib.sha1(f"{source}|{','.join(tickers)}|{start}|{end}".encode()).hexdigest()[:16]
        return os.path.join(self.dir, f"{source}_{key}.csv")

    def get(self, source, tickers, start, end, fetcher, now=None):
        p = self.path(source, tickers, start, end)
        now = dt.datetime.now().timestamp() if now is None else now
        if os.path.exists(p) and now - os.path.getmtime(p) < self.max_age:
            t = hd.load_price_csv(p)
            return t.dates, t.tickers, t.prices, True
        dates, ticks, prices = fetcher(tickers, start, end)
        write_price_csv(p, dates, ticks, prices)
        return dates, ticks, np.asarray(prices, dtype=float), False


def write_price_csv(path, dates, tickers, prices):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Date"] + list(tickers))
        for d, row in zip(dates, np.asarray(prices, dtype=float)):
            w.writerow([d] + ["" if not math.isfinite(x) else repr(float(x)) for x in row])


def prepare(dates, tickers, prices, min_rows):
    """Align, clean and quality-check. Returns (dates, prices, notes) or raises DataError."""
    prices = np.asarray(prices, dtype=float)
    notes = []
    if prices.size == 0:
        raise DataError("no price data returned")
    empty = [t for j, t in enumerate(tickers) if not np.isfinite(prices[:, j]).any()]
    if empty:
        raise DataError(f"no prices found for {', '.join(empty)}; check the symbol")
    firsts = [int(np.flatnonzero(np.isfinite(prices[:, j]))[0]) for j in range(prices.shape[1])]
    start = max(firsts)
    if start > 0:
        late = tickers[int(np.argmax(firsts))]
        notes.append(f"History starts {dates[start]}, when {late} (the newest listing) begins trading.")
    dates, prices = list(dates[start:]), prices[start:]
    ok = np.isfinite(prices).all(axis=1)
    if not ok.all():
        notes.append(f"Dropped {int((~ok).sum())} day(s) on which some ticker had no price (different holidays "
                     "or data gaps).")
        dates = [d for d, k in zip(dates, ok) if k]
        prices = prices[ok]
    if len(dates) < min_rows:
        raise DataError(f"only {len(dates)} days of common history; need at least {min_rows} "
                        "(choose a longer period or tickers with longer histories)")
    cleaned, q = hd.harden(prices)
    if q.bad_ticks:
        notes.append(f"Repaired {len(q.bad_ticks)} isolated bad price print(s).")
    problems = []
    if q.non_positive:
        problems.append("non-positive prices")
    if q.stale:
        problems.append("stale (unchanged) price runs: " + ", ".join(sorted({tickers[s.asset] for s in q.stale})))
    if q.split_candidates:
        problems.append("possible unadjusted splits: " + ", ".join(sorted({tickers[s.asset] for s in q.split_candidates})))
    if problems:
        raise DataError("price data failed the quality checks (" + "; ".join(problems) + ")")
    # repaired bad ticks are now regular prices, so the engine's own quality report sees clean data
    recheck = hd.harden(cleaned)[1]
    if recheck.bad_ticks or recheck.stale or recheck.split_candidates or recheck.non_positive:
        raise DataError("price data still fails the quality checks after repair")
    return dates, cleaned, notes


def date_range(years, today=None):
    today = today or dt.date.today()
    start = today.replace(year=today.year - int(years)) if not (today.month == 2 and today.day == 29) else \
        today.replace(year=today.year - int(years), day=28)
    return start.isoformat(), (today + dt.timedelta(days=1)).isoformat()
