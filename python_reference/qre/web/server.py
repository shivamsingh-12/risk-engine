"""Phase 15: the app's HTTP server (standard library only).

    GET  /                 the React front end (static/)
    GET  /api/status       which engine will run, whether yfinance is installed
    GET  /api/examples     preset portfolios
    POST /api/analyze      {holdings: [{ticker, weight}], years, value, confidence, models, combination, source,
                            refresh, live}
                           -> data notes, engine report (next-day VaR/ES, backtests, forecast series), analytics and
                              the latest day's move
    POST /api/stress       {holdings, value, source, shocks: [{ticker, shock}], horizon_days, confidence,
                            reverse_loss, refresh} -> historical scenarios, hypothetical shock, reverse stress test
    POST /api/optimize     {holdings, source, years, method, min_weight, max_weight, gamma}
                           -> suggested weights and current vs suggested volatility, VaR and risk shares (Phase 21)
    POST /api/stream/start {holdings, value, source, mode: simulated|live, rate, speed} -> starts qre_stream and
                           returns its WebSocket URL (Phase 22);  POST /api/stream/stop  stops it

Weights are relative and scaled to sum to 100%; negative weights are short positions (Phase 21).

Binds to 127.0.0.1 by default: this is a local tool, not a public service. Analyses run one at a time (the C++
engine already uses every core).
"""
import json
import math
import mimetypes
import os
import threading
import datetime as dt
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from qre.web import analytics, data, runner
from qre.web import optimize as optimize_app
from qre.web import streaming
from qre.web import stress as stress_app

STATIC = os.path.join(os.path.dirname(__file__), "static")
EXAMPLES = [
    {"name": "Bundled six stocks (offline)", "source": "bundled",
     "holdings": [{"ticker": t, "weight": 1} for t in ("AAPL", "CAT", "JNJ", "JPM", "PG", "XOM")]},
    {"name": "Big tech", "source": "yahoo",
     "holdings": [{"ticker": t, "weight": 1} for t in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META")]},
    {"name": "60/40 stocks and bonds", "source": "yahoo",
     "holdings": [{"ticker": "SPY", "weight": 60}, {"ticker": "AGG", "weight": 40}]},
    {"name": "Defensive consumer", "source": "yahoo",
     "holdings": [{"ticker": t, "weight": 1} for t in ("PG", "KO", "PEP", "WMT", "JNJ")]},
    {"name": "All-in on one stock", "source": "yahoo", "holdings": [{"ticker": "TSLA", "weight": 1}]},
]
MAX_GROSS = 5.0                    # longs plus shorts, as a multiple of capital
MODEL_SETS = {"all": runner.ALL_MODELS, "fast": runner.FAST_MODELS}


def _clean_json(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean_json(v) for v in x]
    return x


class App:
    def __init__(self, cache_dir, fetchers=None, qre_run=None, quotes=None, clock=None, qre_stream=None):
        self.cache = data.PriceCache(cache_dir)
        self.fetchers = fetchers or {"yahoo": data.yfinance_fetcher, "bundled": data.bundled_fetcher}
        self.stress_fetchers = {"bundled": data.long_history_fetcher}
        if "yahoo" in self.fetchers:
            self.stress_fetchers["yahoo"] = self.fetchers["yahoo"]
        if fetchers and "stored" in fetchers:                  # tests plug in their own long history
            self.stress_fetchers["bundled"] = fetchers["stored"]
        self.quotes = quotes or data.yfinance_quotes          # live intraday quotes (Yahoo only)
        self.clock = clock or data.new_york_now
        self.qre_run = qre_run
        self.lock = threading.Lock()
        self.streams = streaming.StreamManager(qre_stream, quotes=self.quotes)

    def status(self):
        try:
            import yfinance  # noqa: F401
            yf = True
        except ImportError:
            yf = False
        exe = runner.find_qre_run(self.qre_run)
        return {"engine": "C++ (qre_run)" if exe else "Python reference", "qre_run": exe, "yfinance": yf,
                "default_models": "all" if exe else "fast", "stream": self.streams.status()}

    @staticmethod
    def _holdings(req):
        holdings = req.get("holdings") or []
        tickers = data.normalize_tickers([h.get("ticker", "") for h in holdings])
        try:
            raw_w = [float(h.get("weight", 0)) for h in holdings if str(h.get("ticker", "")).strip()]
        except (TypeError, ValueError):
            raise data.DataError("weights must be numbers") from None
        if any(not math.isfinite(x) for x in raw_w) or sum(raw_w) <= 0:
            raise data.DataError("weights must be numbers with a positive sum (negative weights are short positions)")
        total = sum(raw_w)
        w = [x / total for x in raw_w]
        if sum(abs(x) for x in w) > MAX_GROSS:
            raise data.DataError(f"gross exposure (longs plus shorts) is {sum(abs(x) for x in w):.0%} of capital; "
                                 f"the limit is {MAX_GROSS:.0%}")
        return tickers, w

    @staticmethod
    def _value(req):
        value = float(req.get("value", 100000))
        if not (math.isfinite(value) and value > 0):
            raise data.DataError("portfolio value must be positive")
        return value

    def analyze(self, req):
        tickers, weights = self._holdings(req)
        years = int(req.get("years", 3))
        if not 2 <= years <= 15:
            raise data.DataError("history must be between 2 and 15 years")
        value = self._value(req)
        conf = [float(c) for c in req.get("confidence", [0.99, 0.975])]
        if not conf or any(not 0.9 <= c < 1.0 for c in conf) or len(set(conf)) != len(conf):
            raise data.DataError("confidence levels must be distinct and between 0.90 and 0.999")
        combination = req.get("combination", "equal")
        if combination not in ("equal", "select", "exp"):
            raise data.DataError("combination must be equal, select or exp")
        mset = req.get("models") or self.status()["default_models"]
        if mset not in MODEL_SETS:
            raise data.DataError("models must be 'all' or 'fast'")
        source = req.get("source", "yahoo")
        if source not in self.fetchers:
            raise data.DataError(f"unknown data source '{source}'")
        live = bool(req.get("live", False))
        if live and source == "bundled":
            raise data.DataError("live mode needs Yahoo Finance as the data source (the bundled file ends in 2023)")
        refresh = bool(req.get("refresh", False)) or live

        t0 = time.perf_counter()
        start, end = req.get("start"), req.get("end")
        now = self.clock()
        if not (start and end):
            today = data.bundled_last_date() if source == "bundled" else now.date()   # the bundled file ends in 2023
            start, end = data.date_range(years, today)
        dates, ticks, prices, cached = self.cache.get(source, tickers, start, end, self.fetchers[source],
                                                      refresh=refresh)
        if list(ticks) != tickers:
            raise data.DataError("data source returned different tickers than requested")
        live_info = None
        if live:
            is_open = data.market_open(now)
            live_info = {"market_open": is_open, "checked": now.strftime("%Y-%m-%d %H:%M:%S"), "action": "none",
                         "asof": None}
            if is_open:
                try:
                    q = self.quotes(tickers)
                except Exception as e:                  # quotes are a bonus: keep the daily data if they fail
                    live_info["error"] = f"{type(e).__name__}: {e}"
                else:
                    dates, prices, info = data.apply_live(dates, tickers, prices, q, now.date())
                    live_info.update(info)
        dates, prices, notes = data.prepare(dates, tickers, prices, min_rows=250 + 100 + 30)
        if live_info is not None:
            live_info["provisional"] = bool(live_info["market_open"] and dates[-1] == now.date().isoformat())
        t1 = time.perf_counter()
        with self.lock:
            risk = runner.run_risk(dates, tickers, prices, weights, conf, MODEL_SETS[mset], combination,
                                   qre_run=self.qre_run)
        t2 = time.perf_counter()
        stats = analytics.analyze(dates, tickers, prices, weights)
        t3 = time.perf_counter()
        last = np.asarray(prices[-1]) / np.asarray(prices[-2]) - 1.0
        latest = {"date": dates[-1], "previous": dates[-2], "return": float(np.dot(weights, last)),
                  "pnl": float(np.dot(weights, last)) * value,
                  "holdings": [{"ticker": t, "return": float(x)} for t, x in zip(tickers, last)]}
        return _clean_json({
            "input": {"tickers": tickers, "weights": weights, "value": value, "confidence": conf,
                      "combination": combination, "models": mset, "source": source, "years": years, "live": live},
            "data": {"start": dates[0], "end": dates[-1], "rows": len(dates), "cached": cached, "notes": notes},
            "latest": latest, "live": live_info,
            "risk": risk, "analytics": stats,
            "timing_s": {"data": t1 - t0, "engine": t2 - t1, "analytics": t3 - t2},
        })


    def optimize(self, req):
        tickers, weights = self._holdings(req)
        years = int(req.get("years", 3))
        if not 2 <= years <= 15:
            raise data.DataError("history must be between 2 and 15 years")
        source = req.get("source", "yahoo")
        if source not in self.fetchers:
            raise data.DataError(f"unknown data source '{source}'")
        try:
            min_w = float(req.get("min_weight", 0.0))
            max_w = float(req.get("max_weight", 1.0))
            gamma = float(req.get("gamma", 5.0))
        except (TypeError, ValueError):
            raise data.DataError("weight limits and risk aversion must be numbers") from None
        if not gamma > 0:
            raise data.DataError("risk aversion must be positive")
        start, end = req.get("start"), req.get("end")
        if not (start and end):
            today = data.bundled_last_date() if source == "bundled" else self.clock().date()
            start, end = data.date_range(years, today)
        dates, ticks, prices, cached = self.cache.get(source, tickers, start, end, self.fetchers[source])
        if list(ticks) != tickers:
            raise data.DataError("data source returned different tickers than requested")
        dates, prices, notes = data.prepare(dates, tickers, prices, min_rows=120)
        p = np.asarray(prices, dtype=float)
        rets = p[1:] / p[:-1] - 1.0
        try:
            out = optimize_app.suggest(rets, weights, req.get("method", "min_variance"), min_w, max_w, gamma)
        except ValueError as e:
            raise data.DataError(str(e)) from None
        out["tickers"] = tickers
        out["data"] = {"start": dates[0], "end": dates[-1], "notes": notes, "cached": cached}
        return _clean_json(out)

    def stream_start(self, req):
        tickers, weights = self._holdings(req)
        value = self._value(req)
        source = req.get("source", "yahoo")
        if source not in self.fetchers:
            raise data.DataError(f"unknown data source '{source}'")
        mode = req.get("mode", "simulated")
        if mode == "live" and source != "yahoo":
            raise data.DataError("live quotes need Yahoo Finance as the data source")
        today = data.bundled_last_date() if source == "bundled" else self.clock().date()
        start, end = data.date_range(2, today)
        dates, ticks, prices, _ = self.cache.get(source, tickers, start, end, self.fetchers[source])
        if list(ticks) != tickers:
            raise data.DataError("data source returned different tickers than requested")
        dates, prices, _ = data.prepare(dates, tickers, prices, min_rows=251)
        try:
            rate, speed = float(req.get("rate", 1000)), float(req.get("speed", 0.02))
        except (TypeError, ValueError):
            raise data.DataError("rate and speed must be numbers") from None
        return _clean_json(self.streams.start(dates, tickers, prices, weights, value, mode, rate, speed))

    def stress(self, req):
        tickers, weights = self._holdings(req)
        value = self._value(req)
        source = req.get("source", "yahoo")
        if source not in self.stress_fetchers:
            raise data.DataError(f"unknown data source '{source}'")
        shocks = []
        for sh in req.get("shocks") or []:
            t = data.normalize_tickers([sh.get("ticker", "")])[0]
            try:
                x = float(sh.get("shock"))
            except (TypeError, ValueError):
                raise data.DataError(f"the shock for {t} must be a number") from None
            if not (math.isfinite(x) and -1.0 < x <= 5.0):
                raise data.DataError(f"the shock for {t} must be a return between -100% and +500%")
            if t in [s for s, _ in shocks]:
                raise data.DataError(f"{t} is shocked twice")
            shocks.append((t, x))
        horizon = float(req.get("horizon_days", 10))
        if not 1 <= horizon <= 250:
            raise data.DataError("the shock horizon must be between 1 and 250 days")
        confidence = float(req.get("confidence", 0.99))
        if not 0.9 <= confidence < 1.0:
            raise data.DataError("confidence must be between 0.90 and 0.999")
        reverse = float(req.get("reverse_loss", 0.2))
        if not 0 < reverse < 1:
            raise data.DataError("the reverse stress loss must be between 0% and 100%")
        universe = list(tickers)
        for t in [stress_app.PROXY] + [t for t, _ in shocks]:
            if t not in universe:
                universe.append(t)
        if len(universe) > data.MAX_TICKERS + 5:
            raise data.DataError("too many tickers")
        today = self.clock().date() if source == "yahoo" else None
        end = (today + dt.timedelta(days=1)).isoformat() if today else "9999-12-31"
        t0 = time.perf_counter()
        dates, ticks, prices, cached = self.cache.get(source, universe, data.STRESS_START, end,
                                                      self.stress_fetchers[source], refresh=bool(req.get("refresh")))
        if list(ticks) != universe:
            raise data.DataError("data source returned different tickers than requested")
        prices = np.asarray(prices, dtype=float)
        empty = [t for j, t in enumerate(universe) if not np.isfinite(prices[:, j]).any()]
        if empty:
            raise data.DataError(f"no prices found for {', '.join(empty)}; check the symbol")
        try:
            out = stress_app.run_stress(dates, universe, prices, weights, value, shocks, horizon, confidence, reverse)
        except ValueError as e:
            raise data.DataError(str(e)) from None
        out["timing_s"] = time.perf_counter() - t0
        out["cached"] = cached
        out["input"] = {"tickers": tickers, "weights": weights, "value": value, "source": source}
        return _clean_json(out)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = "qre-web/1.0"

        def log_message(self, fmt, *args):     # quiet; errors are returned to the browser
            pass

        def _send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                return self._send(200, app.status())
            if path == "/api/examples":
                return self._send(200, EXAMPLES)
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.normpath(os.path.join(STATIC, name))
            if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            with open(full, "rb") as fh:
                self._send(200, fh.read(), mimetypes.guess_type(full)[0] or "application/octet-stream")

        def do_POST(self):
            routes = {"/api/analyze": app.analyze, "/api/stress": app.stress, "/api/optimize": app.optimize,
                      "/api/stream/start": app.stream_start, "/api/stream/stop": lambda req: app.streams.stop()}
            if self.path not in routes:
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 100_000:
                    return self._send(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(req, dict):
                    raise data.DataError("request must be a JSON object")
                return self._send(200, routes[self.path](req))
            except (data.DataError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                  # engine or unexpected failure: report, keep serving
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return Handler


def serve(host="127.0.0.1", port=8000, cache_dir=None, qre_run=None, open_browser=False, qre_stream=None):
    cache_dir = cache_dir or os.path.join(data.REPO_ROOT, "runs", "cache")
    app = App(cache_dir, qre_run=qre_run, qre_stream=qre_stream)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{httpd.server_address[1]}/"
    st = app.status()
    print(f"qre portfolio risk app at {url}  (engine: {st['engine']}, yfinance: {'yes' if st['yfinance'] else 'no'})")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.streams.stop()
        httpd.server_close()
