"""Phase 15: the app's HTTP server (standard library only).

    GET  /                 the React front end (static/)
    GET  /api/status       which engine will run, whether yfinance is installed
    GET  /api/examples     preset portfolios
    POST /api/analyze      {holdings: [{ticker, weight}], years, value, confidence, models, combination, source}
                           -> data notes, engine report (next-day VaR/ES, backtests, forecast series) and analytics

Binds to 127.0.0.1 by default: this is a local tool, not a public service. Analyses run one at a time (the C++
engine already uses every core).
"""
import json
import math
import mimetypes
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from qre.web import analytics, data, runner

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
    def __init__(self, cache_dir, fetchers=None, qre_run=None):
        self.cache = data.PriceCache(cache_dir)
        self.fetchers = fetchers or {"yahoo": data.yfinance_fetcher, "bundled": data.bundled_fetcher}
        self.qre_run = qre_run
        self.lock = threading.Lock()

    def status(self):
        try:
            import yfinance  # noqa: F401
            yf = True
        except ImportError:
            yf = False
        exe = runner.find_qre_run(self.qre_run)
        return {"engine": "C++ (qre_run)" if exe else "Python reference", "qre_run": exe, "yfinance": yf,
                "default_models": "all" if exe else "fast"}

    def analyze(self, req):
        holdings = req.get("holdings") or []
        tickers = data.normalize_tickers([h.get("ticker", "") for h in holdings])
        try:
            raw_w = [float(h.get("weight", 0)) for h in holdings if str(h.get("ticker", "")).strip()]
        except (TypeError, ValueError):
            raise data.DataError("weights must be numbers") from None
        if any(not math.isfinite(x) or x < 0 for x in raw_w) or sum(raw_w) <= 0:
            raise data.DataError("weights must be non-negative and not all zero (long-only portfolios)")
        total = sum(raw_w)
        weights = [x / total for x in raw_w]
        years = int(req.get("years", 3))
        if not 2 <= years <= 15:
            raise data.DataError("history must be between 2 and 15 years")
        value = float(req.get("value", 100000))
        if not (math.isfinite(value) and value > 0):
            raise data.DataError("portfolio value must be positive")
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

        t0 = time.perf_counter()
        start, end = req.get("start"), req.get("end")
        if not (start and end):
            today = data.bundled_last_date() if source == "bundled" else None   # the bundled file ends in 2023
            start, end = data.date_range(years, today)
        dates, ticks, prices, cached = self.cache.get(source, tickers, start, end, self.fetchers[source])
        if list(ticks) != tickers:
            raise data.DataError("data source returned different tickers than requested")
        dates, prices, notes = data.prepare(dates, tickers, prices, min_rows=250 + 100 + 30)
        t1 = time.perf_counter()
        with self.lock:
            risk = runner.run_risk(dates, tickers, prices, weights, conf, MODEL_SETS[mset], combination,
                                   qre_run=self.qre_run)
        t2 = time.perf_counter()
        stats = analytics.analyze(dates, tickers, prices, weights)
        t3 = time.perf_counter()
        return _clean_json({
            "input": {"tickers": tickers, "weights": weights, "value": value, "confidence": conf,
                      "combination": combination, "models": mset, "source": source, "years": years},
            "data": {"start": dates[0], "end": dates[-1], "rows": len(dates), "cached": cached, "notes": notes},
            "risk": risk, "analytics": stats,
            "timing_s": {"data": t1 - t0, "engine": t2 - t1, "analytics": t3 - t2},
        })


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
            if self.path != "/api/analyze":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 100_000:
                    return self._send(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(req, dict):
                    raise data.DataError("request must be a JSON object")
                return self._send(200, app.analyze(req))
            except (data.DataError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:                  # engine or unexpected failure: report, keep serving
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
    return Handler


def serve(host="127.0.0.1", port=8000, cache_dir=None, qre_run=None, open_browser=False):
    cache_dir = cache_dir or os.path.join(data.REPO_ROOT, "runs", "cache")
    app = App(cache_dir, qre_run=qre_run)
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
        httpd.server_close()
