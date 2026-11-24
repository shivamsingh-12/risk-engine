**Portfolio risk app (Phases 15 and 20 to 22)**

An interactive page for testing the risk of any portfolio: enter tickers and weights, and it shows tomorrow's VaR
and Expected Shortfall from all eight engine models and their combination, volatility (realized, EWMA, GARCH),
each holding's share of the risk, correlations, drawdowns and how the forecasts would have done in a backtest.

Phase 20 added:
- **Stress tests:** what ten past crises (2007 to 2025) would do to the portfolio today, a "what if" for shocks to
  any tickers (e.g. SPY −20%, or SPY −15% and TLT −10% together), and a reverse stress test (the market fall that
  would cost a given amount). A holding that did not trade during a crisis is replaced by its beta to SPY, and the
  page says so.
Phase 21 added:
- **Short positions:** a negative weight is a short; weights are scaled to sum to 100% (gross exposure up to 500%).
- **Component VaR and ES:** each holding's part of the 1-day VaR (normal) and ES (historical), in dollars.
- **10-day risk:** VaR and ES over 10 days by five methods (Experiment 26).
- **Suggest weights:** minimum variance, risk parity, mean-variance, inverse volatility or equal weights for the same
  holdings, with a largest-weight limit and optional shorts, and a button that applies them.

Phase 22 added:
- **Live stream:** starts `qre_stream` for the portfolio and shows value, P&L and 1-day VaR updating on every tick
  over a WebSocket, with the tick-to-risk latency. The source is a simulated market with the holdings' own
  covariance (a trading day passes in about a minute) or live Yahoo Finance quotes polled every 5 seconds. Needs
  `qre_stream` built (`cmake --build build`).

Also from Phase 20:
- **Refresh data:** downloads prices again instead of using the copy saved in the last 12 hours.
- **Live mode:** re-runs the analysis every minute. While the US market is open, the latest intraday prices
  become a provisional row for today, so today's P&L and the risk numbers move during the day; outside trading
  hours it keeps the last close.

**Run it** (from the repo root, after building the C++ engine with `cmake --build build`):

    pip install -e "python_reference[web]"      # once: the package and yfinance (the bundled data works without it)
    qre-web --open                              # or: python -m qre.web --open

The page opens at http://127.0.0.1:8000/. Options: `--port`, `--qre-run PATH` (default `build/apps/qre_run`),
`--cache-dir DIR` (default `runs/cache`). Without a built `qre_run` it falls back to the Python engine, which is
much slower for the GARCH models, so it defaults to the four fast models in that case.

**How it works**

| Piece | File | Job |
| :---- | :---- | :---- |
| Server | server.py | standard-library HTTP server; `/api/status`, `/api/examples`, `POST /api/analyze`, `POST /api/stress`, `POST /api/optimize`, `POST /api/stream/start` and `/stop`; local only |
| Data | data.py | yfinance download (adjusted closes and intraday quotes) or stored files, disk cache, alignment, quality checks |
| Stress tests | stress.py | scenario replay, shocks and reverse stress on top of `qre.risk.stress` |
| Optimizer | optimize.py | suggested weights on top of `qre.portfolio.construction` |
| Streaming | streaming.py | starts and stops `qre_stream` and pipes live quotes into it |
| Engine | runner.py | writes a config, runs `qre_run` (or the Python engine), reads the report files back |
| Analytics | analytics.py | volatility measures, GARCH persistence, risk contributions, correlations, drawdowns |
| Front end | static/ | React 18 + htm from unpkg, no build step; plain SVG charts |

Notes: weights are relative (they are scaled to sum to 1); negative weights are shorts. History is 2 to 15 years; a ticker that
listed recently shortens the common history for every holding (stress tests use each holding's own history back
to 2007 instead). All eight models take roughly 5 to 30 seconds, mostly GARCH-t; the four fast models take about
a second. With the bundled data source, stress tests use the stored 2005-2026 history of about 50 tickers
(`data/market_data/long_history_close.csv`). Live mode needs Yahoo Finance, checks only the regular session
(9:30 to 16:00 New York time, Monday to Friday; exchange holidays are not known to it) and adds today's row only
when every holding has a quote from today.
