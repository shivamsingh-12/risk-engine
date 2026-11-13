**Portfolio risk app (Phase 15)**

An interactive page for testing the risk of any portfolio: enter tickers and weights, and it shows tomorrow's VaR
and Expected Shortfall from all eight engine models and their combination, volatility (realized, EWMA, GARCH),
each holding's share of the risk, correlations, drawdowns and how the forecasts would have done in a backtest.

**Run it** (from the repo root, after building the C++ engine with `cmake --build build`):

    pip install yfinance                        # once, for live prices; the bundled six stocks work without it
    PYTHONPATH=python_reference python -m qre.web --open

The page opens at http://127.0.0.1:8000/. Options: `--port`, `--qre-run PATH` (default `build/apps/qre_run`),
`--cache-dir DIR` (default `runs/cache`). Without a built `qre_run` it falls back to the Python engine, which is
much slower for the GARCH models, so it defaults to the four fast models in that case.

**How it works**

| Piece | File | Job |
| :---- | :---- | :---- |
| Server | server.py | standard-library HTTP server; `/api/status`, `/api/examples`, `POST /api/analyze`; local only |
| Data | data.py | yfinance download (adjusted closes) or bundled file, disk cache, alignment, quality checks |
| Engine | runner.py | writes a config, runs `qre_run` (or the Python engine), reads the report files back |
| Analytics | analytics.py | volatility measures, GARCH persistence, risk contributions, correlations, drawdowns |
| Front end | static/ | React 18 + htm from unpkg, no build step; plain SVG charts |

Notes: weights are relative and long-only (they are scaled to sum to 1). History is 2 to 15 years; a ticker that
listed recently shortens the common history for every holding. All eight models take roughly 5 to 30 seconds,
mostly GARCH-t; the four fast models take about a second.
