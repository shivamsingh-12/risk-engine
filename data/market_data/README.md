**Market data**

| File | Contents | Source |
| :---- | :---- | :---- |
| adjusted_close.csv | daily adjusted closes for AAPL, JPM, XOM, JNJ, PG, CAT, 2021 to 2023 (753 days) | Yahoo Finance, by `tools/pull_bundled_data.py` |
| long_history_close.csv | daily adjusted closes for 50 stocks, ETFs and indices, January 2005 to 13 November 2026 (blank before a ticker listed) | Yahoo Finance, by `tools/fetch_data.py` |

`adjusted_close.csv` is the bundled data that every experiment up to Phase 18, the default config and CI use. `long_history_close.csv` is used by Experiment 25 and the app's stress tests. Both are committed so that results reproduce without a network connection.

`tools/fetch_data.py` downloads the larger data sets used from Phase 20 on (long price histories, volumes, option chains, factors, yields, balance sheets) into `data/`, and records what it fetched in `data/FETCHED.md`.
