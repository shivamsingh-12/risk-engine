**Market data**

| File | Contents | Source |
| :---- | :---- | :---- |
| adjusted_close.csv | daily adjusted closes for AAPL, JPM, XOM, JNJ, PG, CAT, 2021 to 2023 (753 days) | Yahoo Finance, by `tools/pull_bundled_data.py` |

This file is the bundled data that every experiment up to Phase 18, the default config and CI use. It is committed so that results reproduce without a network connection.

`tools/fetch_data.py` downloads the larger data sets used from Phase 20 on (long price histories, volumes, option chains, factors, yields, balance sheets) into `data/`, and records what it fetched in `data/FETCHED.md`.
