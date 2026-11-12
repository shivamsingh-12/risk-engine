**Running the engine (Phase 14)**

From the repo root, after building (`cmake -B build -S cpp && cmake --build build`):

    build/apps/qre_run --config configs/daily_risk.conf             # C++, all cores
    build/apps/qre_run --config configs/daily_risk.conf --threads 1 # same numbers, one thread
    PYTHONPATH=python_reference python -m qre.engine.run --config configs/daily_risk.conf --workers 4
    PYTHONPATH=python_reference python -m qre.engine.compare runs/daily runs/daily_py   # compare two reports

Configs are `key = value` lines (see cpp/include/qre/engine/config.hpp for every key). Unknown keys and bad values
stop the run with the line number. Each run writes to `output`:

| File | Contents |
| :---- | :---- |
| report.md | next-day VaR/ES per model and combined, backtest table |
| next_day.csv | the same next-day numbers, with combination weights |
| backtest_summary.csv | breaches, Kupiec, Christoffersen, traffic light, Z2, mean FZ0 per model and combined |
| forecasts_<level>.csv | every backtest day: realized loss and each model's VaR/ES |
| manifest.txt | version, canonical config, data fingerprint (FNV-1a), data range, weights |
| run_log.txt | threads and timing (the only file that can differ between identical runs) |

`daily_risk.conf` is the full eight-model job; `ci_smoke.conf` is the fast job CI runs in both languages.
