import pandas as pd
import numpy as np
import csv
prices = pd.read_csv("data/market_data/adjusted_close.csv", index_col="Date", parse_dates=True)
#drop missing vals
log_returns = np.log(prices/prices.shift(1)).dropna()
windows = [10,20,60,90]
rows=[]
for window in windows:
    for ticker in log_returns.columns:
        rolling_vol = log_returns[ticker].rolling(window).std().dropna()
        rows.append({
            "window": window,
            "ticker": ticker,
            "mean_vol": rolling_vol.mean(),
            "stddev_of_vol": rolling_vol.std(),
        })

with open("python_reference/experiments/03-statistics/results/vol_stab_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)