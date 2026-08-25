#Pulls daily adjusted prices for small basket of tickers, saves as CSV
import yfinance as yf
import pandas as pd

#diverse sectors -> moves in diff directions (test covariance)
TICKERS = [
    "AAPL", "JPM", "XOM", "JNJ", "PG", "CAT",
]
START_DATE = "2021-01-01"
END_DATE = "2024-01-01"
OUTPUT_PATH = "data/market_data/adjusted_close.csv"

def main() -> None:
    print(f"Downloading {len(TICKERS)} tickers")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, threads=False)
    #only want Adj Close
    adj_close = data["Close"]
    missing_counts = adj_close.isna().sum()
    if missing_counts.any():
        print("missing values per ticker: ")
        print(missing_counts[missing_counts > 0])
    adj_close.to_csv(OUTPUT_PATH)
    print(f"\nSaved {adj_close.shape[0]} rows x {adj_close.shape[1]} tickers to {OUTPUT_PATH}")
    print(f"Date range: {adj_close.index.min()} to {adj_close.index.max()}")

if __name__ == "__main__":
    main()
