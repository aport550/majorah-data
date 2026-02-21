import datetime as dt
from io import StringIO
import pandas as pd
import requests
import numpy as np

START_DATE = dt.date(2024, 1, 1)

def stooq_close_series(symbol: str) -> pd.Series:
    s = symbol.lower()
    if "." not in s:
        s = f"{s}.us"

    url = f"https://stooq.com/q/d/l/?s={s}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.Series(dtype="float64")

    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df[df["Date"] >= START_DATE].dropna(subset=["Close"]).sort_values("Date")

    return pd.Series(
        df["Close"].astype(float).values,
        index=df["Date"],
        name=symbol.upper()
    )

def main():
    # ---- Load Universe ----
    universe = pd.read_csv("data/universe.csv")
    if "Ticker" not in universe.columns:
        raise RuntimeError("Expected column 'Ticker' in universe.csv")

    tickers = (
        universe["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    print(f"Fetching prices for {len(tickers)} tickers...")

    closes = {}
    failed = []

    for t in tickers:
        try:
            s = stooq_close_series(t)
            if not s.empty:
                closes[t] = s
            else:
                failed.append(t)
        except Exception:
            failed.append(t)

    if not closes:
        raise RuntimeError("No price data fetched.")

    print(f"Success: {len(closes)} | Failed: {len(failed)}")

    # ---- Build price matrix ----
    prices = pd.DataFrame(closes).sort_index()

    # ---- Daily returns (%) ----
    returns = prices.pct_change() * 100.0
    returns = returns.dropna(how="all")

    # ---- Save daily return matrix ----
    returns.to_csv("data/daily_returns.csv")

    print("Saved daily_returns.csv")

    # ---- Correlation matrix ----
    corr_matrix = returns.corr()
    corr_matrix.to_csv("data/correlation_matrix.csv")

    print("Saved correlation_matrix.csv")

if __name__ == "__main__":
    main()
