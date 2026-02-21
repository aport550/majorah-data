import datetime as dt
import time
import pandas as pd
import numpy as np
import yfinance as yf

START_DATE = "2024-01-01"

def normalize_for_yahoo(t: str) -> str:
    """
    Yahoo uses '-' instead of '.' for class shares, e.g. BRK.B -> BRK-B
    """
    t = str(t).strip().upper()
    if not t:
        return ""
    return t.replace(".", "-")

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    universe = pd.read_csv("data/universe.csv")
    if "Ticker" not in universe.columns:
        raise RuntimeError(f"Expected column 'Ticker' in data/universe.csv. Found: {list(universe.columns)}")

    raw_tickers = (
        universe["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Deduplicate while preserving order
    seen = set()
    tickers = []
    for t in raw_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    # Map your tickers -> yahoo tickers
    yahoo_map = {t: normalize_for_yahoo(t) for t in tickers}

    # We'll download in batches to reduce random failures/timeouts
    BATCH_SIZE = 150  # You can tune 100-250
    PAUSE_SECS = 1.0  # small pause between batches (helps avoid throttling)

    all_adjclose = []

    for b, batch in enumerate(chunk_list(tickers, BATCH_SIZE), start=1):
        yahoo_batch = [yahoo_map[t] for t in batch]
        print(f"\nBatch {b}: downloading {len(yahoo_batch)} tickers...")

        # yfinance returns a DataFrame with columns like:
        # - if group_by="column": columns are OHLC... then tickers
        # We'll ask specifically for Adj Close (fallback to Close if missing)
        try:
            df = yf.download(
                tickers=" ".join(yahoo_batch),
                start=START_DATE,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:
            print(f"Batch {b} hard failed: {e}")
            time.sleep(PAUSE_SECS)
            continue

        if df is None or df.empty:
            print(f"Batch {b} returned empty.")
            time.sleep(PAUSE_SECS)
            continue

        # Try Adj Close first
        if isinstance(df.columns, pd.MultiIndex):
            if ("Adj Close" in df.columns.get_level_values(0)):
                px = df["Adj Close"].copy()
            elif ("Close" in df.columns.get_level_values(0)):
                px = df["Close"].copy()
            else:
                print(f"Batch {b} missing Adj Close/Close columns.")
                time.sleep(PAUSE_SECS)
                continue
        else:
            # If only one ticker, columns may be single-level
            # Try typical names:
            if "Adj Close" in df.columns:
                px = df["Adj Close"].to_frame()
            elif "Close" in df.columns:
                px = df["Close"].to_frame()
            else:
                print(f"Batch {b} unexpected columns: {list(df.columns)[:10]}")
                time.sleep(PAUSE_SECS)
                continue

        # px columns are yahoo tickers; convert index to date
        px.index = pd.to_datetime(px.index).date

        # Rename columns back to your original tickers where possible
        inverse = {v: k for k, v in yahoo_map.items()}
        px = px.rename(columns=lambda c: inverse.get(str(c).upper(), str(c).upper()))

        all_adjclose.append(px)

        time.sleep(PAUSE_SECS)

    if not all_adjclose:
        raise RuntimeError("No price data fetched from Yahoo (all batches empty/failed).")

    prices = pd.concat(all_adjclose, axis=1)

    # Keep only your tickers, in your original order, if present
    present = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]

    prices = prices[present].sort_index()

    print(f"\nPrices built. Present: {len(present)}  Missing: {len(missing)}")
    if missing:
        print("First 25 missing tickers:", missing[:25])

    # Daily returns (%)
    returns = prices.pct_change() * 100.0
    returns = returns.dropna(how="all")  # drop days where everything is NaN

    # Save daily returns matrix
    returns.to_csv("data/daily_returns.csv")
    print("Saved data/daily_returns.csv")

    # Correlation matrix (pairwise corr uses overlapping data by default)
    corr = returns.corr()
    corr.to_csv("data/correlation_matrix.csv")
    print("Saved data/correlation_matrix.csv")

if __name__ == "__main__":
    main()
