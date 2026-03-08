import os
import json
import time
import datetime as dt
import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/universe.csv"
OUT_PATH = "public/data/sectors.json"

PAUSE_SECS = 0.15


def normalize_ticker(t):
    return str(t or "").strip().upper()


def normalize_for_yahoo(t):
    return normalize_ticker(t).replace(".", "-")


def load_universe(path):
    df = pd.read_csv(path)

    if "Ticker" not in df.columns:
        raise RuntimeError("Ticker column missing in universe.csv")

    tickers = []
    seen = set()

    for t in df["Ticker"].dropna():
        t = normalize_ticker(t)
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)

    return tickers


def main():
    tickers = load_universe(UNIVERSE_PATH)

    print("Universe size:", len(tickers))

    sector_map = {}
    failed = []

    for i, ticker in enumerate(tickers, start=1):

        print(f"[{i}/{len(tickers)}] {ticker}")

        yt = normalize_for_yahoo(ticker)

        try:
            info = yf.Ticker(yt).info

            sector = info.get("sector")

            if sector:
                sector_map[ticker] = sector
            else:
                sector_map[ticker] = "Unknown"

        except Exception as e:
            print("FAILED:", ticker, e)
            failed.append(ticker)

        time.sleep(PAUSE_SECS)

    payload = {
        "as_of": dt.date.today().isoformat(),
        "sector": sector_map,
        "failed_tickers": failed
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("Saved:", OUT_PATH)
    print("Sectors:", len(sector_map))
    print("Failed:", len(failed))


if __name__ == "__main__":
    main()
