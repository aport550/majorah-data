import os
import json
import time
import datetime as dt
import traceback

import pandas as pd
import numpy as np
import yfinance as yf

UNIVERSE_PATH = "data/universe.csv"
OUT_PATH = "public/data/dividends.json"

LOOKBACK_DAYS = 365
PRICE_LOOKBACK_DAYS = 15
PAUSE_SECS = 0.2


def normalize_ticker(t: str) -> str:
    return str(t or "").strip().upper()


def normalize_for_yahoo(t: str) -> str:
    """
    Yahoo uses '-' instead of '.' for class shares, e.g. BRK.B -> BRK-B
    """
    return normalize_ticker(t).replace(".", "-")


def load_universe(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise RuntimeError(
            f"Expected column 'Ticker' in {path}. Found: {list(df.columns)}"
        )

    raw = df["Ticker"].dropna().astype(str).tolist()

    seen = set()
    tickers = []
    for t in raw:
        n = normalize_ticker(t)
        if n and n not in seen:
            seen.add(n)
            tickers.append(n)

    return tickers


def safe_round(x, digits=6):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), digits)


def get_latest_price(orig_ticker: str) -> float | None:
    yt = normalize_for_yahoo(orig_ticker)

    try:
        hist = yf.Ticker(yt).history(period=f"{PRICE_LOOKBACK_DAYS}d", auto_adjust=False)
        if hist is None or hist.empty:
            return None

        col = "Adj Close" if "Adj Close" in hist.columns else "Close"
        ser = pd.to_numeric(hist[col], errors="coerce").dropna()
        if ser.empty:
            return None

        return float(ser.iloc[-1])
    except Exception as e:
        print(f"{orig_ticker}: latest price fetch failed: {e}")
        return None


def compute_trailing_annual_dividend_per_share(orig_ticker: str) -> float | None:
    """
    Uses actual dividend events paid in the last LOOKBACK_DAYS by reading the
    'Dividends' column from history(actions=True). This is trailing yield,
    not forward indicated yield.
    """
    yt = normalize_for_yahoo(orig_ticker)

    try:
        hist = yf.Ticker(yt).history(
            period=f"{LOOKBACK_DAYS + 30}d",
            auto_adjust=False,
            actions=True,
        )
    except Exception as e:
        print(f"{orig_ticker}: dividend history fetch failed: {e}")
        return None

    if hist is None or hist.empty:
        return 0.0

    if "Dividends" not in hist.columns:
        return 0.0

    divs = pd.to_numeric(hist["Dividends"], errors="coerce").fillna(0.0)

    if divs.empty:
        return 0.0

    # normalize index to timezone-naive
    idx = pd.to_datetime(divs.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    divs.index = idx

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = divs[divs.index >= cutoff]

    if recent.empty:
        return 0.0

    return float(recent.sum())


def main():
    tickers = load_universe(UNIVERSE_PATH)
    if not tickers:
        raise RuntimeError("Universe is empty.")

    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    dividend_yield_pct = {}
    annual_dividend_per_share = {}
    current_price = {}

    zero_dividend = []
    missing_price_tickers = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        try:
            ann_div = compute_trailing_annual_dividend_per_share(ticker)
            if ann_div is None:
                failed_tickers.append(ticker)
                time.sleep(PAUSE_SECS)
                continue

            price = get_latest_price(ticker)
            if price is None or not np.isfinite(price) or price <= 0:
                annual_dividend_per_share[ticker] = safe_round(ann_div, 6)
                missing_price_tickers.append(ticker)
                time.sleep(PAUSE_SECS)
                continue

            yld = (ann_div / price) * 100.0

            annual_dividend_per_share[ticker] = safe_round(ann_div, 6)
            current_price[ticker] = safe_round(price, 6)
            dividend_yield_pct[ticker] = safe_round(yld, 6)

            if ann_div == 0:
                zero_dividend.append(ticker)

        except Exception as e:
            print(f"{ticker}: failed with error: {e}")
            traceback.print_exc()
            failed_tickers.append(ticker)

        time.sleep(PAUSE_SECS)

    payload = {
        "as_of": dt.date.today().isoformat(),
        "method": "trailing_12m_dividends_from_history_actions_over_latest_price",
        "lookback_days": LOOKBACK_DAYS,
        "price_lookback_days": PRICE_LOOKBACK_DAYS,
        "units": "percent",
        "ticker_count_universe": len(tickers),
        "ticker_count_with_yield": len(dividend_yield_pct),
        "ticker_count_with_price_missing": len(missing_price_tickers),
        "ticker_count_failed": len(failed_tickers),
        "dividend_yield_pct": dict(sorted(dividend_yield_pct.items())),
        "annual_dividend_per_share": dict(sorted(annual_dividend_per_share.items())),
        "current_price": dict(sorted(current_price.items())),
        "missing_price_tickers": missing_price_tickers,
        "failed_tickers": failed_tickers,
        "zero_dividend_tickers": zero_dividend,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes)")
    print(f"Yields computed: {len(dividend_yield_pct)}")
    print(f"Zero-dividend tickers: {len(zero_dividend)}")
    print(f"Missing price tickers: {len(missing_price_tickers)}")
    print(f"Failed tickers: {len(failed_tickers)}")
    if failed_tickers:
        print("First 25 failed:", failed_tickers[:25])


if __name__ == "__main__":
    main()
