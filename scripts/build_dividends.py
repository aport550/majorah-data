import os
import json
import time
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf

UNIVERSE_PATH = "data/universe.csv"
OUT_PATH = "public/data/dividends.json"

LOOKBACK_DAYS = 365
PRICE_LOOKBACK_DAYS = 10
PAUSE_SECS = 0.15


def normalize_ticker(t: str) -> str:
    t = str(t or "").strip().upper()
    return t


def normalize_for_yahoo(t: str) -> str:
    """
    Yahoo uses '-' instead of '.' for class shares, e.g. BRK.B -> BRK-B
    """
    t = normalize_ticker(t)
    return t.replace(".", "-")


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


def fetch_recent_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch recent adjusted/close prices for all tickers in one bulk call when possible.
    Returns original ticker -> latest valid price.
    """
    if not tickers:
        return {}

    yahoo_map = {t: normalize_for_yahoo(t) for t in tickers}
    reverse_map = {v: k for k, v in yahoo_map.items()}

    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = dt.date.today() - dt.timedelta(days=PRICE_LOOKBACK_DAYS)

    try:
        df = yf.download(
            tickers=" ".join(yahoo_map.values()),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as e:
        print(f"Bulk price fetch failed: {e}")
        return {}

    if df is None or df.empty:
        print("Bulk price fetch returned empty data.")
        return {}

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            px = df["Adj Close"].copy()
        elif "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            print("Bulk price fetch missing Adj Close/Close.")
            return {}
    else:
        if "Adj Close" in df.columns:
            px = df["Adj Close"].to_frame()
        elif "Close" in df.columns:
            px = df["Close"].to_frame()
        else:
            print("Bulk price fetch had unexpected columns.")
            return {}

    prices = {}
    for c in px.columns:
        ser = pd.to_numeric(px[c], errors="coerce").dropna()
        if ser.empty:
            continue
        yahoo_ticker = str(c).upper()
        orig = reverse_map.get(yahoo_ticker, yahoo_ticker)
        prices[orig] = float(ser.iloc[-1])

    return prices


def get_latest_price_with_fallback(orig_ticker: str, bulk_prices: dict[str, float]) -> float | None:
    if orig_ticker in bulk_prices and np.isfinite(bulk_prices[orig_ticker]):
        return float(bulk_prices[orig_ticker])

    yt = normalize_for_yahoo(orig_ticker)

    try:
        hist = yf.Ticker(yt).history(period="10d", auto_adjust=False)
        if hist is not None and not hist.empty:
            col = "Adj Close" if "Adj Close" in hist.columns else "Close"
            ser = pd.to_numeric(hist[col], errors="coerce").dropna()
            if not ser.empty:
                return float(ser.iloc[-1])
    except Exception:
        pass

    return None


def compute_trailing_annual_dividend_per_share(orig_ticker: str) -> float | None:
    """
    Uses actual dividends paid in the last LOOKBACK_DAYS.
    This is a trailing annual dividend, not forward indicated yield.
    """
    yt = normalize_for_yahoo(orig_ticker)

    try:
        divs = yf.Ticker(yt).dividends
    except Exception as e:
        print(f"{orig_ticker}: dividends fetch failed: {e}")
        return None

    if divs is None or len(divs) == 0:
        return 0.0

    divs = pd.to_numeric(divs, errors="coerce").dropna()
    if divs.empty:
        return 0.0

    cutoff = pd.Timestamp(dt.datetime.now() - dt.timedelta(days=LOOKBACK_DAYS))
    recent = divs[divs.index >= cutoff]

    if recent.empty:
        return 0.0

    return float(recent.sum())


def safe_round(x, digits=6):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), digits)


def main():
    tickers = load_universe(UNIVERSE_PATH)
    if not tickers:
        raise RuntimeError("Universe is empty.")

    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Bulk prices first for speed/reliability
    bulk_prices = fetch_recent_prices(tickers)
    print(f"Bulk prices fetched for {len(bulk_prices)} tickers")

    dividend_yield_pct = {}
    annual_dividend_per_share = {}
    current_price = {}
    zero_dividend = []
    missing_price = []
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        try:
            price = get_latest_price_with_fallback(ticker, bulk_prices)
            ann_div = compute_trailing_annual_dividend_per_share(ticker)

            if ann_div is None:
                failed.append(ticker)
                time.sleep(PAUSE_SECS)
                continue

            if price is None or not np.isfinite(price) or price <= 0:
                missing_price.append(ticker)
                annual_dividend_per_share[ticker] = safe_round(ann_div, 6)
                time.sleep(PAUSE_SECS)
                continue

            yld = (ann_div / price) * 100.0

            current_price[ticker] = safe_round(price, 6)
            annual_dividend_per_share[ticker] = safe_round(ann_div, 6)
            dividend_yield_pct[ticker] = safe_round(yld, 6)

            if ann_div == 0:
                zero_dividend.append(ticker)

        except Exception as e:
            print(f"{ticker}: failed with error: {e}")
            failed.append(ticker)

        time.sleep(PAUSE_SECS)

    payload = {
        "as_of": dt.date.today().isoformat(),
        "method": "trailing_12m_dividends_over_latest_price",
        "lookback_days": LOOKBACK_DAYS,
        "price_lookback_days": PRICE_LOOKBACK_DAYS,
        "units": "percent",
        "ticker_count_universe": len(tickers),
        "ticker_count_with_yield": len(dividend_yield_pct),
        "ticker_count_with_price_missing": len(missing_price),
        "ticker_count_failed": len(failed),
        "dividend_yield_pct": dict(sorted(dividend_yield_pct.items())),
        "annual_dividend_per_share": dict(sorted(annual_dividend_per_share.items())),
        "current_price": dict(sorted(current_price.items())),
        "missing_price_tickers": missing_price,
        "failed_tickers": failed,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {OUT_PATH} ({os.path.getsize(OUT_PATH)} bytes)")
    print(f"Yields computed: {len(dividend_yield_pct)}")
    print(f"Zero-dividend tickers: {len(zero_dividend)}")
    print(f"Missing price tickers: {len(missing_price)}")
    print(f"Failed tickers: {len(failed)}")
    if failed:
        print("First 25 failed:", failed[:25])


if __name__ == "__main__":
    main()
