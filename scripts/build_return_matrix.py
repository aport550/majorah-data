import time
import os
import json
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf

START_DATE = "2024-01-01"
TRADING_DAYS = 252


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
        yield lst[i : i + n]


def main():
    universe = pd.read_csv("data/universe.csv")
    if "Ticker" not in universe.columns:
        raise RuntimeError(
            f"Expected column 'Ticker' in data/universe.csv. Found: {list(universe.columns)}"
        )

    raw_tickers = (
        universe["Ticker"].dropna().astype(str).str.strip().str.upper().tolist()
    )

    # Deduplicate while preserving order
    seen = set()
    tickers = []
    for t in raw_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    # Ensure SPY is present for defensive / beta / downside metrics
    if "SPY" not in seen:
        tickers.append("SPY")

    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    # Map your tickers -> yahoo tickers
    yahoo_map = {t: normalize_for_yahoo(t) for t in tickers}

    # We'll download in batches to reduce random failures/timeouts
    BATCH_SIZE = 150
    PAUSE_SECS = 1.0

    all_adjclose = []
    all_close = []

    for b, batch in enumerate(chunk_list(tickers, BATCH_SIZE), start=1):
        yahoo_batch = [yahoo_map[t] for t in batch]
        print(f"\nBatch {b}: downloading {len(yahoo_batch)} tickers...")

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

        # Keep both series for two different purposes:
        # - Adj Close drives total-return calculations.
        # - Close lets the frontend recover the actual quoted historical price.
        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns.get_level_values(0):
                px = df["Adj Close"].copy()
            elif "Close" in df.columns.get_level_values(0):
                px = df["Close"].copy()
            else:
                print(f"Batch {b} missing Adj Close/Close columns.")
                time.sleep(PAUSE_SECS)
                continue

            if "Close" in df.columns.get_level_values(0):
                close_px = df["Close"].copy()
            else:
                close_px = px.copy()
        else:
            # Single ticker case
            if "Adj Close" in df.columns:
                px = df["Adj Close"].to_frame()
            elif "Close" in df.columns:
                px = df["Close"].to_frame()
            else:
                print(f"Batch {b} unexpected columns: {list(df.columns)[:10]}")
                time.sleep(PAUSE_SECS)
                continue

            if "Close" in df.columns:
                close_px = df["Close"].to_frame()
            else:
                close_px = px.copy()

        # Convert index to date
        px.index = pd.to_datetime(px.index).date
        close_px.index = pd.to_datetime(close_px.index).date

        # Rename columns back to original tickers where possible
        inverse = {v: k for k, v in yahoo_map.items()}
        px = px.rename(columns=lambda c: inverse.get(str(c).upper(), str(c).upper()))
        close_px = close_px.rename(
            columns=lambda c: inverse.get(str(c).upper(), str(c).upper())
        )

        all_adjclose.append(px)
        all_close.append(close_px)
        time.sleep(PAUSE_SECS)

    if not all_adjclose:
        raise RuntimeError("No price data fetched from Yahoo (all batches empty/failed).")

    prices = pd.concat(all_adjclose, axis=1)
    close_prices = pd.concat(all_close, axis=1)

    # Keep only tickers in original order, if present
    present = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    prices = prices[present].sort_index()
    close_prices = close_prices.reindex(index=prices.index, columns=present)

    print(f"\nPrices built. Present: {len(present)}  Missing: {len(missing)}")
    if missing:
        print("First 25 missing tickers:", missing[:25])

    # Daily returns (%)
    returns = prices.pct_change() * 100.0
    returns = returns.dropna(how="all")

    # Save daily returns matrix
    os.makedirs("data", exist_ok=True)
    returns.to_csv("data/daily_returns.csv")
    print("Saved data/daily_returns.csv")

    # Also save publicly for frontend usage
    os.makedirs("public/data", exist_ok=True)
    public_returns_path = "public/data/daily_returns.csv"
    returns.to_csv(public_returns_path)
    print(f"Saved {public_returns_path} ({os.path.getsize(public_returns_path)} bytes)")

    # Compact metadata used by the frontend to convert the adjusted-price path
    # back to the actual historical closing price. Adjustment factors are
    # piecewise constant, so storing only change points is much smaller than a
    # second full daily price matrix.
    price_anchors = {}
    adjustment_factors = {}

    for ticker in present:
        adjusted = pd.to_numeric(prices[ticker], errors="coerce")
        raw_close = pd.to_numeric(close_prices[ticker], errors="coerce")
        valid_anchor = adjusted.dropna()

        if not valid_anchor.empty:
            anchor_date = valid_anchor.index[-1]
            price_anchors[ticker] = {
                "date": anchor_date.isoformat(),
                "adjusted_close": float(valid_anchor.iloc[-1]),
            }

        ratio = (adjusted / raw_close).replace([np.inf, -np.inf], np.nan).dropna()
        changes = []
        previous_factor = None

        for date, value in ratio.items():
            factor = round(float(value), 8)
            if factor <= 0:
                continue
            if previous_factor is None or abs(factor - previous_factor) >= 1e-7:
                changes.append([date.isoformat(), factor])
                previous_factor = factor

        if changes:
            adjustment_factors[ticker] = changes

    price_metadata = {
        "as_of": dt.date.today().isoformat(),
        "method": "historical_close = reconstructed_adjusted_close / adjustment_factor",
        "price_anchors": price_anchors,
        "adjustment_factors": adjustment_factors,
        "ticker_count": len(price_anchors),
    }

    price_metadata_path = "public/data/price_metadata.json"
    with open(price_metadata_path, "w", encoding="utf-8") as f:
        json.dump(price_metadata, f, separators=(",", ":"))
    print(
        f"Saved {price_metadata_path} "
        f"({os.path.getsize(price_metadata_path)} bytes)"
    )

    # Correlation matrix
    corr = returns.corr()
    corr.to_csv("data/correlation_matrix.csv")
    print("Saved data/correlation_matrix.csv")

    # =========================
    # Public outputs
    # =========================
    os.makedirs("public/data", exist_ok=True)

    # ---- Volatility (daily + annualized), based on % returns
    vol_daily = returns.std(skipna=True)  # % per day
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS)  # % per year

    vols_payload = {
        "as_of": dt.date.today().isoformat(),
        "start_date": START_DATE,
        "trading_days": TRADING_DAYS,
        "units": "percent",
        "vol_daily": {str(k): float(v) for k, v in vol_daily.dropna().items()},
        "vol_annual": {str(k): float(v) for k, v in vol_annual.dropna().items()},
        "ticker_count": int(len(vol_annual.dropna())),
    }

    vols_path = "public/data/vols.json"
    with open(vols_path, "w", encoding="utf-8") as f:
        json.dump(vols_payload, f, indent=2)
    print(f"Saved {vols_path} ({os.path.getsize(vols_path)} bytes)")

    # ---- Covariance matrix (daily), based on % returns
    # Units: (percent^2) per day
    cov_daily = returns.cov()
    cov_path = "public/data/covariance_matrix.csv"
    cov_daily.to_csv(cov_path)
    print(f"Saved {cov_path} ({os.path.getsize(cov_path)} bytes)")

    # ---- Expected returns (annualized), based on % returns
    # Mean daily return (%/day) -> annual (%/year) via *252
    mean_daily = returns.mean(skipna=True)  # % per day
    mean_annual = mean_daily * TRADING_DAYS  # % per year

    exp_payload = {
        "as_of": dt.date.today().isoformat(),
        "start_date": START_DATE,
        "trading_days": TRADING_DAYS,
        "units": "percent",
        "expected_return_daily_pct": {
            str(k): float(v) for k, v in mean_daily.dropna().items()
        },
        "expected_return_annual_pct": {
            str(k): float(v) for k, v in mean_annual.dropna().items()
        },
        "ticker_count": int(len(mean_annual.dropna())),
    }

    exp_path = "public/data/expected_returns.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(exp_payload, f, indent=2)
    print(f"Saved {exp_path} ({os.path.getsize(exp_path)} bytes)")


if __name__ == "__main__":
    main()
