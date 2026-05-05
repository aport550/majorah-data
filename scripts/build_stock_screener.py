import os
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


BATCH_SIZE = 75
PAUSE_SECS = 1.0


def normalize_for_yahoo(t: str) -> str:
    """
    Yahoo uses '-' instead of '.' for class shares, e.g. BRK.B -> BRK-B
    """
    t = str(t).strip().upper()
    if not t:
        return ""
    return t.replace(".", "-")


def safe_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        if isinstance(x, (np.integer, np.floating)):
            return float(x)
        return float(x)
    except Exception:
        return None


def safe_div(a, b):
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def get_first(info: dict, keys):
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def build_row(ticker: str) -> dict:
    yahoo_ticker = normalize_for_yahoo(ticker)
    yf_ticker = yf.Ticker(yahoo_ticker)

    try:
        info = yf_ticker.get_info()
    except Exception as e:
        print(f"{ticker}: failed info fetch: {e}")
        info = {}

    try:
        bs = yf_ticker.balance_sheet
    except Exception:
        bs = pd.DataFrame()

    try:
        fin = yf_ticker.financials
    except Exception:
        fin = pd.DataFrame()

    try:
        cf = yf_ticker.cashflow
    except Exception:
        cf = pd.DataFrame()

    def latest_statement_value(df, possible_rows):
        if df is None or df.empty:
            return None

        for row_name in possible_rows:
            if row_name in df.index:
                series = df.loc[row_name].dropna()
                if len(series) > 0:
                    return safe_float(series.iloc[0])

        return None

    # Yahoo info fields
    name = get_first(info, ["longName", "shortName", "displayName"])
    market_cap = safe_float(info.get("marketCap"))
    enterprise_value = safe_float(info.get("enterpriseValue"))

    revenue = safe_float(
        get_first(info, ["totalRevenue", "revenue"])
    )

    eps = safe_float(
        get_first(info, ["trailingEps", "epsTrailingTwelveMonths"])
    )

    ebitda = safe_float(info.get("ebitda"))

    cash = safe_float(
        get_first(info, ["totalCash", "cash"])
    )

    debt = safe_float(
        get_first(info, ["totalDebt", "debt"])
    )

    shares_outstanding = safe_float(
        get_first(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    )

    equity = safe_float(
        get_first(info, ["bookValue"])
    )

    # If bookValue is per share, convert to total equity when possible
    if equity is not None and shares_outstanding is not None:
        equity = equity * shares_outstanding

    current_assets = latest_statement_value(
        bs,
        [
            "Current Assets",
            "Total Current Assets",
        ],
    )

    current_liabilities = latest_statement_value(
        bs,
        [
            "Current Liabilities",
            "Total Current Liabilities",
        ],
    )

    working_capital = None
    if current_assets is not None and current_liabilities is not None:
        working_capital = current_assets - current_liabilities

    ocf = latest_statement_value(
        cf,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
    )

    capex = latest_statement_value(
        cf,
        [
            "Capital Expenditure",
            "Capital Expenditures",
        ],
    )

    # Yahoo usually reports capex as negative cash flow.
    fcf = None
    if ocf is not None and capex is not None:
        fcf = ocf + capex

    peg = safe_float(info.get("pegRatio"))
    pb = safe_float(info.get("priceToBook"))
    forward_pe = safe_float(info.get("forwardPE"))

    dividend_yield = safe_float(info.get("dividendYield"))
    payout_ratio = safe_float(info.get("payoutRatio"))

    gross_margin = safe_float(info.get("grossMargins"))
    operating_margin = safe_float(info.get("operatingMargins"))
    profit_margin = safe_float(info.get("profitMargins"))

    debt_equity = safe_float(info.get("debtToEquity"))
    if debt_equity is not None:
        debt_equity = debt_equity / 100.0

    current_ratio = safe_float(info.get("currentRatio"))

    # Buyback yield approximation:
    # Yahoo does not reliably expose net buybacks from info.
    repurchase = latest_statement_value(
        cf,
        [
            "Repurchase Of Capital Stock",
            "Repurchase Of Stock",
            "Common Stock Repurchased",
        ],
    )

    buyback_yield = None
    if repurchase is not None and market_cap not in [None, 0]:
        # Repurchase is often negative cash flow. Make yield positive if buybacks occurred.
        buyback_yield = -repurchase / market_cap

    # Fallback EV if Yahoo doesn't provide it
    if enterprise_value is None and market_cap is not None:
        enterprise_value = market_cap + (debt or 0) - (cash or 0)

    # Fallback P/B if missing
    if pb is None and market_cap is not None and equity not in [None, 0]:
        pb = market_cap / equity

    # Fallback D/E if missing
    if debt_equity is None and debt is not None and equity not in [None, 0]:
        debt_equity = debt / equity

    # Fallback current ratio
    if current_ratio is None:
        current_ratio = safe_div(current_assets, current_liabilities)

    return {
        "ticker": ticker,
        "name": name,
        "marketCap": market_cap,
        "enterpriseValue": enterprise_value,
        "revenue": revenue,
        "eps": eps,
        "ebitda": ebitda,
        "cash": cash,
        "debt": debt,
        "sharesOutstanding": shares_outstanding,
        "equity": equity,
        "workingCapital": working_capital,
        "ocf": ocf,
        "fcf": fcf,
        "capex": capex,
        "peg": peg,
        "pb": pb,
        "forwardPE": forward_pe,
        "buybackYield": buyback_yield,
        "dividendYield": dividend_yield,
        "payoutRatio": payout_ratio,
        "grossMargin": gross_margin,
        "operatingMargin": operating_margin,
        "profitMargin": profit_margin,
        "debtEquity": debt_equity,
        "currentRatio": current_ratio,
        "source": "yfinance",
        "yahooTicker": yahoo_ticker,
    }


def main():
    universe_path = "data/universe.csv"

    universe = pd.read_csv(universe_path)

    if "Ticker" not in universe.columns:
        raise RuntimeError(
            f"Expected column 'Ticker' in {universe_path}. Found: {list(universe.columns)}"
        )

    raw_tickers = (
        universe["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    seen = set()
    tickers = []

    for t in raw_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    rows = []
    failed = []

    for i, batch in enumerate(chunk_list(tickers, BATCH_SIZE), start=1):
        print(f"\nBatch {i}: processing {len(batch)} tickers...")

        for ticker in batch:
            try:
                print(f"Fetching {ticker}...")
                row = build_row(ticker)
                rows.append(row)
            except Exception as e:
                print(f"{ticker}: hard failed: {e}")
                failed.append({"ticker": ticker, "error": str(e)})

        time.sleep(PAUSE_SECS)

    if not rows:
        raise RuntimeError("No screener rows built.")

    df = pd.DataFrame(rows)

    column_order = [
        "ticker",
        "name",
        "marketCap",
        "enterpriseValue",
        "revenue",
        "eps",
        "ebitda",
        "cash",
        "debt",
        "sharesOutstanding",
        "equity",
        "workingCapital",
        "ocf",
        "fcf",
        "capex",
        "peg",
        "pb",
        "forwardPE",
        "buybackYield",
        "dividendYield",
        "payoutRatio",
        "grossMargin",
        "operatingMargin",
        "profitMargin",
        "debtEquity",
        "currentRatio",
        "source",
        "yahooTicker",
    ]

    df = df[[c for c in column_order if c in df.columns]]

    os.makedirs("data", exist_ok=True)
    os.makedirs("public/data", exist_ok=True)

    csv_path = "data/stock_screener.csv"
    public_csv_path = "public/data/stock_screener.csv"
    json_path = "public/data/stock_screener.json"

    df.to_csv(csv_path, index=False)
    df.to_csv(public_csv_path, index=False)

    payload = {
        "as_of": dt.date.today().isoformat(),
        "source": "yfinance",
        "ticker_count": int(len(df)),
        "failed_count": int(len(failed)),
        "failed": failed[:100],
        "units": {
            "marketCap": "usd",
            "enterpriseValue": "usd",
            "revenue": "usd",
            "eps": "usd_per_share",
            "ebitda": "usd",
            "cash": "usd",
            "debt": "usd",
            "sharesOutstanding": "shares",
            "equity": "usd",
            "workingCapital": "usd",
            "ocf": "usd",
            "fcf": "usd",
            "capex": "usd",
            "peg": "ratio",
            "pb": "ratio",
            "forwardPE": "ratio",
            "buybackYield": "decimal",
            "dividendYield": "decimal",
            "payoutRatio": "decimal",
            "grossMargin": "decimal",
            "operatingMargin": "decimal",
            "profitMargin": "decimal",
            "debtEquity": "decimal",
            "currentRatio": "ratio",
        },
        "data": df.replace({np.nan: None}).to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {csv_path} ({os.path.getsize(csv_path)} bytes)")
    print(f"Saved {public_csv_path} ({os.path.getsize(public_csv_path)} bytes)")
    print(f"Saved {json_path} ({os.path.getsize(json_path)} bytes)")
    print(f"Failed tickers: {len(failed)}")

    if failed:
        print("First 25 failed:", failed[:25])


if __name__ == "__main__":
    main()
