import os
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


BATCH_SIZE = 75
PAUSE_SECS = 1.0

FX_CACHE = {}


def normalize_for_yahoo(t: str) -> str:
    t = str(t).strip().upper()
    if not t:
        return ""
    return t.replace(".", "-")


def normalize_currency(c):
    if c is None:
        return None

    c = str(c).strip()

    if not c:
        return None

    if c in ["$", "US$", "USD"]:
        return "USD"

    # Yahoo often uses GBp / GBX for London prices quoted in pence.
    # For monetary statement fields, we convert as GBP.
    if c.upper() in ["GBP", "GBX", "GBP=X"]:
        return "GBP"

    if c in ["GBp", "GBp=X"]:
        return "GBp"

    return c.upper()


def money_currency(c):
    c = normalize_currency(c)
    if c in ["GBp", "GBX"]:
        return "GBP"
    return c


def safe_float(x):
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    try:
        x = float(x)
        if not np.isfinite(x):
            return None
        return x
    except Exception:
        return None


def safe_div(a, b):
    a = safe_float(a)
    b = safe_float(b)

    if a is None or b is None or b == 0:
        return None

    return safe_float(a / b)


def normalize_decimal_percent(x, divide_if_over=1.0):
    """
    Converts Yahoo percent-style values into decimals.

    Examples:
      2.5  -> 0.025
      0.025 -> 0.025
      45 -> 0.45
      0.45 -> 0.45
    """
    x = safe_float(x)

    if x is None:
        return None

    if abs(x) > divide_if_over:
        return safe_float(x / 100.0)

    return x


def normalize_ratio_percent(x):
    """
    Useful for payout ratio, margins, debt/equity, etc.
    Some Yahoo fields are decimals already.
    Some are returned as 45 instead of 0.45.
    """
    x = safe_float(x)

    if x is None:
        return None

    if abs(x) > 10:
        return safe_float(x / 100.0)

    return x


def get_first(info: dict, keys):
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return None


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def clean_json_value(x):
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x):
            return None
        return float(x)

    return x


def clean_record(record):
    return {k: clean_json_value(v) for k, v in record.items()}


def latest_close(ticker):
    try:
        hist = yf.Ticker(ticker).history(period="7d", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna()
        if close.empty:
            return None

        return safe_float(close.iloc[-1])
    except Exception:
        return None


def get_fx_to_usd(currency):
    """
    Returns USD per 1 unit of currency.

    Example:
      EUR -> 1.07 means 1 EUR = 1.07 USD
      JPY -> 0.0066 means 1 JPY = 0.0066 USD
    """
    currency = money_currency(currency)

    if currency is None:
        return None

    if currency == "USD":
        return 1.0

    if currency in FX_CACHE:
        return FX_CACHE[currency]

    rate = None

    # Direct pair: EURUSD=X means USD per EUR.
    direct_pair = f"{currency}USD=X"
    direct = latest_close(direct_pair)

    if direct is not None and direct > 0:
        rate = direct

    # Inverse pair: USDJPY=X means JPY per USD, so invert.
    if rate is None:
        inverse_pair = f"USD{currency}=X"
        inverse = latest_close(inverse_pair)

        if inverse is not None and inverse > 0:
            rate = 1.0 / inverse

    # Yahoo aliases like JPY=X often mean USDJPY, so invert.
    if rate is None:
        alias_pair = f"{currency}=X"
        alias = latest_close(alias_pair)

        if alias is not None and alias > 0:
            rate = 1.0 / alias

    rate = safe_float(rate)
    FX_CACHE[currency] = rate

    return rate


def money_to_usd(value, currency):
    value = safe_float(value)

    if value is None:
        return None

    currency = money_currency(currency)
    rate = get_fx_to_usd(currency)

    if rate is None:
        return None

    return safe_float(value * rate)


def price_to_usd(value, quote_currency):
    value = safe_float(value)

    if value is None:
        return None

    quote_currency = normalize_currency(quote_currency)

    # London prices are often quoted in pence.
    if quote_currency == "GBp":
        gbp_rate = get_fx_to_usd("GBP")
        if gbp_rate is None:
            return None
        return safe_float((value / 100.0) * gbp_rate)

    rate = get_fx_to_usd(quote_currency)

    if rate is None:
        return None

    return safe_float(value * rate)


def latest_statement_value(df, possible_rows):
    if df is None or df.empty:
        return None

    for row_name in possible_rows:
        if row_name in df.index:
            series = df.loc[row_name].dropna()
            if len(series) > 0:
                return safe_float(series.iloc[0])

    return None


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

    name = get_first(info, ["longName", "shortName", "displayName"])

    quote_currency = normalize_currency(
        get_first(info, ["currency", "quoteCurrency"])
    ) or "USD"

    financial_currency = normalize_currency(
        get_first(info, ["financialCurrency", "currency"])
    ) or quote_currency

    quote_fx_to_usd = get_fx_to_usd(quote_currency)
    financial_fx_to_usd = get_fx_to_usd(financial_currency)

    # -------------------------
    # Price / market values
    # -------------------------
    price_raw = safe_float(
        get_first(
            info,
            [
                "currentPrice",
                "regularMarketPrice",
                "previousClose",
            ],
        )
    )

    price = price_to_usd(price_raw, quote_currency)

    market_cap_raw = safe_float(info.get("marketCap"))
    enterprise_value_raw = safe_float(info.get("enterpriseValue"))

    market_cap = money_to_usd(market_cap_raw, quote_currency)
    enterprise_value = money_to_usd(enterprise_value_raw, quote_currency)

    shares_outstanding = safe_float(
        get_first(info, ["sharesOutstanding", "impliedSharesOutstanding"])
    )

    # -------------------------
    # Income statement
    # -------------------------
    revenue_raw = latest_statement_value(
        fin,
        [
            "Total Revenue",
            "Operating Revenue",
            "Revenue",
        ],
    )

    if revenue_raw is None:
        revenue_raw = safe_float(get_first(info, ["totalRevenue", "revenue"]))

    revenue = money_to_usd(revenue_raw, financial_currency)

    ebitda_raw = latest_statement_value(
        fin,
        [
            "EBITDA",
            "Normalized EBITDA",
        ],
    )

    if ebitda_raw is None:
        ebitda_raw = safe_float(info.get("ebitda"))

    ebitda = money_to_usd(ebitda_raw, financial_currency)

    eps_raw = safe_float(
        get_first(info, ["trailingEps", "epsTrailingTwelveMonths"])
    )

    eps = price_to_usd(eps_raw, quote_currency)

    # -------------------------
    # Balance sheet
    # -------------------------
    cash_raw = latest_statement_value(
        bs,
        [
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash",
        ],
    )

    if cash_raw is None:
        cash_raw = safe_float(get_first(info, ["totalCash", "cash"]))

    cash = money_to_usd(cash_raw, financial_currency)

    debt_raw = latest_statement_value(
        bs,
        [
            "Total Debt",
            "Long Term Debt And Capital Lease Obligation",
            "Long Term Debt",
        ],
    )

    if debt_raw is None:
        debt_raw = safe_float(get_first(info, ["totalDebt", "debt"]))

    debt = money_to_usd(debt_raw, financial_currency)

    equity_statement_raw = latest_statement_value(
        bs,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
            "Total Stockholder Equity",
        ],
    )

    equity_raw = equity_statement_raw
    equity_raw_currency = financial_currency

    if equity_raw is None:
        book_value_per_share_raw = safe_float(info.get("bookValue"))
        if book_value_per_share_raw is not None and shares_outstanding is not None:
            equity_raw = safe_float(book_value_per_share_raw * shares_outstanding)
            equity_raw_currency = quote_currency

    equity = money_to_usd(equity_raw, equity_raw_currency)

    tangible_book_value_raw = latest_statement_value(
        bs,
        [
            "Tangible Book Value",
        ],
    )

    if tangible_book_value_raw is None and equity_statement_raw is not None:
        goodwill_and_intangibles_raw = latest_statement_value(
            bs,
            [
                "Goodwill And Other Intangible Assets",
            ],
        )

        if goodwill_and_intangibles_raw is None:
            goodwill_raw = latest_statement_value(bs, ["Goodwill"])
            intangibles_raw = latest_statement_value(
                bs,
                [
                    "Other Intangible Assets",
                    "Other Intangibles",
                    "Intangible Assets",
                ],
            )

            goodwill_and_intangibles_raw = safe_float(
                (goodwill_raw or 0) + (intangibles_raw or 0)
            )

        tangible_book_value_raw = safe_float(
            equity_statement_raw - (goodwill_and_intangibles_raw or 0)
        )

    tangible_book_value = money_to_usd(tangible_book_value_raw, financial_currency)

    current_assets_raw = latest_statement_value(
        bs,
        [
            "Current Assets",
            "Total Current Assets",
        ],
    )

    current_liabilities_raw = latest_statement_value(
        bs,
        [
            "Current Liabilities",
            "Total Current Liabilities",
        ],
    )

    working_capital_raw = None
    if current_assets_raw is not None and current_liabilities_raw is not None:
        working_capital_raw = safe_float(current_assets_raw - current_liabilities_raw)

    working_capital = money_to_usd(working_capital_raw, financial_currency)

    current_ratio = normalize_ratio_percent(info.get("currentRatio"))

    if current_ratio is None:
        current_ratio = safe_div(current_assets_raw, current_liabilities_raw)

    # -------------------------
    # Cash flow
    # -------------------------
    ocf_raw = latest_statement_value(
        cf,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
    )

    capex_raw = latest_statement_value(
        cf,
        [
            "Capital Expenditure",
            "Capital Expenditures",
        ],
    )

    ocf = money_to_usd(ocf_raw, financial_currency)
    capex = money_to_usd(capex_raw, financial_currency)

    fcf = None
    if ocf is not None and capex is not None:
        # Yahoo usually reports capex as negative cash flow.
        fcf = safe_float(ocf + capex)

    repurchase_raw = latest_statement_value(
        cf,
        [
            "Repurchase Of Capital Stock",
            "Repurchase Of Stock",
            "Common Stock Repurchased",
        ],
    )

    repurchase = money_to_usd(repurchase_raw, financial_currency)

    buyback_yield = None
    if repurchase is not None and market_cap not in [None, 0]:
        # Repurchases are usually negative cash flow. Make yield positive.
        buyback_yield = safe_float(-repurchase / market_cap)

    # -------------------------
    # Valuation / quality ratios
    # -------------------------
    peg = safe_float(info.get("pegRatio"))
    forward_pe = safe_float(info.get("forwardPE"))

    pb = safe_div(market_cap, equity)
    if pb is None:
        pb = safe_float(info.get("priceToBook"))

    ptbv = safe_div(market_cap, tangible_book_value)

    debt_equity = safe_div(debt, equity)

    if debt_equity is None:
        debt_equity = normalize_ratio_percent(info.get("debtToEquity"))

    dividend_yield = normalize_decimal_percent(
        info.get("dividendYield"),
        divide_if_over=1.0,
    )

    payout_ratio = normalize_ratio_percent(info.get("payoutRatio"))

    gross_margin = normalize_ratio_percent(info.get("grossMargins"))
    operating_margin = normalize_ratio_percent(info.get("operatingMargins"))
    profit_margin = normalize_ratio_percent(info.get("profitMargins"))

    institutional_ownership = normalize_decimal_percent(
        info.get("heldPercentInstitutions"),
        divide_if_over=1.0,
    )

    growth_rate = normalize_ratio_percent(
        get_first(
            info,
            [
                "revenueGrowth",
                "earningsGrowth",
                "earningsQuarterlyGrowth",
            ],
        )
    )

    growth_rate_source = None
    if info.get("revenueGrowth") is not None:
        growth_rate_source = "revenueGrowth"
    elif info.get("earningsGrowth") is not None:
        growth_rate_source = "earningsGrowth"
    elif info.get("earningsQuarterlyGrowth") is not None:
        growth_rate_source = "earningsQuarterlyGrowth"

    if enterprise_value is None and market_cap is not None:
        enterprise_value = safe_float(market_cap + (debt or 0) - (cash or 0))

    return {
        "ticker": ticker,
        "name": name,

        # New fields
        "price": price,
        "institutionalOwnership": institutional_ownership,
        "growthRate": growth_rate,
        "growthRateSource": growth_rate_source,
        "tangibleBookValue": tangible_book_value,
        "ptbv": ptbv,

        # Existing screener fields
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

        # Currency/debug fields
        "outputCurrency": "USD",
        "quoteCurrency": quote_currency,
        "financialCurrency": financial_currency,
        "quoteFxToUsd": quote_fx_to_usd,
        "financialFxToUsd": financial_fx_to_usd,
        "priceRaw": price_raw,
        "marketCapRaw": market_cap_raw,
        "revenueRaw": revenue_raw,

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
                rows.append(build_row(ticker))
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
        "price",
        "marketCap",
        "enterpriseValue",
        "revenue",
        "eps",
        "ebitda",
        "cash",
        "debt",
        "sharesOutstanding",
        "equity",
        "tangibleBookValue",
        "workingCapital",
        "ocf",
        "fcf",
        "capex",
        "peg",
        "pb",
        "ptbv",
        "forwardPE",
        "institutionalOwnership",
        "growthRate",
        "growthRateSource",
        "buybackYield",
        "dividendYield",
        "payoutRatio",
        "grossMargin",
        "operatingMargin",
        "profitMargin",
        "debtEquity",
        "currentRatio",
        "outputCurrency",
        "quoteCurrency",
        "financialCurrency",
        "quoteFxToUsd",
        "financialFxToUsd",
        "priceRaw",
        "marketCapRaw",
        "revenueRaw",
        "source",
        "yahooTicker",
    ]

    df = df[[c for c in column_order if c in df.columns]]
    df = df.replace([np.inf, -np.inf], np.nan)

    os.makedirs("data", exist_ok=True)
    os.makedirs("public/data", exist_ok=True)

    csv_path = "data/stock_screener.csv"
    public_csv_path = "public/data/stock_screener.csv"
    json_path = "public/data/stock_screener.json"

    df.to_csv(csv_path, index=False)
    df.to_csv(public_csv_path, index=False)

    records = df.to_dict(orient="records")
    records = [clean_record(r) for r in records]

    payload = {
        "as_of": dt.date.today().isoformat(),
        "source": "yfinance",
        "output_currency": "USD",
        "ticker_count": int(len(df)),
        "failed_count": int(len(failed)),
        "failed": failed[:100],
        "units": {
            "price": "usd",
            "marketCap": "usd",
            "enterpriseValue": "usd",
            "revenue": "usd",
            "eps": "usd_per_share",
            "ebitda": "usd",
            "cash": "usd",
            "debt": "usd",
            "sharesOutstanding": "shares",
            "equity": "usd",
            "tangibleBookValue": "usd",
            "workingCapital": "usd",
            "ocf": "usd",
            "fcf": "usd",
            "capex": "usd",
            "peg": "ratio",
            "pb": "ratio",
            "ptbv": "ratio",
            "forwardPE": "ratio",
            "institutionalOwnership": "decimal",
            "growthRate": "decimal",
            "buybackYield": "decimal",
            "dividendYield": "decimal",
            "payoutRatio": "decimal",
            "grossMargin": "decimal",
            "operatingMargin": "decimal",
            "profitMargin": "decimal",
            "debtEquity": "ratio",
            "currentRatio": "ratio",
            "quoteFxToUsd": "usd_per_quote_currency",
            "financialFxToUsd": "usd_per_financial_currency",
        },
        "data": records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"\nSaved {csv_path} ({os.path.getsize(csv_path)} bytes)")
    print(f"Saved {public_csv_path} ({os.path.getsize(public_csv_path)} bytes)")
    print(f"Saved {json_path} ({os.path.getsize(json_path)} bytes)")
    print(f"Failed tickers: {len(failed)}")

    if failed:
        print("First 25 failed:", failed[:25])


if __name__ == "__main__":
    main()
