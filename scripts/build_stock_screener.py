import os
import json
import time
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf


BATCH_SIZE = int(os.getenv("SCREENER_BATCH_SIZE", "25"))
TICKER_PAUSE_SECS = float(os.getenv("SCREENER_TICKER_PAUSE_SECS", "0.5"))
BATCH_PAUSE_SECS = float(os.getenv("SCREENER_BATCH_PAUSE_SECS", "10"))
MAX_RETRIES = int(os.getenv("SCREENER_MAX_RETRIES", "3"))
RETRY_BASE_SECS = float(os.getenv("SCREENER_RETRY_BASE_SECS", "3"))
MAX_FALLBACK_FRACTION = float(os.getenv("SCREENER_MAX_FALLBACK_FRACTION", "0.05"))
MAX_STALE_DAYS = int(os.getenv("SCREENER_MAX_STALE_DAYS", "7"))

BENCHMARK_TICKER = "SPY"
RSI_PERIOD = 14
BETA_LOOKBACK_PERIOD = "1y"
THREE_YEAR_LOOKBACK_PERIOD = "3y"

FX_CACHE = {}
HISTORY_CACHE = {}

CORE_FIELDS = [
    "price",
    "rsi14",
    "beta",
    "fiftyTwoWeekLow",
    "threeYearLow",
]

QUALITY_FIELDS = [
    "name",
    "price",
    "rsi14",
    "beta",
    "fiftyTwoWeekLow",
    "threeYearLow",
    "threeYearHigh",
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
    "ocf",
    "fcf",
    "pe",
    "pb",
    "ptbv",
    "ps",
    "forwardPE",
    "forwardRevenue",
    "dividendYield",
]


if os.getenv("YF_DEBUG", "0") == "1":
    yf.enable_debug_mode()


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
    x = safe_float(x)

    if x is None:
        return None

    if abs(x) > divide_if_over:
        return safe_float(x / 100.0)

    return x


def normalize_ratio_percent(x):
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


def row_quality_score(row):
    if not row:
        return 0
    return sum(row.get(field) is not None for field in QUALITY_FIELDS)


def row_needs_retry(row, previous=None):
    if not row or not row.get("name"):
        return True
    if all(row.get(field) is None for field in CORE_FIELDS):
        return True

    # Catch a partial Yahoo response even when price history succeeded. Missing
    # fields that were also missing previously remain legitimate and do not
    # trigger retries.
    previous_score = row_quality_score(previous)
    current_score = row_quality_score(row)
    return previous_score >= 10 and current_score < previous_score * 0.60


def clear_history_cache_for_ticker(ticker):
    yahoo_ticker = normalize_for_yahoo(ticker)
    keys = [key for key in HISTORY_CACHE if key[0] == yahoo_ticker]
    for key in keys:
        HISTORY_CACHE.pop(key, None)


def load_previous_screener(path):
    if not os.path.exists(path):
        return {}, None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("data", [])
        previous = {
            str(row.get("ticker", "")).strip().upper(): row
            for row in rows
            if row.get("ticker")
        }
        return previous, payload.get("as_of")
    except Exception as e:
        print(f"Could not load previous screener for fallback: {e}", flush=True)
        return {}, None


def previous_row_is_recent(row, previous_as_of):
    date_text = row.get("dataAsOf") or previous_as_of
    if not date_text:
        return False

    try:
        data_date = dt.date.fromisoformat(str(date_text)[:10])
    except ValueError:
        return False

    return (dt.date.today() - data_date).days <= MAX_STALE_DAYS


def merge_with_previous(current, previous, previous_as_of):
    merged = dict(previous)
    for key, value in current.items():
        if value is not None:
            merged[key] = value

    merged["ticker"] = current.get("ticker") or previous.get("ticker")
    merged["dataStatus"] = "stale_fallback"
    merged["dataAsOf"] = previous.get("dataAsOf") or previous_as_of
    return merged


def get_close_history(yahoo_ticker, period="1y"):
    key = (yahoo_ticker, period)

    if key in HISTORY_CACHE:
        return HISTORY_CACHE[key]

    try:
        hist = yf.Ticker(yahoo_ticker).history(
            period=period,
            interval="1d",
            auto_adjust=True,
        )

        if hist is None or hist.empty or "Close" not in hist.columns:
            print(
                f"{yahoo_ticker}: empty history response for period={period}",
                flush=True,
            )
            return pd.Series(dtype=float)

        close = hist["Close"].dropna()

        if close.empty:
            print(
                f"{yahoo_ticker}: history contained no closing prices "
                f"for period={period}",
                flush=True,
            )
            return pd.Series(dtype=float)

        HISTORY_CACHE[key] = close.astype(float)
        return HISTORY_CACHE[key]

    except Exception as e:
        print(
            f"{yahoo_ticker}: history fetch failed for period={period}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        return pd.Series(dtype=float)


def latest_close(ticker):
    close = get_close_history(ticker, period="7d")

    if close is None or close.empty:
        return None

    return safe_float(close.iloc[-1])


def compute_rsi(close, period=14):
    if close is None or close.empty or len(close) <= period:
        return None

    try:
        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        last = rsi.dropna()

        if last.empty:
            return None

        return safe_float(last.iloc[-1])

    except Exception:
        return None


def compute_beta(asset_close, benchmark_close):
    if asset_close is None or benchmark_close is None:
        return None

    if asset_close.empty or benchmark_close.empty:
        return None

    try:
        returns = pd.concat(
            [
                asset_close.pct_change().rename("asset"),
                benchmark_close.pct_change().rename("benchmark"),
            ],
            axis=1,
        ).dropna()

        if len(returns) < 60:
            return None

        benchmark_var = returns["benchmark"].var()

        if benchmark_var is None or benchmark_var == 0:
            return None

        beta = returns["asset"].cov(returns["benchmark"]) / benchmark_var
        return safe_float(beta)

    except Exception:
        return None


def get_fx_to_usd(currency):
    currency = money_currency(currency)

    if currency is None:
        return None

    if currency == "USD":
        return 1.0

    if currency in FX_CACHE:
        return FX_CACHE[currency]

    rate = None

    direct_pair = f"{currency}USD=X"
    direct = latest_close(direct_pair)

    if direct is not None and direct > 0:
        rate = direct

    if rate is None:
        inverse_pair = f"USD{currency}=X"
        inverse = latest_close(inverse_pair)

        if inverse is not None and inverse > 0:
            rate = 1.0 / inverse

    if rate is None:
        alias_pair = f"{currency}=X"
        alias = latest_close(alias_pair)

        if alias is not None and alias > 0:
            rate = 1.0 / alias

    rate = safe_float(rate)
    # Do not cache a failed FX lookup for the entire run. A transient Yahoo
    # error would otherwise blank every later ticker in that currency.
    if rate is not None:
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


def latest_two_statement_values(df, possible_rows):
    if df is None or df.empty:
        return None, None

    for row_name in possible_rows:
        if row_name in df.index:
            series = df.loc[row_name].dropna()
            values = [safe_float(value) for value in series.tolist()]
            values = [value for value in values if value is not None]

            if len(values) >= 2:
                return values[0], values[1]

    return None, None


def analyst_estimate_value(df, periods, possible_columns):
    if df is None or df.empty:
        return None

    for period in periods:
        if period not in df.index:
            continue

        row = df.loc[period]

        for column in possible_columns:
            if column in row.index:
                value = safe_float(row[column])
                if value is not None:
                    return value

    return None


def build_row(ticker: str) -> dict:
    yahoo_ticker = normalize_for_yahoo(ticker)
    yf_ticker = yf.Ticker(yahoo_ticker)

    try:
        info = yf_ticker.get_info()
    except Exception as e:
        print(f"{ticker}: failed info fetch: {e}", flush=True)
        info = {}

    try:
        bs = yf_ticker.balance_sheet
    except Exception as e:
        print(f"{ticker}: balance-sheet fetch failed: {e}", flush=True)
        bs = pd.DataFrame()

    try:
        fin = yf_ticker.financials
    except Exception as e:
        print(f"{ticker}: financials fetch failed: {e}", flush=True)
        fin = pd.DataFrame()

    try:
        cf = yf_ticker.cashflow
    except Exception as e:
        print(f"{ticker}: cash-flow fetch failed: {e}", flush=True)
        cf = pd.DataFrame()

    try:
        revenue_estimate = yf_ticker.revenue_estimate
    except Exception as e:
        print(f"{ticker}: revenue-estimate fetch failed: {e}", flush=True)
        revenue_estimate = pd.DataFrame()

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
    # Technical / price history metrics
    # -------------------------
    close_history = get_close_history(yahoo_ticker, period=BETA_LOOKBACK_PERIOD)
    benchmark_close_history = get_close_history(BENCHMARK_TICKER, period=BETA_LOOKBACK_PERIOD)
    three_year_close_history = get_close_history(
        yahoo_ticker,
        period=THREE_YEAR_LOOKBACK_PERIOD,
    )

    rsi14 = compute_rsi(close_history, period=RSI_PERIOD)
    beta = compute_beta(close_history, benchmark_close_history)

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

    if price_raw is None and close_history is not None and not close_history.empty:
        price_raw = safe_float(close_history.iloc[-1])

    price = price_to_usd(price_raw, quote_currency)

    fifty_two_week_low_raw = safe_float(info.get("fiftyTwoWeekLow"))

    if fifty_two_week_low_raw is None and close_history is not None and not close_history.empty:
        fifty_two_week_low_raw = safe_float(close_history.min())

    fifty_two_week_low = price_to_usd(fifty_two_week_low_raw, quote_currency)

    pct_above_52_week_low = None
    if price_raw is not None and fifty_two_week_low_raw not in [None, 0]:
        pct_above_52_week_low = safe_float(
            (price_raw - fifty_two_week_low_raw) / fifty_two_week_low_raw
        )
    elif price is not None and fifty_two_week_low not in [None, 0]:
        pct_above_52_week_low = safe_float(
            (price - fifty_two_week_low) / fifty_two_week_low
        )

    three_year_low_raw = None
    three_year_high_raw = None

    if three_year_close_history is not None and not three_year_close_history.empty:
        three_year_low_raw = safe_float(three_year_close_history.min())
        three_year_high_raw = safe_float(three_year_close_history.max())

    three_year_low = price_to_usd(three_year_low_raw, quote_currency)
    three_year_high = price_to_usd(three_year_high_raw, quote_currency)

    pct_above_3_year_low = None
    if price_raw is not None and three_year_low_raw not in [None, 0]:
        pct_above_3_year_low = safe_float(
            (price_raw - three_year_low_raw) / three_year_low_raw
        )

    pct_below_3_year_high = None
    if price_raw is not None and three_year_high_raw not in [None, 0]:
        pct_below_3_year_high = safe_float(
            (three_year_high_raw - price_raw) / three_year_high_raw
        )

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

    pe = None
    if (
        price_raw is not None
        and price_raw > 0
        and eps_raw is not None
        and eps_raw > 0
    ):
        pe = safe_div(price_raw, eps_raw)

    if pe is None:
        pe = safe_float(get_first(info, ["trailingPE", "trailingPe"]))

    pretax_income_raw = latest_statement_value(
        fin,
        [
            "Pretax Income",
            "Income Before Tax",
            "Earnings Before Tax",
        ],
    )

    tax_provision_raw = latest_statement_value(
        fin,
        [
            "Tax Provision",
            "Income Tax Expense",
            "Provision For Income Taxes",
        ],
    )

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
        buyback_yield = safe_float(-repurchase / market_cap)

    # -------------------------
    # Valuation / quality ratios
    # -------------------------
    peg = safe_float(info.get("pegRatio"))
    forward_pe = safe_float(info.get("forwardPE"))

    ps = safe_div(market_cap, revenue)
    if ps is None:
        ps = safe_float(
            get_first(
                info,
                [
                    "priceToSalesTrailing12Months",
                    "priceToSales",
                ],
            )
        )

    forward_revenue_raw = analyst_estimate_value(
        revenue_estimate,
        ["+1y", "0y"],
        ["avg", "average", "avgEstimate"],
    )
    forward_revenue = money_to_usd(
        forward_revenue_raw,
        financial_currency,
    )

    forward_ps = safe_div(market_cap, forward_revenue)
    if forward_ps is None:
        forward_ps = safe_float(
            get_first(
                info,
                [
                    "forwardPriceToSales",
                    "forwardPS",
                    "priceToSalesForward",
                ],
            )
        )

    effective_tax_rate = safe_float(info.get("effectiveTaxRate"))
    effective_tax_rate = normalize_decimal_percent(
        effective_tax_rate,
        divide_if_over=1.0,
    )

    if effective_tax_rate is None:
        calculated_tax_rate = safe_div(tax_provision_raw, pretax_income_raw)
        if calculated_tax_rate is not None and calculated_tax_rate >= 0:
            effective_tax_rate = calculated_tax_rate

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

    latest_share_count, prior_share_count = latest_two_statement_values(
        fin,
        [
            "Diluted Average Shares",
            "Basic Average Shares",
        ],
    )

    if latest_share_count is None or prior_share_count is None:
        latest_share_count, prior_share_count = latest_two_statement_values(
            bs,
            [
                "Ordinary Shares Number",
                "Share Issued",
            ],
        )

    net_share_change = None
    if prior_share_count not in [None, 0] and latest_share_count is not None:
        net_share_change = safe_float(
            (latest_share_count - prior_share_count) / prior_share_count
        )

    if enterprise_value is None and market_cap is not None:
        enterprise_value = safe_float(market_cap + (debt or 0) - (cash or 0))

    return {
        "ticker": ticker,
        "name": name,

        # Price / technical fields
        "price": price,
        "rsi14": rsi14,
        "beta": beta,
        "betaBenchmark": BENCHMARK_TICKER,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "pctAbove52WeekLow": pct_above_52_week_low,
        "threeYearLow": three_year_low,
        "threeYearHigh": three_year_high,
        "pctAbove3YearLow": pct_above_3_year_low,
        "pctBelow3YearHigh": pct_below_3_year_high,

        # Ownership / growth / book value fields
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
        "netShareChange": net_share_change,
        "equity": equity,
        "workingCapital": working_capital,
        "ocf": ocf,
        "fcf": fcf,
        "capex": capex,
        "peg": peg,
        "pe": pe,
        "pb": pb,
        "ps": ps,
        "forwardPS": forward_ps,
        "forwardPE": forward_pe,
        "forwardRevenue": forward_revenue,
        "effectiveTaxRate": effective_tax_rate,
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
        "fiftyTwoWeekLowRaw": fifty_two_week_low_raw,
        "threeYearLowRaw": three_year_low_raw,
        "threeYearHighRaw": three_year_high_raw,
        "marketCapRaw": market_cap_raw,
        "revenueRaw": revenue_raw,

        "source": "yfinance",
        "yahooTicker": yahoo_ticker,
    }


def build_row_with_retries(ticker, previous=None):
    last_row = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            last_row = build_row(ticker)
            last_error = None
        except Exception as e:
            last_error = e
            print(
                f"{ticker}: hard failure on attempt {attempt}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

        if last_row is not None and not row_needs_retry(last_row, previous):
            return last_row, attempt - 1, None

        if attempt < MAX_RETRIES:
            delay = RETRY_BASE_SECS * (2 ** (attempt - 1))
            print(
                f"{ticker}: incomplete Yahoo response; retrying in "
                f"{delay:.1f}s ({attempt}/{MAX_RETRIES})",
                flush=True,
            )
            clear_history_cache_for_ticker(ticker)
            time.sleep(delay)

    if last_row is None:
        last_row = {"ticker": ticker, "yahooTicker": normalize_for_yahoo(ticker)}

    reason = (
        f"{type(last_error).__name__}: {last_error}"
        if last_error is not None
        else "Yahoo returned an incomplete row after all retries"
    )
    return last_row, MAX_RETRIES - 1, reason


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

    json_path = "public/data/stock_screener.json"
    previous_rows, previous_as_of = load_previous_screener(json_path)
    previous_unresolved = sum(
        row_needs_retry(row) for row in previous_rows.values()
    )

    rows = []
    failed = []
    stale_fallback = []
    unresolved = []
    retry_count = 0
    today = dt.date.today().isoformat()

    for i, batch in enumerate(chunk_list(tickers, BATCH_SIZE), start=1):
        print(f"\nBatch {i}: processing {len(batch)} tickers...")

        for ticker in batch:
            try:
                print(f"Fetching {ticker}...", flush=True)
                previous = previous_rows.get(ticker)
                row, retries_used, error = build_row_with_retries(
                    ticker,
                    previous,
                )
                retry_count += retries_used

                if row_needs_retry(row, previous):
                    if (
                        previous
                        and not row_needs_retry(previous)
                        and previous_row_is_recent(previous, previous_as_of)
                    ):
                        row = merge_with_previous(row, previous, previous_as_of)
                        stale_fallback.append(ticker)
                        print(
                            f"{ticker}: using previous successful data "
                            f"from {row.get('dataAsOf')}",
                            flush=True,
                        )
                    else:
                        row["dataStatus"] = "unresolved"
                        row["dataAsOf"] = None
                        unresolved.append(ticker)
                        failed.append(
                            {
                                "ticker": ticker,
                                "error": error or "Incomplete Yahoo response",
                            }
                        )
                else:
                    row["dataStatus"] = "fresh"
                    row["dataAsOf"] = today

                rows.append(row)
            except Exception as e:
                print(f"{ticker}: hard failed: {e}", flush=True)
                failed.append({"ticker": ticker, "error": str(e)})
                unresolved.append(ticker)
                rows.append(
                    {
                        "ticker": ticker,
                        "yahooTicker": normalize_for_yahoo(ticker),
                        "dataStatus": "unresolved",
                        "dataAsOf": None,
                    }
                )

            time.sleep(TICKER_PAUSE_SECS)

        if i * BATCH_SIZE < len(tickers):
            print(
                f"Batch {i} complete; pausing {BATCH_PAUSE_SECS:.1f}s...",
                flush=True,
            )
            time.sleep(BATCH_PAUSE_SECS)

    if not rows:
        raise RuntimeError("No screener rows built.")

    allowed_unresolved = max(3, previous_unresolved + 3)
    allowed_fallback = max(10, int(len(tickers) * MAX_FALLBACK_FRACTION))

    print("\nScreener coverage summary", flush=True)
    print(f"Rows built: {len(rows)}/{len(tickers)}", flush=True)
    print(f"Retries used: {retry_count}", flush=True)
    print(
        f"Previous-data fallbacks: {len(stale_fallback)} "
        f"(maximum allowed: {allowed_fallback})",
        flush=True,
    )
    print(
        f"Unresolved rows: {len(unresolved)} "
        f"(maximum allowed: {allowed_unresolved})",
        flush=True,
    )
    if stale_fallback:
        print(f"Fallback tickers: {stale_fallback}", flush=True)
    if unresolved:
        print(f"Unresolved tickers: {unresolved}", flush=True)

    # Abort before overwriting the last good output if Yahoo broadly failed.
    if len(stale_fallback) > allowed_fallback:
        raise RuntimeError(
            f"Likely Yahoo rate limiting: {len(stale_fallback)} tickers "
            "required previous-data fallback. Existing outputs were preserved."
        )
    if len(unresolved) > allowed_unresolved:
        raise RuntimeError(
            f"Screener coverage regressed: {len(unresolved)} unresolved tickers "
            f"versus {previous_unresolved} in the previous output. "
            "Existing outputs were preserved."
        )
    if len(rows) != len(tickers):
        raise RuntimeError(
            f"Screener produced {len(rows)} rows for {len(tickers)} tickers. "
            "Existing outputs were preserved."
        )

    df = pd.DataFrame(rows)

    column_order = [
        "ticker",
        "name",
        "price",
        "rsi14",
        "beta",
        "betaBenchmark",
        "fiftyTwoWeekLow",
        "pctAbove52WeekLow",
        "threeYearLow",
        "threeYearHigh",
        "pctAbove3YearLow",
        "pctBelow3YearHigh",
        "marketCap",
        "enterpriseValue",
        "revenue",
        "eps",
        "ebitda",
        "cash",
        "debt",
        "sharesOutstanding",
        "netShareChange",
        "equity",
        "tangibleBookValue",
        "workingCapital",
        "ocf",
        "fcf",
        "capex",
        "peg",
        "pe",
        "pb",
        "ptbv",
        "ps",
        "forwardPS",
        "forwardPE",
        "forwardRevenue",
        "effectiveTaxRate",
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
        "fiftyTwoWeekLowRaw",
        "threeYearLowRaw",
        "threeYearHighRaw",
        "marketCapRaw",
        "revenueRaw",
        "source",
        "yahooTicker",
        "dataStatus",
        "dataAsOf",
    ]

    df = df[[c for c in column_order if c in df.columns]]
    df = df.replace([np.inf, -np.inf], np.nan)

    os.makedirs("data", exist_ok=True)
    os.makedirs("public/data", exist_ok=True)

    csv_path = "data/stock_screener.csv"
    public_csv_path = "public/data/stock_screener.csv"
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
        "retry_count": int(retry_count),
        "stale_fallback_count": int(len(stale_fallback)),
        "stale_fallback_tickers": stale_fallback,
        "unresolved_count": int(len(unresolved)),
        "unresolved_tickers": unresolved,
        "units": {
            "price": "usd",
            "rsi14": "0_to_100",
            "beta": f"vs_{BENCHMARK_TICKER}",
            "fiftyTwoWeekLow": "usd",
            "pctAbove52WeekLow": "decimal",
            "threeYearLow": "usd",
            "threeYearHigh": "usd",
            "pctAbove3YearLow": "decimal",
            "pctBelow3YearHigh": "decimal",
            "marketCap": "usd",
            "enterpriseValue": "usd",
            "revenue": "usd",
            "eps": "usd_per_share",
            "ebitda": "usd",
            "cash": "usd",
            "debt": "usd",
            "sharesOutstanding": "shares",
            "netShareChange": "decimal",
            "equity": "usd",
            "tangibleBookValue": "usd",
            "workingCapital": "usd",
            "ocf": "usd",
            "fcf": "usd",
            "capex": "usd",
            "peg": "ratio",
            "pe": "ratio",
            "pb": "ratio",
            "ptbv": "ratio",
            "ps": "ratio",
            "forwardPS": "ratio",
            "forwardPE": "ratio",
            "forwardRevenue": "usd",
            "effectiveTaxRate": "decimal",
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
