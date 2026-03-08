import os
import json
import math
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests

UNIVERSE_PATH = "data/universe.csv"
OUTPUT_PATH = "public/data/fundamental_trends.json"

FMP_API_KEY = os.environ.get("FMP_API_KEY", "").strip()

# Revenue trend weights
REV_WEIGHTS = {
    "r01": 0.25,  # most recent quarter vs prior quarter
    "r12": 0.15,  # prior quarter vs two quarters ago
    "r03": 0.60,  # most recent quarter vs same quarter last year
}

# Margin trend weights
MARGIN_WEIGHTS = {
    "m01": 0.25,
    "m12": 0.15,
    "m03": 0.60,
}

REQUEST_SLEEP = 0.20
FMP_LIMIT = 8  # fetch a few extra quarters for safety


def safe_num(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def clamp(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None:
        return None
    return max(lo, min(hi, x))


def weighted_sum(parts: Dict[str, Optional[float]], weights: Dict[str, float]) -> Optional[float]:
    total = 0.0
    used = 0.0

    for key, weight in weights.items():
        value = parts.get(key)
        if value is None:
            continue
        total += value * weight
        used += weight

    if used == 0:
        return None

    # Re-weight if some inputs are missing
    return total / used


def percentile_rank_map(series: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    valid = [(k, v) for k, v in series.items() if v is not None]
    out = {k: None for k in series}

    if not valid:
        return out

    valid_sorted = sorted(valid, key=lambda kv: kv[1])
    n = len(valid_sorted)

    if n == 1:
        out[valid_sorted[0][0]] = 100.0
        return out

    for i, (ticker, _) in enumerate(valid_sorted):
        pct = 100.0 * i / (n - 1)
        out[ticker] = round(pct, 2)

    return out


def read_universe() -> List[str]:
    df = pd.read_csv(UNIVERSE_PATH)
    if "Ticker" not in df.columns:
        raise RuntimeError(f"Expected column 'Ticker' in {UNIVERSE_PATH}. Found: {list(df.columns)}")

    raw = df["Ticker"].dropna().astype(str).str.strip().str.upper().tolist()

    seen = set()
    tickers = []
    for t in raw:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    return tickers


def fetch_income_statement_quarterly(symbol: str) -> List[dict]:
    url = (
        "https://financialmodelingprep.com/stable/income-statement"
        f"?symbol={symbol}&period=quarter&limit={FMP_LIMIT}&apikey={FMP_API_KEY}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        return []
    return data

def normalize_rows(rows: List[dict]) -> List[dict]:
    """
    Normalize FMP quarterly rows into a simpler structure.
    FMP usually returns newest first, but we sort oldest -> newest.
    """
    out = []

    for r in rows:
        revenue = safe_num(r.get("revenue"))
        operating_income = safe_num(r.get("operatingIncome"))
        gross_profit = safe_num(r.get("grossProfit"))

        operating_margin = (
            operating_income / revenue
            if revenue not in (None, 0) and operating_income is not None
            else None
        )
        gross_margin = (
            gross_profit / revenue
            if revenue not in (None, 0) and gross_profit is not None
            else None
        )

        out.append(
            {
                "date": r.get("date"),
                "calendarYear": str(r.get("calendarYear")) if r.get("calendarYear") is not None else None,
                "period": r.get("period"),
                "revenue": revenue,
                "operatingIncome": operating_income,
                "grossProfit": gross_profit,
                "operatingMargin": operating_margin,
                "grossMargin": gross_margin,
            }
        )

    out = [x for x in out if x.get("date")]
    out.sort(key=lambda x: x["date"])
    return out


def find_same_quarter_last_year(rows: List[dict], latest_row: dict) -> Optional[dict]:
    target_period = latest_row.get("period")
    target_year = latest_row.get("calendarYear")

    if not target_period or not target_year:
        return None

    try:
        prior_year = str(int(target_year) - 1)
    except Exception:
        return None

    for r in rows:
        if r.get("period") == target_period and r.get("calendarYear") == prior_year:
            return r

    return None


def compute_revenue_block(rows: List[dict]) -> dict:
    """
    R0 = latest quarter
    R1 = previous quarter
    R2 = two quarters ago
    R3 = same quarter last year
    """
    if len(rows) < 3:
        return {"valid": False}

    row2 = rows[-3]
    row1 = rows[-2]
    row0 = rows[-1]
    row3 = find_same_quarter_last_year(rows, row0)

    R0 = safe_num(row0.get("revenue"))
    R1 = safe_num(row1.get("revenue"))
    R2 = safe_num(row2.get("revenue"))
    R3 = safe_num(row3.get("revenue")) if row3 else None

    r01 = (R0 / R1 - 1) if R0 is not None and R1 not in (None, 0) else None
    r12 = (R1 / R2 - 1) if R1 is not None and R2 not in (None, 0) else None
    r03 = (R0 / R3 - 1) if R0 is not None and R3 not in (None, 0) else None

    # Clamp extreme outliers
    r01 = clamp(r01, -0.50, 1.00)
    r12 = clamp(r12, -0.50, 1.00)
    r03 = clamp(r03, -0.50, 1.50)

    raw_score = weighted_sum(
        {
            "r01": r01,
            "r12": r12,
            "r03": r03,
        },
        REV_WEIGHTS,
    )

    return {
        "valid": True,
        "R0": R0,
        "R1": R1,
        "R2": R2,
        "R3": R3,
        "r01": r01,
        "r12": r12,
        "r03": r03,
        "rawScore": raw_score,
        "latestDate": row0.get("date"),
        "latestPeriod": row0.get("period"),
        "latestYear": row0.get("calendarYear"),
    }


def compute_margin_block(rows: List[dict], margin_key: str, label: str) -> dict:
    """
    Generic function for either operating margin or gross margin.

    Example:
    label='O' => O0, O1, O2, O3 and o01, o12, o03
    label='G' => G0, G1, G2, G3 and g01, g12, g03
    """
    if len(rows) < 3:
        return {"valid": False}

    row2 = rows[-3]
    row1 = rows[-2]
    row0 = rows[-1]
    row3 = find_same_quarter_last_year(rows, row0)

    M0 = safe_num(row0.get(margin_key))
    M1 = safe_num(row1.get(margin_key))
    M2 = safe_num(row2.get(margin_key))
    M3 = safe_num(row3.get(margin_key)) if row3 else None

    d01 = (M0 - M1) if M0 is not None and M1 is not None else None
    d12 = (M1 - M2) if M1 is not None and M2 is not None else None
    d03 = (M0 - M3) if M0 is not None and M3 is not None else None

    # Margin deltas are decimals, so 0.05 means +5 percentage points
    d01 = clamp(d01, -0.15, 0.15)
    d12 = clamp(d12, -0.15, 0.15)
    d03 = clamp(d03, -0.20, 0.20)

    raw_score = weighted_sum(
        {
            "m01": d01,
            "m12": d12,
            "m03": d03,
        },
        MARGIN_WEIGHTS,
    )

    lower = label.lower()

    return {
        "valid": True,
        f"{label}0": M0,
        f"{label}1": M1,
        f"{label}2": M2,
        f"{label}3": M3,
        f"{lower}01": d01,
        f"{lower}12": d12,
        f"{lower}03": d03,
        "rawScore": raw_score,
    }


def main():
    if not FMP_API_KEY:
        raise RuntimeError("Missing FMP_API_KEY environment variable")

    tickers = read_universe()
    print(f"Universe tickers: {len(tickers)}")
    print("First 10 tickers:", tickers[:10])

    os.makedirs("public/data", exist_ok=True)

    payload = {
        "as_of": dt.date.today().isoformat(),
        "source": "Financial Modeling Prep",
        "revenue_weights": REV_WEIGHTS,
        "margin_weights": MARGIN_WEIGHTS,
        "stocks": {},
    }

    tam_raw_map: Dict[str, Optional[float]] = {}
    op_raw_map: Dict[str, Optional[float]] = {}
    gross_raw_map: Dict[str, Optional[float]] = {}
    moat_raw_map: Dict[str, Optional[float]] = {}

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker}")

        stock_obj = {
            "ticker": ticker,
            "revenue": None,
            "operatingMargin": None,
            "grossMargin": None,
            "tamScore": None,
            "moatScore": None,
            "operatingMarginTrendScore": None,
            "grossMarginTrendScore": None,
            "moatSource": None,
            "error": None,
        }

        try:
            rows = fetch_income_statement_quarterly(ticker)
            rows = normalize_rows(rows)

            if len(rows) < 4:
                stock_obj["error"] = "Not enough quarterly rows"
                payload["stocks"][ticker] = stock_obj
                time.sleep(REQUEST_SLEEP)
                continue

            revenue_block = compute_revenue_block(rows)
            op_block = compute_margin_block(rows, "operatingMargin", "O")
            gross_block = compute_margin_block(rows, "grossMargin", "G")

            stock_obj["revenue"] = revenue_block if revenue_block.get("valid") else None
            stock_obj["operatingMargin"] = op_block if op_block.get("valid") else None
            stock_obj["grossMargin"] = gross_block if gross_block.get("valid") else None

            tam_raw = revenue_block.get("rawScore") if revenue_block.get("valid") else None
            op_raw = op_block.get("rawScore") if op_block.get("valid") else None
            gross_raw = gross_block.get("rawScore") if gross_block.get("valid") else None

            O0 = op_block.get("O0") if op_block.get("valid") else None

            # Your fallback rule:
            # if most recent operating margin is negative (or missing), use gross margin trend for MOAT
            if O0 is None or O0 < 0:
                moat_raw = gross_raw
                moat_source = "grossMargin"
            else:
                moat_raw = op_raw
                moat_source = "operatingMargin"

            stock_obj["moatSource"] = moat_source

            tam_raw_map[ticker] = tam_raw
            op_raw_map[ticker] = op_raw
            gross_raw_map[ticker] = gross_raw
            moat_raw_map[ticker] = moat_raw

        except Exception as e:
            stock_obj["error"] = str(e)
            tam_raw_map[ticker] = None
            op_raw_map[ticker] = None
            gross_raw_map[ticker] = None
            moat_raw_map[ticker] = None

        payload["stocks"][ticker] = stock_obj
        time.sleep(REQUEST_SLEEP)

    tam_pct = percentile_rank_map(tam_raw_map)
    op_pct = percentile_rank_map(op_raw_map)
    gross_pct = percentile_rank_map(gross_raw_map)
    moat_pct = percentile_rank_map(moat_raw_map)

    for ticker, stock_obj in payload["stocks"].items():
        stock_obj["tamScore"] = tam_pct.get(ticker)
        stock_obj["operatingMarginTrendScore"] = op_pct.get(ticker)
        stock_obj["grossMarginTrendScore"] = gross_pct.get(ticker)
        stock_obj["moatScore"] = moat_pct.get(ticker)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)")


if __name__ == "__main__":
    main()
