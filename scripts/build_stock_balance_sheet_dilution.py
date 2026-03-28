#!/usr/bin/env python3
"""
Build per-stock balance sheet and dilution metrics for Majorah.

Reads:
- data/universe.csv   (expects a Ticker column)

Writes:
- public/data/stock_balance_sheet_dilution.json
- data/stock_balance_sheet_dilution.csv

Metrics included per ticker:
- cash
- total debt
- net cash
- current ratio
- quick ratio
- debt to equity
- free cash flow
- shares outstanding
- shares outstanding 1 year ago
- shares outstanding 3 years ago
- share count growth 1Y
- share count growth 3Y
- stock based compensation
- sbc to revenue
- simple balance sheet risk score (0-100, higher = better)
- simple dilution risk score (0-100, higher = worse)

Notes:
- Uses yfinance info/financial statement fields where available.
- Many fields can be missing or inconsistent by ticker.
- The scoring here is intentionally simple and transparent so it can be improved later.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = ROOT / "public" / "data"

UNIVERSE_PATH = DATA_DIR / "universe.csv"
OUTPUT_JSON_PATH = PUBLIC_DATA_DIR / "stock_balance_sheet_dilution.json"
OUTPUT_CSV_PATH = DATA_DIR / "stock_balance_sheet_dilution.csv"


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def get_first_present(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            val = safe_float(d.get(k))
            if val is not None:
                return val
    return None


def extract_series_value(df: pd.DataFrame, row_candidates: List[str], col_index: int = 0) -> Optional[float]:
    """
    Pull a value from a financial statement dataframe by row name candidates.
    Uses positional column index (0 = most recent, 1 = prior year, etc.).
    """
    if df is None or df.empty:
        return None

    for row_name in row_candidates:
        if row_name in df.index:
            row = df.loc[row_name]
            if hasattr(row, "iloc") and len(row) > col_index:
                return safe_float(row.iloc[col_index])
    return None


def compute_balance_sheet_score(
    current_ratio: Optional[float],
    quick_ratio: Optional[float],
    debt_to_equity: Optional[float],
    net_cash: Optional[float],
    free_cash_flow: Optional[float],
) -> Optional[float]:
    parts = []

    if current_ratio is not None:
        if current_ratio >= 2.0:
            parts.append(100)
        elif current_ratio >= 1.5:
            parts.append(85)
        elif current_ratio >= 1.0:
            parts.append(65)
        else:
            parts.append(30)

    if quick_ratio is not None:
        if quick_ratio >= 1.5:
            parts.append(100)
        elif quick_ratio >= 1.0:
            parts.append(80)
        elif quick_ratio >= 0.75:
            parts.append(60)
        else:
            parts.append(30)

    if debt_to_equity is not None:
        if debt_to_equity <= 0.3:
            parts.append(100)
        elif debt_to_equity <= 0.7:
            parts.append(80)
        elif debt_to_equity <= 1.5:
            parts.append(55)
        else:
            parts.append(25)

    if net_cash is not None:
        parts.append(100 if net_cash > 0 else 40)

    if free_cash_flow is not None:
        parts.append(100 if free_cash_flow > 0 else 35)

    if not parts:
        return None

    return round(float(np.mean(parts)), 1)


def compute_dilution_risk_score(
    share_count_growth_1y: Optional[float],
    share_count_growth_3y: Optional[float],
    sbc_to_revenue: Optional[float],
) -> Optional[float]:
    """
    Higher = worse dilution risk.
    """
    parts = []

    if share_count_growth_1y is not None:
        g = share_count_growth_1y
        if g <= 0:
            parts.append(5)
        elif g <= 0.01:
            parts.append(15)
        elif g <= 0.03:
            parts.append(35)
        elif g <= 0.07:
            parts.append(65)
        else:
            parts.append(90)

    if share_count_growth_3y is not None:
        g = share_count_growth_3y
        if g <= 0:
            parts.append(5)
        elif g <= 0.03:
            parts.append(15)
        elif g <= 0.08:
            parts.append(35)
        elif g <= 0.15:
            parts.append(65)
        else:
            parts.append(90)

    if sbc_to_revenue is not None:
        s = sbc_to_revenue
        if s <= 0.01:
            parts.append(5)
        elif s <= 0.03:
            parts.append(20)
        elif s <= 0.06:
            parts.append(45)
        elif s <= 0.10:
            parts.append(70)
        else:
            parts.append(90)

    if not parts:
        return None

    return round(float(np.mean(parts)), 1)


def normalize_ticker_for_yfinance(ticker: str) -> str:
    return ticker.replace(".", "-")


def fetch_metrics_for_ticker(ticker: str) -> Dict[str, Any]:
    yf_ticker = normalize_ticker_for_yfinance(ticker)
    tk = yf.Ticker(yf_ticker)

    result: Dict[str, Any] = {
        "ticker": ticker,
        "cash": None,
        "totalDebt": None,
        "netCash": None,
        "currentRatio": None,
        "quickRatio": None,
        "debtToEquity": None,
        "freeCashFlow": None,
        "sharesOutCurrent": None,
        "sharesOut1Y": None,
        "sharesOut3Y": None,
        "shareCountGrowth1Y": None,
        "shareCountGrowth3Y": None,
        "stockBasedComp": None,
        "revenue": None,
        "sbcToRevenue": None,
        "balanceSheetRiskScore": None,
        "dilutionRiskScore": None,
    }

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    try:
        balance_sheet = tk.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()

    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    try:
        financials = tk.financials
    except Exception:
        financials = pd.DataFrame()

    cash = get_first_present(
        info,
        ["totalCash", "cash", "cashAndCashEquivalents"],
    )
    total_debt = get_first_present(
        info,
        ["totalDebt"],
    )
    current_ratio = get_first_present(
        info,
        ["currentRatio"],
    )
    quick_ratio = get_first_present(
        info,
        ["quickRatio"],
    )
    debt_to_equity = get_first_present(
        info,
        ["debtToEquity"],
    )
    free_cash_flow = get_first_present(
        info,
        ["freeCashflow", "freeCashFlow"],
    )
    shares_out_current = get_first_present(
        info,
        ["sharesOutstanding", "impliedSharesOutstanding"],
    )

    if cash is None:
        cash = extract_series_value(
            balance_sheet,
            ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash"],
            0,
        )

    if total_debt is None:
        total_debt = extract_series_value(
            balance_sheet,
            ["Total Debt", "Current Debt And Capital Lease Obligation", "Long Term Debt"],
            0,
        )

    if shares_out_current is None:
        shares_out_current = extract_series_value(
            balance_sheet,
            ["Ordinary Shares Number", "Share Issued"],
            0,
        )

    shares_out_1y = extract_series_value(
        balance_sheet,
        ["Ordinary Shares Number", "Share Issued"],
        1,
    )
    shares_out_3y = extract_series_value(
        balance_sheet,
        ["Ordinary Shares Number", "Share Issued"],
        3,
    )

    stock_based_comp = extract_series_value(
        cashflow,
        ["Stock Based Compensation", "StockBasedCompensation"],
        0,
    )

    revenue = extract_series_value(
        financials,
        ["Total Revenue", "Operating Revenue", "Revenue"],
        0,
    )

    net_cash = None
    if cash is not None and total_debt is not None:
        net_cash = cash - total_debt
    elif cash is not None and total_debt is None:
        net_cash = cash

    share_count_growth_1y = pct_change(shares_out_current, shares_out_1y)
    share_count_growth_3y = pct_change(shares_out_current, shares_out_3y)

    sbc_to_revenue = None
    if stock_based_comp is not None and revenue not in (None, 0):
        sbc_to_revenue = stock_based_comp / revenue

    balance_sheet_score = compute_balance_sheet_score(
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        debt_to_equity=debt_to_equity,
        net_cash=net_cash,
        free_cash_flow=free_cash_flow,
    )

    dilution_risk_score = compute_dilution_risk_score(
        share_count_growth_1y=share_count_growth_1y,
        share_count_growth_3y=share_count_growth_3y,
        sbc_to_revenue=sbc_to_revenue,
    )

    result.update(
        {
            "cash": cash,
            "totalDebt": total_debt,
            "netCash": net_cash,
            "currentRatio": current_ratio,
            "quickRatio": quick_ratio,
            "debtToEquity": debt_to_equity,
            "freeCashFlow": free_cash_flow,
            "sharesOutCurrent": shares_out_current,
            "sharesOut1Y": shares_out_1y,
            "sharesOut3Y": shares_out_3y,
            "shareCountGrowth1Y": share_count_growth_1y,
            "shareCountGrowth3Y": share_count_growth_3y,
            "stockBasedComp": stock_based_comp,
            "revenue": revenue,
            "sbcToRevenue": sbc_to_revenue,
            "balanceSheetRiskScore": balance_sheet_score,
            "dilutionRiskScore": dilution_risk_score,
        }
    )

    return result


def load_universe() -> List[str]:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Missing universe file: {UNIVERSE_PATH}")

    df = pd.read_csv(UNIVERSE_PATH)
    if "Ticker" not in df.columns:
        raise ValueError("data/universe.csv must contain a 'Ticker' column")

    tickers = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .tolist()
    )
    return tickers


def main() -> None:
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_universe()
    rows: List[Dict[str, Any]] = []

    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{total}] {ticker}")
        try:
            row = fetch_metrics_for_ticker(ticker)
            rows.append(row)
        except Exception as e:
            print(f"  Failed for {ticker}: {e}")
            rows.append({"ticker": ticker})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote CSV:  {OUTPUT_CSV_PATH}")
    print(f"Wrote JSON: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
