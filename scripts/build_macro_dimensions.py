#!/usr/bin/env python3
"""
Build daily macro dimension scores for Majorah.

Outputs:
- data/macro_anchor_prices.csv
- data/macro_anchor_returns.csv
- data/macro_dimension_scores.csv
- public/data/macro_dimension_scores.json

Dimensions:
1) Inflation
2) Growth
3) Liquidity

Scoring approach:
- Pull daily adjusted close prices for all anchors
- Compute daily returns
- Convert each anchor's return into a rolling z-score
- For each dimension:
    score = mean(zscores of positive anchors) - mean(zscores of negative anchors)

Sign convention:
- Higher inflation_score  => more inflationary day
- Higher growth_score     => more growth / risk-on day
- Higher liquidity_score  => easier liquidity / easier financial conditions

If you want "tight liquidity" instead, multiply liquidity_score by -1.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


# =========================
# Config
# =========================

START_DATE = "2024-01-01"
WARMUP_START = "2023-06-01"   # extra history so rolling z-scores stabilize
ROLLING_WINDOW = 60           # trading days for z-score normalization
MIN_PERIODS = 20
CLIP_Z = 3.0                  # clip extreme z-scores
SMOOTH_SPAN = 5               # EMA smoothing for optional smoothed scores

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = ROOT / "public" / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Anchor definitions
# ---------------------------------------------------------
# "positive" = anchors that should push the dimension score UP
# "negative" = anchors that should push the dimension score DOWN
#
# These are based on your simplified framework, with one practical tweak:
# for liquidity, positive means easier liquidity, so HYG is positive and UUP is negative.
#
# You can edit these freely.
# ---------------------------------------------------------

DIMENSIONS: Dict[str, Dict[str, List[str]]] = {
    "inflation": {
        "positive": ["GLD", "XLE"],
        "negative": ["TLT"],
    },
    "growth": {
        "positive": ["QQQ", "IWM"],
        "negative": ["PSQ", "SH"],
    },
    "liquidity": {
        "positive": ["HYG"],   # easier liquidity / better credit conditions
        "negative": ["UUP"],   # dollar strength = tighter global liquidity
    },
}

# If you decide later that inverse ETFs are too path-dependent,
# a cleaner alternative growth definition would be:
#
# "growth": {
#     "positive": ["QQQ", "IWM"],
#     "negative": ["XLP", "XLU"],
# }


# =========================
# Helpers
# =========================

def flatten_unique_tickers(dim_map: Dict[str, Dict[str, List[str]]]) -> List[str]:
    tickers = []
    seen = set()
    for dim_cfg in dim_map.values():
        for side in ("positive", "negative"):
            for tkr in dim_cfg.get(side, []):
                if tkr not in seen:
                    seen.add(tkr)
                    tickers.append(tkr)
    return tickers


def download_prices(tickers: List[str], start_date: str) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    """
    if not tickers:
        raise ValueError("No tickers supplied.")

    df = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df.empty:
        raise ValueError("No price data downloaded.")

    # yfinance returns different shapes depending on number of tickers
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.levels[0]:
            prices = df["Close"].copy()
        else:
            # Fallback to adjusted close naming if needed
            first_level = list(df.columns.levels[0])
            raise ValueError(f"Unexpected yfinance column structure: {first_level}")
    else:
        # single ticker case
        prices = df.to_frame(name=tickers[0])

    prices = prices.sort_index()
    prices = prices.dropna(how="all")

    # Keep only requested tickers in the original order
    prices = prices[[c for c in tickers if c in prices.columns]]

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"Warning: missing tickers in downloaded data: {missing}")

    return prices


def rolling_zscore(
    s: pd.Series,
    window: int = 60,
    min_periods: int = 20,
    clip_z: float | None = 3.0,
) -> pd.Series:
    """
    Rolling z-score of a series.
    """
    roll_mean = s.rolling(window=window, min_periods=min_periods).mean()
    roll_std = s.rolling(window=window, min_periods=min_periods).std(ddof=0)

    z = (s - roll_mean) / roll_std.replace(0, np.nan)

    if clip_z is not None:
        z = z.clip(lower=-clip_z, upper=clip_z)

    return z


def build_dimension_score(
    zret: pd.DataFrame,
    positive: List[str],
    negative: List[str],
) -> pd.Series:
    """
    score = mean(positive anchors) - mean(negative anchors)
    """
    pos_cols = [c for c in positive if c in zret.columns]
    neg_cols = [c for c in negative if c in zret.columns]

    if not pos_cols and not neg_cols:
        return pd.Series(index=zret.index, dtype=float)

    pos_mean = zret[pos_cols].mean(axis=1) if pos_cols else pd.Series(0.0, index=zret.index)
    neg_mean = zret[neg_cols].mean(axis=1) if neg_cols else pd.Series(0.0, index=zret.index)

    return pos_mean - neg_mean


def pct_rank_expanding(s: pd.Series) -> pd.Series:
    """
    Expanding percentile rank:
    on each date, rank today's value against all values observed up to that date.
    """
    vals = []
    ranks = []

    for x in s.values:
        vals.append(x)
        arr = pd.Series(vals, dtype=float).dropna()
        if arr.empty or pd.isna(x):
            ranks.append(np.nan)
        else:
            ranks.append((arr <= x).mean())

    return pd.Series(ranks, index=s.index)


def safe_round(v, n=6):
    if pd.isna(v):
        return None
    return round(float(v), n)


# =========================
# Main
# =========================

def main():
    tickers = flatten_unique_tickers(DIMENSIONS)
    print("Downloading anchors:", tickers)

    prices = download_prices(tickers, WARMUP_START)
    returns = prices.pct_change()

    # Rolling z-score normalize each anchor return series
    zret = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        zret[col] = rolling_zscore(
            returns[col],
            window=ROLLING_WINDOW,
            min_periods=MIN_PERIODS,
            clip_z=CLIP_Z,
        )

    # Build dimension scores
    scores = pd.DataFrame(index=returns.index)

    for dim_name, cfg in DIMENSIONS.items():
        raw = build_dimension_score(
            zret=zret,
            positive=cfg.get("positive", []),
            negative=cfg.get("negative", []),
        )

        smoothed = raw.ewm(span=SMOOTH_SPAN, adjust=False).mean()
        rank = pct_rank_expanding(raw)

        scores[f"{dim_name}_score"] = raw
        scores[f"{dim_name}_score_ema{SMOOTH_SPAN}"] = smoothed
        scores[f"{dim_name}_pct_rank"] = rank

    # Restrict final output to requested period
    prices_out = prices.loc[prices.index >= pd.Timestamp(START_DATE)].copy()
    returns_out = returns.loc[returns.index >= pd.Timestamp(START_DATE)].copy()
    scores_out = scores.loc[scores.index >= pd.Timestamp(START_DATE)].copy()

    # Add helpful metadata columns
    scores_out = scores_out.reset_index().rename(columns={"Date": "date", "index": "date"})
    scores_out["date"] = pd.to_datetime(scores_out["date"]).dt.strftime("%Y-%m-%d")

    # Save CSVs
    prices_out.to_csv(DATA_DIR / "macro_anchor_prices.csv", index_label="date")
    returns_out.to_csv(DATA_DIR / "macro_anchor_returns.csv", index_label="date")
    scores_out.to_csv(DATA_DIR / "macro_dimension_scores.csv", index=False)

    # Save JSON for frontend use
    json_rows = []
    for _, row in scores_out.iterrows():
        json_rows.append(
            {
                "date": row["date"],
                "inflation_score": safe_round(row.get("inflation_score")),
                f"inflation_score_ema{SMOOTH_SPAN}": safe_round(row.get(f"inflation_score_ema{SMOOTH_SPAN}")),
                "inflation_pct_rank": safe_round(row.get("inflation_pct_rank")),
                "growth_score": safe_round(row.get("growth_score")),
                f"growth_score_ema{SMOOTH_SPAN}": safe_round(row.get(f"growth_score_ema{SMOOTH_SPAN}")),
                "growth_pct_rank": safe_round(row.get("growth_pct_rank")),
                "liquidity_score": safe_round(row.get("liquidity_score")),
                f"liquidity_score_ema{SMOOTH_SPAN}": safe_round(row.get(f"liquidity_score_ema{SMOOTH_SPAN}")),
                "liquidity_pct_rank": safe_round(row.get("liquidity_pct_rank")),
            }
        )

    with open(PUBLIC_DATA_DIR / "macro_dimension_scores.json", "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)

    # Also save the z-scored anchor returns for debugging / diagnostics
    zret_out = zret.loc[zret.index >= pd.Timestamp(START_DATE)].copy()
    zret_out.to_csv(DATA_DIR / "macro_anchor_zscores.csv", index_label="date")

    print("Done.")
    print(f"Saved: {DATA_DIR / 'macro_anchor_prices.csv'}")
    print(f"Saved: {DATA_DIR / 'macro_anchor_returns.csv'}")
    print(f"Saved: {DATA_DIR / 'macro_anchor_zscores.csv'}")
    print(f"Saved: {DATA_DIR / 'macro_dimension_scores.csv'}")
    print(f"Saved: {PUBLIC_DATA_DIR / 'macro_dimension_scores.json'}")


if __name__ == "__main__":
    main()
