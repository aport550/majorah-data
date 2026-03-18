#!/usr/bin/env python3
"""
Build stock / ETF macro exposure scores for Majorah.

Reads:
- data/daily_returns.csv
- data/macro_dimension_scores.csv

Writes:
- data/stock_macro_exposures.csv
- public/data/stock_macro_exposures.json

For each stock/ETF, computes exposure to:
- inflation
- growth
- liquidity

Metrics included per dimension:
- correlation
- beta
- r_squared
- signed score (-100 to 100)
- normalized score (0 to 100)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = ROOT / "public" / "data"

INPUT_DAILY_RETURNS = DATA_DIR / "daily_returns.csv"
INPUT_MACRO_SCORES = DATA_DIR / "macro_dimension_scores.csv"

OUTPUT_CSV = DATA_DIR / "stock_macro_exposures.csv"
OUTPUT_JSON = PUBLIC_DATA_DIR / "stock_macro_exposures.json"

DATE_COL = "date"

MACRO_DIMENSIONS = [
    "inflation",
    "growth",
    "liquidity",
]

MACRO_SCORE_COLS = {
    "inflation": "inflation_score",
    "growth": "growth_score",
    "liquidity": "liquidity_score",
}

MIN_OBS = 40

WINSORIZE_RETURNS = True
RETURN_LOWER_Q = 0.01
RETURN_UPPER_Q = 0.99

SIGNED_SCORE_CLIP = 100.0


# =========================================================
# Helpers
# =========================================================

def read_csv_with_date_index(path: Path, date_col: str = DATE_COL) -> pd.DataFrame:
    """
    Read a CSV and set a date index robustly.

    Supported cases:
    - explicit 'date' column
    - explicit 'Date' column
    - pandas-saved unnamed first index column, e.g. 'Unnamed: 0'
    """
    df = pd.read_csv(path)
    print(f"Reading {path}")
    print(f"Columns found: {list(df.columns)}")

    chosen_date_col = None

    if date_col in df.columns:
        chosen_date_col = date_col
    elif "Date" in df.columns:
        chosen_date_col = "Date"
    elif len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        chosen_date_col = df.columns[0]

    if chosen_date_col is None:
        raise ValueError(
            f"{path} is missing a '{DATE_COL}' or 'Date' column, "
            f"and first column is not an unnamed date index. "
            f"Found columns: {list(df.columns)}"
        )

    df[chosen_date_col] = pd.to_datetime(df[chosen_date_col], errors="coerce")
    df = df.dropna(subset=[chosen_date_col])
    df = df.sort_values(chosen_date_col).set_index(chosen_date_col)

    # Remove leftover unnamed columns after setting index
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    return df


def winsorize_series(s: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < MIN_OBS or len(y) < MIN_OBS:
        return np.nan
    if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return np.nan
    return x.corr(y)


def safe_beta(asset_ret: pd.Series, factor: pd.Series) -> float:
    """
    Beta = Cov(asset, factor) / Var(factor)
    """
    if len(asset_ret) < MIN_OBS or len(factor) < MIN_OBS:
        return np.nan

    var_f = factor.var(ddof=0)
    if pd.isna(var_f) or var_f == 0:
        return np.nan

    cov = np.cov(asset_ret.values, factor.values, ddof=0)[0, 1]
    return cov / var_f


def safe_r_squared(asset_ret: pd.Series, factor: pd.Series) -> float:
    """
    For simple 1-factor regression with intercept,
    R^2 is corr(asset, factor)^2.
    """
    corr = safe_corr(asset_ret, factor)
    if pd.isna(corr):
        return np.nan
    return corr * corr


def normalize_cross_section(series: pd.Series) -> pd.Series:
    """
    Min-max normalize to 0..100.
    If constant or all NaN, returns NaN series.
    """
    s = series.copy()
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)

    mn = valid.min()
    mx = valid.max()

    if mn == mx:
        out = pd.Series(50.0, index=s.index)
        out[s.isna()] = np.nan
        return out

    out = (s - mn) / (mx - mn) * 100.0
    return out


def signed_score_from_beta_corr(beta: pd.Series, corr: pd.Series) -> pd.Series:
    """
    Blend beta magnitude and directional correlation into a signed score.
    """
    beta_rank = beta.rank(pct=True)
    beta_centered = (beta_rank - 0.5) * 2.0

    corr_filled = corr.copy().clip(-1, 1)

    signed_raw = 0.65 * beta_centered + 0.35 * corr_filled
    signed_score = (signed_raw * 100.0).clip(-SIGNED_SCORE_CLIP, SIGNED_SCORE_CLIP)
    signed_score[beta.isna() | corr.isna()] = np.nan
    return signed_score


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_json_value(x, digits=None):
    """
    Convert pandas/numpy NaN to None so output is valid JSON.
    Optionally round floats.
    """
    if pd.isna(x):
        return None

    if isinstance(x, (np.integer, int)):
        return int(x)

    if isinstance(x, (np.floating, float)):
        val = float(x)
        if digits is not None:
            val = round(val, digits)
        return val

    return x


# =========================================================
# Main pipeline
# =========================================================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DAILY_RETURNS.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_DAILY_RETURNS}")

    if not INPUT_MACRO_SCORES.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_MACRO_SCORES}")

    # -----------------------------
    # Load inputs
    # -----------------------------
    daily_returns = read_csv_with_date_index(INPUT_DAILY_RETURNS)
    macro_scores = read_csv_with_date_index(INPUT_MACRO_SCORES)

    daily_returns = clean_numeric_columns(daily_returns)
    macro_scores = clean_numeric_columns(macro_scores)

    print(f"daily_returns shape: {daily_returns.shape}")
    print(f"macro_scores shape: {macro_scores.shape}")

    needed_macro_cols = [MACRO_SCORE_COLS[d] for d in MACRO_DIMENSIONS]
    missing_macro_cols = [c for c in needed_macro_cols if c not in macro_scores.columns]
    if missing_macro_cols:
        raise ValueError(
            f"macro_dimension_scores.csv is missing required columns: {missing_macro_cols}. "
            f"Found columns: {list(macro_scores.columns)}"
        )

    macro_scores = macro_scores[needed_macro_cols].copy()

    # Optional winsorization of returns
    if WINSORIZE_RETURNS:
        for col in daily_returns.columns:
            daily_returns[col] = winsorize_series(
                daily_returns[col],
                RETURN_LOWER_Q,
                RETURN_UPPER_Q,
            )

    # -----------------------------
    # Align dates
    # -----------------------------
    common_dates = daily_returns.index.intersection(macro_scores.index)
    common_dates = common_dates.sort_values()

    print(f"Overlapping dates: {len(common_dates)}")

    if len(common_dates) < MIN_OBS:
        raise ValueError(
            f"Not enough overlapping dates between returns and macro scores: {len(common_dates)}"
        )

    daily_returns = daily_returns.loc[common_dates].copy()
    macro_scores = macro_scores.loc[common_dates].copy()

    daily_returns = daily_returns.dropna(axis=1, how="all")

    # -----------------------------
    # Compute exposures
    # -----------------------------
    tickers = list(daily_returns.columns)
    rows = []

    for ticker in tickers:
        asset = daily_returns[ticker].dropna()

        if asset.empty:
            continue

        row = {
            "ticker": ticker,
            "n_obs_total": int(asset.shape[0]),
        }

        for dim in MACRO_DIMENSIONS:
            factor_col = MACRO_SCORE_COLS[dim]

            tmp = pd.concat(
                [daily_returns[[ticker]], macro_scores[[factor_col]]],
                axis=1,
                join="inner",
            ).dropna()

            n_obs = len(tmp)
            row[f"{dim}_n_obs"] = int(n_obs)

            if n_obs < MIN_OBS:
                row[f"{dim}_corr"] = np.nan
                row[f"{dim}_beta"] = np.nan
                row[f"{dim}_r2"] = np.nan
                continue

            asset_ret = tmp[ticker]
            factor = tmp[factor_col]

            row[f"{dim}_corr"] = safe_corr(asset_ret, factor)
            row[f"{dim}_beta"] = safe_beta(asset_ret, factor)
            row[f"{dim}_r2"] = safe_r_squared(asset_ret, factor)

        rows.append(row)

    exposures = pd.DataFrame(rows)

    if exposures.empty:
        raise ValueError("No exposures were computed. Check your input files.")

    # -----------------------------
    # Build signed and normalized scores
    # -----------------------------
    for dim in MACRO_DIMENSIONS:
        beta_col = f"{dim}_beta"
        corr_col = f"{dim}_corr"
        signed_col = f"{dim}_signed_score"
        norm_col = f"{dim}_score"

        exposures[signed_col] = signed_score_from_beta_corr(
            exposures[beta_col],
            exposures[corr_col],
        )
        exposures[norm_col] = normalize_cross_section(exposures[signed_col])

    # -----------------------------
    # Dominant dimension labels
    # -----------------------------
    def get_top_dimension(row: pd.Series):
        vals = {
            dim: abs(row.get(f"{dim}_signed_score", np.nan))
            for dim in MACRO_DIMENSIONS
        }
        vals = {k: v for k, v in vals.items() if pd.notna(v)}
        if not vals:
            return None
        return max(vals, key=vals.get)

    exposures["primary_macro_dimension"] = exposures.apply(get_top_dimension, axis=1)

    # -----------------------------
    # Reorder columns nicely
    # -----------------------------
    ordered_cols = ["ticker", "n_obs_total"]

    for dim in MACRO_DIMENSIONS:
        ordered_cols += [
            f"{dim}_n_obs",
            f"{dim}_corr",
            f"{dim}_beta",
            f"{dim}_r2",
            f"{dim}_signed_score",
            f"{dim}_score",
        ]

    ordered_cols += ["primary_macro_dimension"]

    exposures = exposures[[c for c in ordered_cols if c in exposures.columns]].copy()
    exposures = exposures.sort_values("ticker").reset_index(drop=True)

    # -----------------------------
    # Save CSV
    # -----------------------------
    exposures.to_csv(OUTPUT_CSV, index=False)

    # -----------------------------
    # Save JSON
    # -----------------------------
    json_rows = []
    for _, r in exposures.iterrows():
        item = {
            "ticker": clean_json_value(r.get("ticker")),
            "n_obs_total": clean_json_value(r.get("n_obs_total")),
            "primary_macro_dimension": clean_json_value(r.get("primary_macro_dimension")),
        }

        for dim in MACRO_DIMENSIONS:
            item[f"{dim}_n_obs"] = clean_json_value(r.get(f"{dim}_n_obs"))
            item[f"{dim}_corr"] = clean_json_value(r.get(f"{dim}_corr"), 6)
            item[f"{dim}_beta"] = clean_json_value(r.get(f"{dim}_beta"), 6)
            item[f"{dim}_r2"] = clean_json_value(r.get(f"{dim}_r2"), 6)
            item[f"{dim}_signed_score"] = clean_json_value(r.get(f"{dim}_signed_score"), 4)
            item[f"{dim}_score"] = clean_json_value(r.get(f"{dim}_score"), 2)

        json_rows.append(item)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2, allow_nan=False)

    print("Done.")
    print(f"Saved CSV:  {OUTPUT_CSV}")
    print(f"Saved JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
