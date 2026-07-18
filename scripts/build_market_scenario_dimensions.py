#!/usr/bin/env python3
"""
Build seven daily market-scenario scores and estimate each stock's exposure.

Expected input:
- data/universe.csv

The universe file may contain a ticker column named one of:
- ticker
- Ticker
- symbol
- Symbol

Outputs:
- data/scenario_anchor_prices.csv
- data/scenario_anchor_returns.csv
- data/scenario_anchor_zscores.csv
- data/market_scenario_daily_scores.csv
- public/data/market_scenario_daily_scores.json
- data/universe_scenario_exposures.csv
- public/data/universe_scenario_exposures.json

Seven scenario dimensions:
1) Jobs Numbers Terrible
2) High CPI Surprise
3) Corporate Tax Skyrockets
4) Margins Compress Because of AI
5) US Credit Crisis
6) Geopolitical Disaster
7) Liquidity Crash

Daily scoring approach:
- Download adjusted-close prices for the scenario ETFs.
- Compute daily ETF returns.
- Convert each ETF return to a rolling z-score.
- For each scenario:
      scenario_score = mean(positive-anchor z-scores)
                       - mean(negative-anchor z-scores)
- A higher score means the market is behaving more like that scenario.

Stock exposure approach:
- Download adjusted-close prices for every ticker in universe.csv.
- Compute stock daily returns.
- Align each stock's returns with each daily scenario score.
- Calculate:
    * Pearson correlation
    * OLS beta: covariance(stock return, scenario score) / variance(scenario score)
    * t-statistic for the correlation
    * observation count
    * exposure score from 0 to 100:
          0   = strongly harmed as the scenario intensifies
          50  = little relationship
          100 = strongly benefits as the scenario intensifies
      exposure_score = 50 * (correlation + 1)
    * vulnerability score from 0 to 100:
          vulnerability_score = 100 - exposure_score

Important interpretation:
- The scenario score is a market-behavior proxy, not proof that a particular
  economic or political event actually occurred.
- Corporate-tax shocks are especially difficult to identify from ETF prices
  alone. The selected ETFs measure the expected market pattern, not legislation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2024-01-01"
WARMUP_START = "2023-06-01"
ROLLING_WINDOW = 60
MIN_PERIODS = 20
CLIP_Z = 3.0
SMOOTH_SPAN = 5

# Minimum overlapping daily observations required for a stock/scenario estimate.
MIN_CORRELATION_OBSERVATIONS = 60

# Download stocks in batches to reduce memory pressure and make failures easier
# to recover from.
STOCK_DOWNLOAD_BATCH_SIZE = 100

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = ROOT / "public" / "data"
UNIVERSE_PATH = DATA_DIR / "universe.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Scenario definitions
# =============================================================================

# Sign convention:
# - positive anchors rising pushes the scenario score higher.
# - negative anchors rising pushes the scenario score lower.
#
# Therefore, a high score means ETF behavior is consistent with the named shock.
SCENARIOS: Dict[str, Dict[str, object]] = {
    "jobs_terrible": {
        "label": "Jobs Numbers Terrible",
        "positive": ["TLT", "XLP"],
        "negative": ["IWM", "XLY"],
        "description": (
            "Long Treasuries and consumer staples outperform while small caps "
            "and consumer discretionary weaken."
        ),
    },
    "high_cpi": {
        "label": "High CPI Surprise",
        "positive": ["RINF", "PDBC"],
        "negative": ["TLT"],
        "description": (
            "Inflation expectations and commodities strengthen while long-duration "
            "Treasuries weaken."
        ),
    },
    "corporate_tax": {
        "label": "Corporate Tax Skyrockets",
        "positive": ["VXUS"],
        "negative": ["IWM", "VTI"],
        "description": (
            "International equities outperform domestically exposed US equities."
        ),
    },
    "ai_margin_compression": {
        "label": "Margins Compress Because of AI",
        "positive": ["SMH"],
        "negative": ["XSW", "IGV"],
        "description": (
            "AI infrastructure and semiconductors outperform software businesses "
            "whose pricing power may be pressured by AI."
        ),
    },
    "credit_crisis": {
        "label": "US Credit Crisis",
        "positive": ["IEF"],
        "negative": ["HYG", "LQD", "KRE"],
        "description": (
            "Treasuries outperform high-yield credit, investment-grade credit, "
            "and regional banks."
        ),
    },
    "geopolitical_disaster": {
        "label": "Geopolitical Disaster",
        "positive": ["USO", "GLD", "XAR"],
        "negative": ["SPY"],
        "description": (
            "Oil, gold, and defense equities strengthen while the broad equity "
            "market weakens."
        ),
    },
    "liquidity_crash": {
        "label": "Liquidity Crash",
        "positive": ["IEF"],
        "negative": ["HYG", "LQD", "SPY"],
        "description": (
            "Treasuries outperform risky credit, investment-grade credit, and "
            "equities during broad forced selling."
        ),
    },
}


# =============================================================================
# General helpers
# =============================================================================


def normalize_ticker(value: object) -> str:
    """Normalize a ticker while preserving dots and hyphens."""
    return re.sub(r"[^A-Z0-9.\-]", "", str(value or "").strip().upper())


def flatten_unique_tickers(
    scenario_map: Mapping[str, Mapping[str, object]],
) -> List[str]:
    tickers: List[str] = []
    seen: set[str] = set()

    for config in scenario_map.values():
        for side in ("positive", "negative"):
            for ticker in config.get(side, []):
                normalized = normalize_ticker(ticker)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    tickers.append(normalized)

    return tickers


def chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def safe_round(value: object, digits: int = 6):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None

    return round(numeric, digits)


def frame_to_json_records(df: pd.DataFrame, digits: int = 6) -> List[dict]:
    records: List[dict] = []

    for _, row in df.iterrows():
        record = {}
        for column, value in row.items():
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                record[column] = pd.Timestamp(value).strftime("%Y-%m-%d")
            elif isinstance(value, (float, np.floating, int, np.integer)):
                record[column] = safe_round(value, digits)
            elif pd.isna(value):
                record[column] = None
            else:
                record[column] = value
        records.append(record)

    return records


# =============================================================================
# Input and download helpers
# =============================================================================


def read_universe_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found: {path}. Expected data/universe.csv."
        )

    df = pd.read_csv(path)
    ticker_column = next(
        (column for column in ("ticker", "Ticker", "symbol", "Symbol") if column in df.columns),
        None,
    )

    if ticker_column is None:
        raise ValueError(
            "universe.csv must contain one of these columns: "
            "ticker, Ticker, symbol, Symbol"
        )

    tickers: List[str] = []
    seen: set[str] = set()

    for raw in df[ticker_column].tolist():
        ticker = normalize_ticker(raw)
        if ticker and ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError("No valid tickers found in universe.csv.")

    return tickers


def extract_close_prices(downloaded: pd.DataFrame, requested: Sequence[str]) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-ticker output into a price frame."""
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        first_level = downloaded.columns.get_level_values(0)
        if "Close" not in first_level:
            raise ValueError(
                "Unexpected yfinance response: no Close field in MultiIndex columns."
            )
        prices = downloaded["Close"].copy()
    else:
        # A single ticker may return ordinary OHLCV columns.
        if "Close" not in downloaded.columns:
            raise ValueError("Unexpected yfinance response: no Close column.")
        name = requested[0] if requested else "ticker"
        prices = downloaded[["Close"]].rename(columns={"Close": name})

    if isinstance(prices, pd.Series):
        name = requested[0] if requested else "ticker"
        prices = prices.to_frame(name=name)

    prices.columns = [normalize_ticker(column) for column in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index().dropna(how="all")

    ordered = [ticker for ticker in requested if ticker in prices.columns]
    return prices[ordered] if ordered else pd.DataFrame(index=prices.index)


def download_prices(tickers: Sequence[str], start_date: str) -> pd.DataFrame:
    if not tickers:
        raise ValueError("No tickers supplied.")

    downloaded = yf.download(
        tickers=list(tickers),
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = extract_close_prices(downloaded, tickers)
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        print(f"Warning: no downloaded price series for {len(missing)} tickers: {missing}")

    return prices


def download_stock_prices_batched(
    tickers: Sequence[str],
    start_date: str,
    batch_size: int,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    batches = list(chunked(list(tickers), batch_size))
    for batch_index, batch in enumerate(batches, start=1):
        print(
            f"Downloading universe batch {batch_index}/{len(batches)} "
            f"({len(batch)} tickers)..."
        )
        try:
            batch_prices = download_prices(batch, start_date)
        except Exception as exc:
            print(f"Warning: batch failed: {exc}")
            continue

        if not batch_prices.empty:
            frames.append(batch_prices)

    if not frames:
        raise ValueError("No universe stock prices were downloaded.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index().dropna(how="all")

    ordered = [ticker for ticker in tickers if ticker in prices.columns]
    return prices[ordered]


# =============================================================================
# Scenario-score construction
# =============================================================================


def rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
    clip_z: float | None,
) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)

    if clip_z is not None:
        zscore = zscore.clip(lower=-clip_z, upper=clip_z)

    return zscore


def build_scenario_score(
    z_returns: pd.DataFrame,
    positive: Sequence[str],
    negative: Sequence[str],
) -> pd.Series:
    positive_columns = [ticker for ticker in positive if ticker in z_returns.columns]
    negative_columns = [ticker for ticker in negative if ticker in z_returns.columns]

    if not positive_columns and not negative_columns:
        return pd.Series(index=z_returns.index, dtype=float)

    positive_mean = (
        z_returns[positive_columns].mean(axis=1)
        if positive_columns
        else pd.Series(0.0, index=z_returns.index)
    )
    negative_mean = (
        z_returns[negative_columns].mean(axis=1)
        if negative_columns
        else pd.Series(0.0, index=z_returns.index)
    )

    return positive_mean - negative_mean


def expanding_percentile_rank(series: pd.Series) -> pd.Series:
    """Rank each value against only the history available through that date."""
    observed: List[float] = []
    ranks: List[float] = []

    for value in series.to_numpy():
        observed.append(value)
        clean = pd.Series(observed, dtype=float).dropna()

        if pd.isna(value) or clean.empty:
            ranks.append(np.nan)
        else:
            ranks.append(float((clean <= value).mean()))

    return pd.Series(ranks, index=series.index)


def build_daily_scenario_scores(
    z_returns: pd.DataFrame,
) -> pd.DataFrame:
    output = pd.DataFrame(index=z_returns.index)

    for scenario_key, config in SCENARIOS.items():
        raw = build_scenario_score(
            z_returns,
            positive=config.get("positive", []),
            negative=config.get("negative", []),
        )

        output[f"{scenario_key}_score"] = raw
        output[f"{scenario_key}_score_ema{SMOOTH_SPAN}"] = raw.ewm(
            span=SMOOTH_SPAN,
            adjust=False,
        ).mean()
        output[f"{scenario_key}_pct_rank"] = expanding_percentile_rank(raw)

    return output


# =============================================================================
# Stock exposure estimation
# =============================================================================


def correlation_t_stat(correlation: float, observations: int) -> float:
    if observations <= 2 or not math.isfinite(correlation):
        return math.nan

    denominator = max(1.0 - correlation**2, 1e-12)
    return correlation * math.sqrt((observations - 2) / denominator)


def estimate_one_exposure(
    stock_returns: pd.Series,
    scenario_scores: pd.Series,
    min_observations: int,
) -> dict:
    aligned = pd.concat(
        [
            stock_returns.rename("stock_return"),
            scenario_scores.rename("scenario_score"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    observations = int(len(aligned))
    if observations < min_observations:
        return {
            "correlation": math.nan,
            "beta": math.nan,
            "t_stat": math.nan,
            "observations": observations,
            "exposure_score": math.nan,
            "vulnerability_score": math.nan,
        }

    stock = aligned["stock_return"]
    scenario = aligned["scenario_score"]

    correlation = float(stock.corr(scenario))
    scenario_variance = float(scenario.var(ddof=1))
    covariance = float(stock.cov(scenario))
    beta = covariance / scenario_variance if scenario_variance > 0 else math.nan

    exposure_score = (
        float(np.clip(50.0 * (correlation + 1.0), 0.0, 100.0))
        if math.isfinite(correlation)
        else math.nan
    )
    vulnerability_score = (
        100.0 - exposure_score if math.isfinite(exposure_score) else math.nan
    )

    return {
        "correlation": correlation,
        "beta": beta,
        "t_stat": correlation_t_stat(correlation, observations),
        "observations": observations,
        "exposure_score": exposure_score,
        "vulnerability_score": vulnerability_score,
    }


def build_universe_exposures(
    stock_returns: pd.DataFrame,
    daily_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []

    raw_score_columns = {
        key: f"{key}_score"
        for key in SCENARIOS
        if f"{key}_score" in daily_scores.columns
    }

    for ticker in stock_returns.columns:
        result: dict = {"ticker": ticker}

        for scenario_key, score_column in raw_score_columns.items():
            estimate = estimate_one_exposure(
                stock_returns[ticker],
                daily_scores[score_column],
                min_observations=MIN_CORRELATION_OBSERVATIONS,
            )

            result[f"{scenario_key}_correlation"] = estimate["correlation"]
            result[f"{scenario_key}_beta"] = estimate["beta"]
            result[f"{scenario_key}_t_stat"] = estimate["t_stat"]
            result[f"{scenario_key}_observations"] = estimate["observations"]
            result[f"{scenario_key}_exposure_score"] = estimate["exposure_score"]
            result[f"{scenario_key}_vulnerability_score"] = estimate[
                "vulnerability_score"
            ]

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    scenario_tickers = flatten_unique_tickers(SCENARIOS)
    universe_tickers = read_universe_tickers(UNIVERSE_PATH)

    print(f"Scenario ETF anchors ({len(scenario_tickers)}): {scenario_tickers}")
    print(f"Universe tickers: {len(universe_tickers)}")

    # -------------------------------------------------------------------------
    # Build daily scenario scores.
    # -------------------------------------------------------------------------
    anchor_prices = download_prices(scenario_tickers, WARMUP_START)
    if anchor_prices.empty:
        raise ValueError("No scenario anchor prices were downloaded.")

    anchor_returns = anchor_prices.pct_change(fill_method=None)

    anchor_zscores = pd.DataFrame(index=anchor_returns.index)
    for ticker in anchor_returns.columns:
        anchor_zscores[ticker] = rolling_zscore(
            anchor_returns[ticker],
            window=ROLLING_WINDOW,
            min_periods=MIN_PERIODS,
            clip_z=CLIP_Z,
        )

    daily_scores = build_daily_scenario_scores(anchor_zscores)

    period_start = pd.Timestamp(START_DATE)
    anchor_prices_out = anchor_prices.loc[anchor_prices.index >= period_start].copy()
    anchor_returns_out = anchor_returns.loc[anchor_returns.index >= period_start].copy()
    anchor_zscores_out = anchor_zscores.loc[anchor_zscores.index >= period_start].copy()
    daily_scores_out = daily_scores.loc[daily_scores.index >= period_start].copy()

    # -------------------------------------------------------------------------
    # Download universe returns and estimate each stock's scenario exposures.
    # -------------------------------------------------------------------------
    stock_prices = download_stock_prices_batched(
        universe_tickers,
        start_date=WARMUP_START,
        batch_size=STOCK_DOWNLOAD_BATCH_SIZE,
    )
    stock_returns = stock_prices.pct_change(fill_method=None)
    stock_returns = stock_returns.loc[stock_returns.index >= period_start]

    exposures = build_universe_exposures(stock_returns, daily_scores_out)

    # -------------------------------------------------------------------------
    # Save CSV outputs.
    # -------------------------------------------------------------------------
    anchor_prices_out.to_csv(
        DATA_DIR / "scenario_anchor_prices.csv",
        index_label="date",
    )
    anchor_returns_out.to_csv(
        DATA_DIR / "scenario_anchor_returns.csv",
        index_label="date",
    )
    anchor_zscores_out.to_csv(
        DATA_DIR / "scenario_anchor_zscores.csv",
        index_label="date",
    )

    daily_scores_csv = daily_scores_out.reset_index().rename(
        columns={"Date": "date", "index": "date"}
    )
    daily_scores_csv["date"] = pd.to_datetime(daily_scores_csv["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    daily_scores_csv.to_csv(
        DATA_DIR / "market_scenario_daily_scores.csv",
        index=False,
    )

    exposures.to_csv(
        DATA_DIR / "universe_scenario_exposures.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save frontend-friendly JSON outputs.
    # -------------------------------------------------------------------------
    daily_json = {
        "metadata": {
            "start_date": START_DATE,
            "warmup_start": WARMUP_START,
            "rolling_window": ROLLING_WINDOW,
            "min_periods": MIN_PERIODS,
            "clip_z": CLIP_Z,
            "smooth_span": SMOOTH_SPAN,
            "scenarios": {
                key: {
                    "label": config["label"],
                    "positive": config["positive"],
                    "negative": config["negative"],
                    "description": config["description"],
                }
                for key, config in SCENARIOS.items()
            },
        },
        "data": frame_to_json_records(daily_scores_csv),
    }

    with open(
        PUBLIC_DATA_DIR / "market_scenario_daily_scores.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(daily_json, file, indent=2)

    exposure_json = {
        "metadata": {
            "start_date": START_DATE,
            "minimum_observations": MIN_CORRELATION_OBSERVATIONS,
            "score_interpretation": {
                "exposure_score_0": "Strongly harmed as the scenario intensifies",
                "exposure_score_50": "Little or no linear relationship",
                "exposure_score_100": "Strongly benefits as the scenario intensifies",
                "vulnerability_score": "100 minus exposure score",
            },
            "scenarios": {
                key: config["label"] for key, config in SCENARIOS.items()
            },
        },
        "data": frame_to_json_records(exposures),
    }

    with open(
        PUBLIC_DATA_DIR / "universe_scenario_exposures.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(exposure_json, file, indent=2)

    print("Done.")
    print(f"Saved: {DATA_DIR / 'scenario_anchor_prices.csv'}")
    print(f"Saved: {DATA_DIR / 'scenario_anchor_returns.csv'}")
    print(f"Saved: {DATA_DIR / 'scenario_anchor_zscores.csv'}")
    print(f"Saved: {DATA_DIR / 'market_scenario_daily_scores.csv'}")
    print(f"Saved: {PUBLIC_DATA_DIR / 'market_scenario_daily_scores.json'}")
    print(f"Saved: {DATA_DIR / 'universe_scenario_exposures.csv'}")
    print(f"Saved: {PUBLIC_DATA_DIR / 'universe_scenario_exposures.json'}")


if __name__ == "__main__":
    main()
