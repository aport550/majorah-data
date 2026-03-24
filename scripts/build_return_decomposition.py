#!/usr/bin/env python3
"""
Build additive daily return decomposition for Majorah.

Goal
----
For each stock and each day, produce:

    stock_return = macro_component + cluster_component + idiosyncratic_component

Interpretation
--------------
- macro_component:
    portion of the stock's return explained by the broad market factor
- cluster_component:
    portion explained by cluster/peer moves AFTER removing what those peer moves
    were already doing because of the broad market
- idiosyncratic_component:
    everything left over, including stock-specific effects and intercept drift

Model
-----
Step 1: build raw peer factor
    cluster_raw_t = mean(peer returns excluding the stock itself)

Step 2: orthogonalize cluster factor to market using a rolling regression
    cluster_raw_t = a_t + g_t * market_t + cluster_residual_t

Step 3: regress stock return on market and orthogonalized cluster factor
    stock_t = alpha_t + beta_m_t * market_t + beta_c_t * cluster_residual_t + eps_t

Displayed additive decomposition
--------------------------------
We define:

    macro_component_t = beta_m_t * market_t
    cluster_component_t = beta_c_t * cluster_residual_t
    idiosyncratic_component_t = stock_t - macro_component_t - cluster_component_t

So the 3 displayed components sum EXACTLY to stock return each day.

Outputs
-------
- public/data/return_decomposition_summary.json
- public/data/return_decomposition_index.json
- public/data/return_decomposition_shards/shard_00.json ... shard_15.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve()

DEFAULT_DAILY_RETURNS_CANDIDATES = [
    ROOT / "data" / "daily_returns.csv",
    ROOT / "public" / "data" / "daily_returns.csv",
]

DEFAULT_CLUSTER_CANDIDATES = [
    ROOT / "data" / "stock_clusters.json",
    ROOT / "data" / "clusters.json",
    ROOT / "public" / "data" / "stock_clusters.json",
    ROOT / "public" / "data" / "clusters.json",
    ROOT / "data" / "stock_clusters.csv",
    ROOT / "data" / "clusters.csv",
    ROOT / "public" / "data" / "stock_clusters.csv",
    ROOT / "public" / "data" / "clusters.csv",
]

OUTPUT_SUMMARY_JSON = ROOT / "public" / "data" / "return_decomposition_summary.json"
OUTPUT_INDEX_JSON = ROOT / "public" / "data" / "return_decomposition_index.json"
OUTPUT_SHARDS_DIR = ROOT / "public" / "data" / "return_decomposition_shards"


def normalize_ticker(value: object) -> str:
    s = str(value or "").strip().upper()
    return "".join(ch for ch in s if ch.isalnum() or ch in {".", "-", "_"})


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def round_or_none(v: object, ndigits: int = 6):
    try:
        num = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return round(num, ndigits)


def read_daily_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    cols = list(df.columns)
    if cols and str(cols[0]).lower().startswith("unnamed"):
        df = df.rename(columns={cols[0]: "date"})

    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in {"date", "datetime", "day"}:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)

    value_cols = [c for c in df.columns if c != "date"]
    renamed = {}
    for c in value_cols:
        renamed[c] = normalize_ticker(c)
    df = df.rename(columns=renamed)

    for c in df.columns:
        if c == "date":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def read_cluster_mapping(path: Path) -> Dict[str, str]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(path.read_text())

        ticker_to_cluster: Dict[str, str] = {}

        if isinstance(payload, dict):
            if "clusters" in payload and isinstance(payload["clusters"], list):
                # format: {"clusters": [{"cluster_id": "...", "tickers": [...]}, ...]}
                for item in payload["clusters"]:
                    cluster_id = str(
                        item.get("cluster_id")
                        or item.get("cluster")
                        or item.get("id")
                        or ""
                    ).strip()
                    tickers = item.get("tickers") or item.get("members") or []
                    for t in tickers:
                        nt = normalize_ticker(t)
                        if nt and cluster_id:
                            ticker_to_cluster[nt] = cluster_id
            else:
                # either {ticker: cluster_id} or {cluster_id: [tickers]}
                values_are_lists = any(isinstance(v, list) for v in payload.values())
                if values_are_lists:
                    for cluster_id, tickers in payload.items():
                        for t in tickers:
                            nt = normalize_ticker(t)
                            if nt:
                                ticker_to_cluster[nt] = str(cluster_id)
                else:
                    for t, cluster_id in payload.items():
                        nt = normalize_ticker(t)
                        cid = str(cluster_id).strip()
                        if nt and cid:
                            ticker_to_cluster[nt] = cid

        return ticker_to_cluster

    # csv path
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    ticker_col = None
    cluster_col = None

    for c in df.columns:
        if c in {"ticker", "symbol", "stock"}:
            ticker_col = c
        if c in {"cluster", "cluster_id", "group"}:
            cluster_col = c

    if ticker_col is None or cluster_col is None:
        raise ValueError(f"Cluster CSV {path} must contain ticker/symbol and cluster/cluster_id columns.")

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        t = normalize_ticker(row.get(ticker_col))
        cid = str(row.get(cluster_col) or "").strip()
        if t and cid:
            out[t] = cid
    return out


def build_cluster_to_members(ticker_to_cluster: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ticker, cluster_id in ticker_to_cluster.items():
        out.setdefault(cluster_id, []).append(ticker)
    return out


def build_cluster_return_series(
    returns_wide: pd.DataFrame,
    ticker: str,
    ticker_to_cluster: Dict[str, str],
    cluster_to_members: Dict[str, List[str]],
) -> pd.Series:
    cluster_id = ticker_to_cluster.get(ticker)
    if cluster_id is None:
        return pd.Series(np.nan, index=returns_wide.index, dtype=float)

    members = cluster_to_members.get(cluster_id, [])
    peers = [m for m in members if m != ticker and m in returns_wide.columns]

    if not peers:
        return pd.Series(np.nan, index=returns_wide.index, dtype=float)

    return returns_wide[peers].mean(axis=1, skipna=True)


def rolling_ols_1factor(
    y: np.ndarray,
    x: np.ndarray,
    window: int,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling OLS:
        y = alpha + beta * x + residual
    Returns alpha, beta arrays.
    """
    n = len(y)
    alpha = np.full(n, np.nan, dtype=float)
    beta = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)

        ys = y[start : i + 1]
        xs = x[start : i + 1]

        mask = np.isfinite(ys) & np.isfinite(xs)
        if mask.sum() < min_obs:
            continue

        Y = ys[mask]
        X = np.column_stack([np.ones(mask.sum()), xs[mask]])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha[i] = coef[0]
        beta[i] = coef[1]

    return alpha, beta


def rolling_ols_2factor(
    y: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    window: int,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rolling OLS:
        y = alpha + b1 * x1 + b2 * x2 + residual
    Returns alpha, b1, b2 arrays.
    """
    n = len(y)
    alpha = np.full(n, np.nan, dtype=float)
    b1 = np.full(n, np.nan, dtype=float)
    b2 = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)

        ys = y[start : i + 1]
        x1s = x1[start : i + 1]
        x2s = x2[start : i + 1]

        mask = np.isfinite(ys) & np.isfinite(x1s) & np.isfinite(x2s)
        if mask.sum() < min_obs:
            continue

        Y = ys[mask]
        X = np.column_stack([np.ones(mask.sum()), x1s[mask], x2s[mask]])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha[i] = coef[0]
        b1[i] = coef[1]
        b2[i] = coef[2]

    return alpha, b1, b2


def rolling_zscore(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def classify_dominant_driver(row: pd.Series) -> str:
    vals = {
        "macro": abs(row.get("macro_component", np.nan)),
        "cluster": abs(row.get("cluster_component", np.nan)),
        "idiosyncratic": abs(row.get("idiosyncratic_component", np.nan)),
    }
    finite_vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not finite_vals:
        return "unknown"
    return max(finite_vals, key=finite_vals.get)


def summarize_ticker(df_ticker: pd.DataFrame) -> Dict[str, object]:
    valid = df_ticker.dropna(
        subset=["return", "macro_component", "cluster_component", "idiosyncratic_component"]
    )
    ticker = df_ticker["ticker"].iloc[0]
    cluster_id = df_ticker["cluster_id"].iloc[0] if "cluster_id" in df_ticker.columns else None

    if valid.empty:
        return {
            "ticker": ticker,
            "cluster_id": cluster_id,
            "observations": 0,
            "mean_abs_return": None,
            "mean_abs_macro_component": None,
            "mean_abs_cluster_component": None,
            "mean_abs_idiosyncratic_component": None,
            "pct_abs_explained_by_macro": None,
            "pct_abs_explained_by_cluster": None,
            "pct_abs_explained_by_idiosyncratic": None,
            "mean_beta_macro": None,
            "mean_beta_cluster": None,
            "r2_proxy": None,
            "dominant_macro_rate": None,
            "dominant_cluster_rate": None,
            "dominant_idiosyncratic_rate": None,
            "idio_event_rate": None,
            "cluster_event_rate": None,
            "macro_event_rate": None,
        }

    abs_ret = float(valid["return"].abs().sum())
    abs_macro = float(valid["macro_component"].abs().sum())
    abs_cluster = float(valid["cluster_component"].abs().sum())
    abs_idio = float(valid["idiosyncratic_component"].abs().sum())

    y = valid["return"].values
    residual = valid["idiosyncratic_component"].values
    y_var = float(np.nanvar(y, ddof=1)) if len(y) > 1 else np.nan
    resid_var = float(np.nanvar(residual, ddof=1)) if len(residual) > 1 else np.nan
    r2_proxy = None
    if np.isfinite(y_var) and y_var > 0 and np.isfinite(resid_var):
        r2_proxy = max(0.0, min(1.0, 1.0 - resid_var / y_var))

    def ratio(part: float, total: float) -> Optional[float]:
        if not np.isfinite(total) or total <= 0:
            return None
        return float(part / total)

    dominant_counts = (
        valid["dominant_driver"].value_counts(normalize=True)
        if "dominant_driver" in valid.columns
        else {}
    )

    def mean_col(name: str) -> Optional[float]:
        if name not in valid.columns:
            return None
        s = pd.to_numeric(valid[name], errors="coerce")
        if s.dropna().empty:
            return None
        return float(s.mean())

    return {
        "ticker": ticker,
        "cluster_id": cluster_id,
        "observations": int(len(valid)),
        "mean_abs_return": round(float(valid["return"].abs().mean()), 6),
        "mean_abs_macro_component": round(float(valid["macro_component"].abs().mean()), 6),
        "mean_abs_cluster_component": round(float(valid["cluster_component"].abs().mean()), 6),
        "mean_abs_idiosyncratic_component": round(float(valid["idiosyncratic_component"].abs().mean()), 6),
        "pct_abs_explained_by_macro": round_or_none(ratio(abs_macro, abs_ret)),
        "pct_abs_explained_by_cluster": round_or_none(ratio(abs_cluster, abs_ret)),
        "pct_abs_explained_by_idiosyncratic": round_or_none(ratio(abs_idio, abs_ret)),
        "mean_beta_macro": round_or_none(mean_col("beta_macro")),
        "mean_beta_cluster": round_or_none(mean_col("beta_cluster")),
        "r2_proxy": round_or_none(r2_proxy),
        "dominant_macro_rate": round_or_none(float(dominant_counts.get("macro", np.nan))) if len(dominant_counts) else None,
        "dominant_cluster_rate": round_or_none(float(dominant_counts.get("cluster", np.nan))) if len(dominant_counts) else None,
        "dominant_idiosyncratic_rate": round_or_none(float(dominant_counts.get("idiosyncratic", np.nan))) if len(dominant_counts) else None,
        "idio_event_rate": round_or_none(mean_col("is_idio_event")),
        "cluster_event_rate": round_or_none(mean_col("is_cluster_event")),
        "macro_event_rate": round_or_none(mean_col("is_macro_event")),
    }


def shard_filename_for_ticker(ticker: str, num_shards: int) -> str:
    h = hashlib.md5(ticker.encode("utf-8")).hexdigest()
    shard_num = int(h, 16) % num_shards
    return f"shard_{shard_num:02d}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-returns", type=str, default=None)
    parser.add_argument("--clusters", type=str, default=None)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--min-obs", type=int, default=40)
    parser.add_argument("--market", type=str, default="SPY")
    parser.add_argument("--event-window", type=int, default=60)
    parser.add_argument("--event-z-threshold", type=float, default=2.0)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--max-days", type=int, default=252)
    args = parser.parse_args()

    daily_returns_path = Path(args.daily_returns) if args.daily_returns else find_first_existing(DEFAULT_DAILY_RETURNS_CANDIDATES)
    if daily_returns_path is None:
        raise FileNotFoundError("Could not find daily_returns.csv")

    cluster_path = Path(args.clusters) if args.clusters else find_first_existing(DEFAULT_CLUSTER_CANDIDATES)
    if cluster_path is None:
        raise FileNotFoundError("Could not find cluster file")

    print(f"Reading daily returns: {daily_returns_path}")
    returns_df = read_daily_returns(daily_returns_path)

    market_ticker = normalize_ticker(args.market)
    if market_ticker not in returns_df.columns:
        raise ValueError(f"Market ticker '{market_ticker}' not found in daily_returns.csv")

    print(f"Reading clusters: {cluster_path}")
    ticker_to_cluster = read_cluster_mapping(cluster_path)
    cluster_to_members = build_cluster_to_members(ticker_to_cluster)

    returns_wide = returns_df.set_index("date").sort_index()
    market_series = returns_wide[market_ticker]

    eligible_tickers = [c for c in returns_wide.columns if c != market_ticker and c in ticker_to_cluster]
    if not eligible_tickers:
        raise ValueError("No eligible tickers found that are both in daily_returns.csv and cluster mapping.")

    print(f"Eligible tickers: {len(eligible_tickers)}")

    per_ticker_frames: List[pd.DataFrame] = []

    for idx, ticker in enumerate(sorted(eligible_tickers), start=1):
        cluster_id = ticker_to_cluster.get(ticker)

        stock_series = returns_wide[ticker]
        cluster_raw_series = build_cluster_return_series(
            returns_wide=returns_wide,
            ticker=ticker,
            ticker_to_cluster=ticker_to_cluster,
            cluster_to_members=cluster_to_members,
        )

        y = stock_series.values.astype(float)
        x_market = market_series.values.astype(float)
        x_cluster_raw = cluster_raw_series.values.astype(float)

        # Step 1: orthogonalize cluster factor to market
        # cluster_raw = a + g * market + cluster_residual
        cluster_alpha, cluster_gamma = rolling_ols_1factor(
            y=x_cluster_raw,
            x=x_market,
            window=args.window,
            min_obs=args.min_obs,
        )
        x_cluster_resid = x_cluster_raw - (cluster_alpha + cluster_gamma * x_market)

        # Step 2: stock regression on market + orthogonalized cluster factor
        reg_alpha, beta_macro, beta_cluster = rolling_ols_2factor(
            y=y,
            x1=x_market,
            x2=x_cluster_resid,
            window=args.window,
            min_obs=args.min_obs,
        )

        macro_component = beta_macro * x_market
        cluster_component = beta_cluster * x_cluster_resid

        # Fold alpha into idiosyncratic so 3 displayed components sum exactly
        idiosyncratic_component = y - macro_component - cluster_component

        explained_no_idio = macro_component + cluster_component

        valid_mask = (
            np.isfinite(y)
            & np.isfinite(x_market)
            & np.isfinite(x_cluster_raw)
            & np.isfinite(x_cluster_resid)
            & np.isfinite(beta_macro)
            & np.isfinite(beta_cluster)
            & np.isfinite(macro_component)
            & np.isfinite(cluster_component)
            & np.isfinite(idiosyncratic_component)
        )

        df_ticker = pd.DataFrame(
            {
                "date": returns_wide.index,
                "ticker": ticker,
                "cluster_id": cluster_id,
                "return": y,
                "market_return": x_market,
                "cluster_raw_return": x_cluster_raw,
                "cluster_orthogonal_return": x_cluster_resid,
                "cluster_market_alpha": cluster_alpha,
                "cluster_market_beta": cluster_gamma,
                "regression_alpha": reg_alpha,
                "beta_macro": beta_macro,
                "beta_cluster": beta_cluster,
                "macro_component": macro_component,
                "cluster_component": cluster_component,
                "idiosyncratic_component": idiosyncratic_component,
                "explained_component_ex_idio": explained_no_idio,
                "reconstruction_error": y - (
                    macro_component + cluster_component + idiosyncratic_component
                ),
                "valid_decomposition": valid_mask.astype(int),
            }
        )

        min_periods = max(20, min(args.event_window, args.min_obs))

        df_ticker["idio_zscore"] = rolling_zscore(
            df_ticker["idiosyncratic_component"],
            window=args.event_window,
            min_periods=min_periods,
        )
        df_ticker["cluster_zscore"] = rolling_zscore(
            df_ticker["cluster_component"],
            window=args.event_window,
            min_periods=min_periods,
        )
        df_ticker["macro_zscore"] = rolling_zscore(
            df_ticker["macro_component"],
            window=args.event_window,
            min_periods=min_periods,
        )

        threshold = float(args.event_z_threshold)
        df_ticker["is_idio_event"] = (df_ticker["idio_zscore"].abs() >= threshold).astype(float)
        df_ticker["is_cluster_event"] = (df_ticker["cluster_zscore"].abs() >= threshold).astype(float)
        df_ticker["is_macro_event"] = (df_ticker["macro_zscore"].abs() >= threshold).astype(float)

        invalid_rows = df_ticker["valid_decomposition"] != 1
        for col in [
            "idio_zscore",
            "cluster_zscore",
            "macro_zscore",
            "is_idio_event",
            "is_cluster_event",
            "is_macro_event",
        ]:
            df_ticker.loc[invalid_rows, col] = np.nan

        df_ticker["dominant_driver"] = df_ticker.apply(classify_dominant_driver, axis=1)
        df_ticker.loc[invalid_rows, "dominant_driver"] = "unknown"

        # keep recent days if requested
        if args.max_days and args.max_days > 0:
            df_ticker = df_ticker.tail(args.max_days).copy()

        per_ticker_frames.append(df_ticker)

        if idx % 100 == 0 or idx == len(eligible_tickers):
            print(f"Processed {idx}/{len(eligible_tickers)}")

    all_df = pd.concat(per_ticker_frames, ignore_index=True)

    # summary
    summary_rows = []
    for ticker, grp in all_df.groupby("ticker", sort=True):
        summary_rows.append(summarize_ticker(grp))
    summary_rows = sorted(summary_rows, key=lambda x: x["ticker"])

    summary_payload = {
        "model": {
            "market_factor": market_ticker,
            "cluster_factor": "peer cluster average excluding stock, orthogonalized to market",
            "display_identity": "return = macro_component + cluster_component + idiosyncratic_component",
            "window": args.window,
            "min_obs": args.min_obs,
            "event_window": args.event_window,
            "event_z_threshold": args.event_z_threshold,
            "max_days": args.max_days,
        },
        "tickers": summary_rows,
    }

    ensure_parent_dir(OUTPUT_SUMMARY_JSON)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")

    # shard outputs
    ensure_dir(OUTPUT_SHARDS_DIR)
    shard_map: Dict[str, List[dict]] = {f"shard_{i:02d}.json": [] for i in range(args.num_shards)}
    index_payload: Dict[str, str] = {}

    frontend_cols = [
        "date",
        "ticker",
        "cluster_id",
        "return",
        "market_return",
        "cluster_raw_return",
        "cluster_orthogonal_return",
        "beta_macro",
        "beta_cluster",
        "macro_component",
        "cluster_component",
        "idiosyncratic_component",
        "regression_alpha",
        "dominant_driver",
        "idio_zscore",
        "cluster_zscore",
        "macro_zscore",
        "is_idio_event",
        "is_cluster_event",
        "is_macro_event",
        "valid_decomposition",
        "reconstruction_error",
    ]

    all_df = all_df.copy()
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.strftime("%Y-%m-%d")

    for ticker, grp in all_df.groupby("ticker", sort=True):
        shard_name = shard_filename_for_ticker(ticker, args.num_shards)
        index_payload[ticker] = shard_name

        records = []
        for _, row in grp[frontend_cols].iterrows():
            rec = {}
            for col in frontend_cols:
                val = row[col]
                if isinstance(val, str):
                    rec[col] = val
                elif pd.isna(val):
                    rec[col] = None
                else:
                    rec[col] = round_or_none(val)
            records.append(rec)

        shard_map[shard_name].append(
            {
                "ticker": ticker,
                "rows": records,
            }
        )

    for shard_name, payload in shard_map.items():
        shard_path = OUTPUT_SHARDS_DIR / shard_name
        shard_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {shard_path}")

    ensure_parent_dir(OUTPUT_INDEX_JSON)
    OUTPUT_INDEX_JSON.write_text(json.dumps(index_payload, indent=2))
    print(f"Wrote {OUTPUT_INDEX_JSON}")


if __name__ == "__main__":
    main()
