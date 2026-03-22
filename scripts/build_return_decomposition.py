#!/usr/bin/env python3
"""
Build daily return decomposition for Majorah.

For each stock and each day:
    stock_return ~= alpha + market_component + cluster_component + idiosyncratic_component

Model:
    R_stock = alpha + beta_m * R_market + beta_c * R_cluster + residual

Where:
- R_market  = SPY daily return
- R_cluster = average return of cluster peers EXCLUDING the stock itself
- residual  = idiosyncratic / company-specific portion

Outputs
-------
- public/data/return_decomposition_summary.json
- public/data/return_decomposition_index.json
- public/data/return_decomposition_shards/shard_00.json ... shard_15.json

Notes
-----
- Uses rolling OLS with intercept
- Writes frontend-lean shard files only
- Summary contains all-stocks aggregate info
- Index maps ticker -> shard filename
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

    renamed = {"date": "date"}
    for c in df.columns:
        if c == "date":
            continue
        renamed[c] = normalize_ticker(c)
    df = df.rename(columns=renamed)

    value_cols = [c for c in df.columns if c != "date"]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    deduped_cols = []
    seen = set()
    for c in df.columns:
        if c not in seen:
            seen.add(c)
            deduped_cols.append(c)
    df = df[deduped_cols]

    return df


def _extract_cluster_id(row: dict, fallback: object = None) -> Optional[str]:
    for key in ("cluster", "cluster_id", "group", "label", "id", "name"):
        if key in row and row[key] is not None:
            return str(row[key])
    if fallback is not None:
        return str(fallback)
    return None


def _extract_members(row: dict) -> List[str]:
    for key in ("members", "tickers", "symbols", "stocks", "items", "constituents"):
        if key in row and isinstance(row[key], list):
            return [normalize_ticker(x) for x in row[key] if normalize_ticker(x)]
    return []


def parse_json_cluster_mapping(data: object) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if isinstance(data, dict):
        scalar_mapping_candidate = {}
        scalar_like = True
        for k, v in data.items():
            if isinstance(v, (str, int, float)) and normalize_ticker(k):
                scalar_mapping_candidate[normalize_ticker(k)] = str(v)
            else:
                scalar_like = False
                break
        if scalar_like and scalar_mapping_candidate:
            return scalar_mapping_candidate

        list_mapping_candidate = {}
        all_values_are_lists = len(data) > 0 and all(isinstance(v, list) for v in data.values())
        if all_values_are_lists:
            for cluster_id, members in data.items():
                for m in members:
                    t = normalize_ticker(m)
                    if t:
                        list_mapping_candidate[t] = str(cluster_id)
            if list_mapping_candidate:
                return list_mapping_candidate

        for key in ("clusters", "cluster_map", "mapping", "data", "results"):
            if key in data:
                try:
                    nested = parse_json_cluster_mapping(data[key])
                    if nested:
                        return nested
                except ValueError:
                    pass

        nested_ticker_mapping = {}
        ticker_style = True
        for k, v in data.items():
            if not isinstance(v, dict):
                ticker_style = False
                break
            cluster_id = _extract_cluster_id(v)
            t = normalize_ticker(k)
            if not t or cluster_id is None:
                ticker_style = False
                break
            nested_ticker_mapping[t] = cluster_id
        if ticker_style and nested_ticker_mapping:
            return nested_ticker_mapping

        cluster_object_mapping = {}
        cluster_object_style = True
        for k, v in data.items():
            if not isinstance(v, dict):
                cluster_object_style = False
                break
            members = _extract_members(v)
            if not members:
                cluster_object_style = False
                break
            cluster_id = _extract_cluster_id(v, fallback=k)
            for t in members:
                cluster_object_mapping[t] = cluster_id
        if cluster_object_style and cluster_object_mapping:
            return cluster_object_mapping

    if isinstance(data, list):
        row_mapping = {}
        row_style_ok = True
        saw_any_row_mapping = False

        for row in data:
            if not isinstance(row, dict):
                row_style_ok = False
                break

            ticker = row.get("ticker", row.get("symbol", row.get("stock", row.get("name"))))
            cluster_id = _extract_cluster_id(row)
            if ticker is not None and cluster_id is not None:
                t = normalize_ticker(ticker)
                if t:
                    row_mapping[t] = cluster_id
                    saw_any_row_mapping = True
                    continue

        if row_style_ok and saw_any_row_mapping and row_mapping:
            return row_mapping

        cluster_list_mapping = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            cluster_id = _extract_cluster_id(row)
            members = _extract_members(row)
            if cluster_id is None or not members:
                continue
            for t in members:
                cluster_list_mapping[t] = cluster_id
        if cluster_list_mapping:
            return cluster_list_mapping

    raise ValueError("Unsupported JSON cluster format.")


def read_cluster_mapping(path: Path) -> Dict[str, str]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = parse_json_cluster_mapping(data)
        if not mapping:
            raise ValueError(f"No valid cluster mapping found in {path}")
        return mapping

    if suffix == ".csv":
        df = pd.read_csv(path)
        lower_map = {str(c).strip().lower(): c for c in df.columns}

        ticker_col = None
        for candidate in ("ticker", "symbol", "stock", "name"):
            if candidate in lower_map:
                ticker_col = lower_map[candidate]
                break

        cluster_col = None
        for candidate in ("cluster", "cluster_id", "group", "label"):
            if candidate in lower_map:
                cluster_col = lower_map[candidate]
                break

        members_col = None
        for candidate in ("members", "tickers", "symbols", "stocks", "items", "constituents"):
            if candidate in lower_map:
                members_col = lower_map[candidate]
                break

        if ticker_col is not None and cluster_col is not None:
            mapping = {}
            for _, row in df.iterrows():
                ticker = normalize_ticker(row[ticker_col])
                cluster_id = row[cluster_col]
                if ticker and pd.notna(cluster_id):
                    mapping[ticker] = str(cluster_id)
            if mapping:
                return mapping

        if cluster_col is not None and members_col is not None:
            mapping = {}
            for _, row in df.iterrows():
                cluster_id = row[cluster_col]
                if pd.isna(cluster_id):
                    continue
                raw_members = str(row[members_col] or "")
                split_members = [
                    normalize_ticker(x)
                    for x in raw_members.replace("|", ",").replace(";", ",").split(",")
                ]
                split_members = [x for x in split_members if x]
                for t in split_members:
                    mapping[t] = str(cluster_id)
            if mapping:
                return mapping

        raise ValueError(
            f"CSV cluster file {path} must contain either "
            f"(ticker/symbol + cluster/cluster_id) or (cluster/cluster_id + members/tickers)."
        )

    raise ValueError(f"Unsupported cluster file type: {path.suffix}")


def build_cluster_to_members(ticker_to_cluster: Dict[str, str]) -> Dict[str, List[str]]:
    cluster_to_members: Dict[str, List[str]] = {}
    for ticker, cluster_id in ticker_to_cluster.items():
        cluster_to_members.setdefault(str(cluster_id), []).append(ticker)
    return cluster_to_members


def rolling_ols_betas(
    y: np.ndarray,
    x_market: np.ndarray,
    x_cluster: np.ndarray,
    window: int,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    alpha = np.full(n, np.nan, dtype=float)
    beta_m = np.full(n, np.nan, dtype=float)
    beta_c = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - window + 1)

        ys = y[start : i + 1]
        xm = x_market[start : i + 1]
        xc = x_cluster[start : i + 1]

        mask = np.isfinite(ys) & np.isfinite(xm) & np.isfinite(xc)
        if mask.sum() < min_obs:
            continue

        Y = ys[mask]
        X = np.column_stack([np.ones(mask.sum()), xm[mask], xc[mask]])

        try:
            coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha[i] = coef[0]
        beta_m[i] = coef[1]
        beta_c[i] = coef[2]

    return alpha, beta_m, beta_c


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


def rolling_zscore(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def classify_dominant_driver(row: pd.Series) -> str:
    vals = {
        "market": abs(row.get("market_component", np.nan)),
        "cluster": abs(row.get("cluster_component", np.nan)),
        "idiosyncratic": abs(row.get("idiosyncratic_component", np.nan)),
    }
    finite_vals = {k: v for k, v in vals.items() if pd.notna(v)}
    if not finite_vals:
        return "unknown"
    return max(finite_vals, key=finite_vals.get)


def summarize_ticker(df_ticker: pd.DataFrame) -> Dict[str, object]:
    valid = df_ticker.dropna(
        subset=["return", "market_component", "cluster_component", "idiosyncratic_component"]
    )
    if valid.empty:
        return {
            "ticker": df_ticker["ticker"].iloc[0],
            "cluster_id": df_ticker["cluster_id"].iloc[0] if "cluster_id" in df_ticker.columns else None,
            "observations": 0,
            "mean_abs_return": None,
            "mean_abs_market_component": None,
            "mean_abs_cluster_component": None,
            "mean_abs_idiosyncratic_component": None,
            "pct_abs_explained_by_market": None,
            "pct_abs_explained_by_cluster": None,
            "pct_abs_explained_idiosyncratic": None,
            "mean_beta_market": None,
            "mean_beta_cluster": None,
            "idio_event_rate": None,
            "cluster_event_rate": None,
            "market_event_rate": None,
            "dominant_market_rate": None,
            "dominant_cluster_rate": None,
            "dominant_idio_rate": None,
            "r2_proxy": None,
        }

    abs_ret = valid["return"].abs().sum()
    abs_market = valid["market_component"].abs().sum()
    abs_cluster = valid["cluster_component"].abs().sum()
    abs_idio = valid["idiosyncratic_component"].abs().sum()

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

    def mean_bool(col: str) -> Optional[float]:
        if col not in valid.columns:
            return None
        s = pd.to_numeric(valid[col], errors="coerce")
        if s.dropna().empty:
            return None
        return float(s.mean())

    dominant_counts = valid["dominant_driver"].value_counts(normalize=True) if "dominant_driver" in valid.columns else {}

    return {
        "ticker": valid["ticker"].iloc[0],
        "cluster_id": valid["cluster_id"].iloc[0] if "cluster_id" in valid.columns else None,
        "observations": int(len(valid)),
        "mean_abs_return": round(float(valid["return"].abs().mean()), 6),
        "mean_abs_market_component": round(float(valid["market_component"].abs().mean()), 6),
        "mean_abs_cluster_component": round(float(valid["cluster_component"].abs().mean()), 6),
        "mean_abs_idiosyncratic_component": round(float(valid["idiosyncratic_component"].abs().mean()), 6),
        "pct_abs_explained_by_market": ratio(abs_market, abs_ret),
        "pct_abs_explained_by_cluster": ratio(abs_cluster, abs_ret),
        "pct_abs_explained_idiosyncratic": ratio(abs_idio, abs_ret),
        "mean_beta_market": round(float(valid["beta_market"].mean()), 6) if "beta_market" in valid.columns else None,
        "mean_beta_cluster": round(float(valid["beta_cluster"].mean()), 6) if "beta_cluster" in valid.columns else None,
        "idio_event_rate": mean_bool("is_idio_event"),
        "cluster_event_rate": mean_bool("is_cluster_event"),
        "market_event_rate": mean_bool("is_market_event"),
        "dominant_market_rate": float(dominant_counts.get("market", 0.0)),
        "dominant_cluster_rate": float(dominant_counts.get("cluster", 0.0)),
        "dominant_idio_rate": float(dominant_counts.get("idiosyncratic", 0.0)),
        "r2_proxy": r2_proxy,
    }


def assign_shard(ticker: str, num_shards: int) -> str:
    h = hashlib.md5(ticker.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % num_shards
    return f"shard_{idx:02d}.json"


def round_or_none(v: object, ndigits: int = 6):
    try:
        num = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(num):
        return None
    return round(num, ndigits)


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

    rows: List[pd.DataFrame] = []
    eligible_tickers = [c for c in returns_wide.columns if c != market_ticker and c in ticker_to_cluster]

    if not eligible_tickers:
        raise ValueError("No eligible tickers found that are both in daily_returns.csv and cluster mapping.")

    print(f"Eligible tickers: {len(eligible_tickers)}")

    for idx, ticker in enumerate(sorted(eligible_tickers), start=1):
        cluster_id = ticker_to_cluster.get(ticker)
        cluster_series = build_cluster_return_series(
            returns_wide=returns_wide,
            ticker=ticker,
            ticker_to_cluster=ticker_to_cluster,
            cluster_to_members=cluster_to_members,
        )
        stock_series = returns_wide[ticker]

        y = stock_series.values.astype(float)
        x_market = market_series.values.astype(float)
        x_cluster = cluster_series.values.astype(float)

        alpha, beta_m, beta_c = rolling_ols_betas(
            y=y,
            x_market=x_market,
            x_cluster=x_cluster,
            window=args.window,
            min_obs=args.min_obs,
        )

        alpha_component = alpha
        market_component = beta_m * x_market
        cluster_component = beta_c * x_cluster
        explained_component = alpha_component + market_component + cluster_component
        idiosyncratic_component = y - explained_component

        valid_mask = (
            np.isfinite(y)
            & np.isfinite(alpha_component)
            & np.isfinite(beta_m)
            & np.isfinite(beta_c)
            & np.isfinite(x_market)
            & np.isfinite(x_cluster)
        )

        df_ticker = pd.DataFrame({
            "date": returns_wide.index,
            "ticker": ticker,
            "cluster_id": cluster_id,
            "return": y,
            "beta_market": beta_m,
            "beta_cluster": beta_c,
            "market_component": market_component,
            "cluster_component": cluster_component,
            "idiosyncratic_component": idiosyncratic_component,
            "valid_decomposition": valid_mask.astype(int),
        })

        df_ticker["idio_zscore"] = rolling_zscore(
            df_ticker["idiosyncratic_component"],
            window=args.event_window,
            min_periods=max(20, min(args.event_window, args.min_obs)),
        )
        df_ticker["cluster_zscore"] = rolling_zscore(
            df_ticker["cluster_component"],
            window=args.event_window,
            min_periods=max(20, min(args.event_window, args.min_obs)),
        )
        df_ticker["market_zscore"] = rolling_zscore(
            df_ticker["market_component"],
            window=args.event_window,
            min_periods=max(20, min(args.event_window, args.min_obs)),
        )

        threshold = float(args.event_z_threshold)
        df_ticker["is_idio_event"] = (df_ticker["idio_zscore"].abs() >= threshold).astype(float)
        df_ticker["is_cluster_event"] = (df_ticker["cluster_zscore"].abs() >= threshold).astype(float)
        df_ticker["is_market_event"] = (df_ticker["market_zscore"].abs() >= threshold).astype(float)

        invalid_rows = df_ticker["valid_decomposition"] != 1
        for col in [
            "idio_zscore",
            "cluster_zscore",
            "market_zscore",
            "is_idio_event",
            "is_cluster_event",
            "is_market_event",
        ]:
            df_ticker.loc[invalid_rows, col] = np.nan

        df_ticker["dominant_driver"] = df_ticker.apply(classify_dominant_driver, axis=1)

        rows.append(df_ticker)

        if idx % 25 == 0 or idx == len(eligible_tickers):
            print(f"Processed {idx}/{len(eligible_tickers)} tickers")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # summary
    summary_rows = []
    for ticker, grp in out.groupby("ticker", sort=True):
        summary_rows.append(summarize_ticker(grp))
    summary_df = pd.DataFrame(summary_rows).sort_values("ticker").reset_index(drop=True)

    ensure_parent_dir(OUTPUT_SUMMARY_JSON)
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, ensure_ascii=False)
    print(f"Wrote summary JSON: {OUTPUT_SUMMARY_JSON}")

    # shard output
    ensure_dir(OUTPUT_SHARDS_DIR)

    # clean old shards
    for old_file in OUTPUT_SHARDS_DIR.glob("shard_*.json"):
        old_file.unlink()

    index_map: Dict[str, str] = {}
    shards: Dict[str, Dict[str, List[dict]]] = {}

    for ticker, grp in out.groupby("ticker", sort=True):
        grp = grp.sort_values("date").tail(args.max_days).copy()
        shard_name = assign_shard(ticker, args.num_shards)
        index_map[ticker] = shard_name
        shards.setdefault(shard_name, {})

        grp["date"] = pd.to_datetime(grp["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        frontend_rows = []
        for _, row in grp.iterrows():
            frontend_rows.append({
                "date": row["date"],
                "return": round_or_none(row["return"]),
                "market_component": round_or_none(row["market_component"]),
                "cluster_component": round_or_none(row["cluster_component"]),
                "idiosyncratic_component": round_or_none(row["idiosyncratic_component"]),
                "dominant_driver": row["dominant_driver"] if pd.notna(row["dominant_driver"]) else None,
                "is_idio_event": int(row["is_idio_event"]) if pd.notna(row["is_idio_event"]) else None,
            })

        shards[shard_name][ticker] = frontend_rows

    with open(OUTPUT_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(index_map, f, ensure_ascii=False)
    print(f"Wrote index JSON: {OUTPUT_INDEX_JSON}")

    for shard_name, shard_payload in sorted(shards.items()):
        out_path = OUTPUT_SHARDS_DIR / shard_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(shard_payload, f, ensure_ascii=False)
        print(f"Wrote shard: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
