#!/usr/bin/env python3
"""Majorah: five daily ETF proxies, four main regimes, 32 subregimes.

Install: python -m pip install pandas numpy yfinance
Save in your project's scripts/ folder and run from any directory.
Alternatively pass --root /path/to/project. Python 3.10+.

All inputs are adjusted ETF daily returns. Labels describe daily market
direction, NOT economic levels or measured changes in inflation/real yields.
Positive scores: inflation rising, growth/risk-on, dollar strengthening,
credit improving, real rates rising. Scores preserve the raw signal's sign.

Duration assumptions are fixed approximate model parameters, not historical
fund durations. Income/carry, CPI accrual, curve shifts, ETF premiums/discounts,
and changing durations contaminate these proxies. TIP appears in two signals;
the five dimensions need not be independent. Validate against yield/spread
data before interpreting these as macroeconomic measurements.

Existing JSON remains a list of daily rows. Liquidity is replaced by dollar,
credit, and real_rates: update frontend consumers accordingly. Additional
catalog and metadata JSON files document all 32 IDs and the assumptions.
Use completed daily labels only for subsequent-return predictive backtests.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right, insort_right
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

START_DATE = "2020-01-01"
WARMUP_START = "2019-06-01"
ROLLING_WINDOW = 60
MIN_PERIODS = 20
CLIP_Z = 3.0
SMOOTH_SPAN = 5
MAX_ATTEMPTS = 3
TICKERS = ["SPY", "TIP", "IEF", "UUP", "HYG"]
DIMENSIONS = ["inflation", "growth", "dollar", "credit", "real_rates"]
# Illustrative fixed assumptions in YEARS, not claimed current fund data.
DURATION = {"TIP": 6.5, "IEF": 7.0, "HYG": 3.0}
FLAT_EPSILON = 1e-12   # numerical equality only, not a noise threshold
STATES = {
    "inflation": ("Expectations falling", "Expectations rising"),
    "growth": ("Contraction / risk-off", "Growth / risk-on"),
    "dollar": ("Dollar weakening", "Dollar strengthening"),
    "credit": ("Credit deteriorating", "Credit improving"),
    "real_rates": ("Real rates falling", "Real rates rising"),
}


def extract_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=tickers, dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        levels = [i for i in range(df.columns.nlevels)
                  if "Close" in df.columns.get_level_values(i)]
        if not levels:
            raise ValueError("Download has no Close field")
        prices = df.xs("Close", axis=1, level=levels[0]).copy()
    else:
        if len(tickers) != 1 or "Close" not in df:
            raise ValueError("Unexpected download column structure")
        prices = df["Close"].to_frame(tickers[0])
    prices.columns = [str(c).upper() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    return prices.reindex(columns=tickers).apply(pd.to_numeric, errors="coerce")


def download_prices(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    prices = pd.DataFrame()
    pending = TICKERS.copy()
    for attempt in range(MAX_ATTEMPTS):
        try:
            downloaded = yf.download(
                tickers=pending, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, group_by="column",
                threads=2, timeout=30,
            )
            part = extract_close(downloaded, pending)
            prices = prices.combine_first(part)
        except Exception as exc:
            print(f"Download attempt {attempt + 1}: {exc}")
        pending = [t for t in TICKERS
                   if t not in prices or not prices[t].notna().any()]
        if not pending:
            break
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(5 * 2 ** attempt)
    if pending:
        raise RuntimeError(f"Missing all price history for {pending}; outputs not replaced")
    prices = prices.reindex(columns=TICKERS).sort_index()
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    # SPY supplies the equity trading calendar; missing anchors remain missing.
    prices = prices.loc[prices["SPY"].notna()]
    if prices.empty:
        raise RuntimeError("No valid SPY trading dates")
    return prices


def daily_signals(returns: pd.DataFrame) -> pd.DataFrame:
    r = returns
    signals = pd.DataFrame(index=r.index)
    # Same-duration comparison: TIP - (D_TIP / D_IEF) * IEF.
    signals["inflation"] = r["TIP"] - DURATION["TIP"] / DURATION["IEF"] * r["IEF"]
    signals["growth"] = r["SPY"]
    signals["dollar"] = r["UUP"]
    signals["credit"] = r["HYG"] - DURATION["HYG"] / DURATION["IEF"] * r["IEF"]
    signals["real_rates"] = -r["TIP"]
    return signals[DIMENSIONS]


def expanding_rank(s: pd.Series) -> pd.Series:
    """Empirical CDF through today, including ties; no future observations."""
    history, ranks = [], []
    for x in s:
        if pd.isna(x):
            ranks.append(np.nan)
        else:
            insort_right(history, float(x))
            ranks.append(bisect_right(history, float(x)) / len(history))
    return pd.Series(ranks, index=s.index)


def describe(bits: tuple[int, ...]) -> dict:
    inflation, growth, dollar, credit, real_rates = bits
    main = {
        (1, 1): "Inflationary growth", (0, 1): "Disinflationary growth",
        (1, 0): "Stagflationary", (0, 0): "Deflationary contraction",
    }[(inflation, growth)]
    code = "".join(map(str, bits))
    result = {
        "regime_id": 1 + int(code, 2), "regime_code": code,
        "main_regime": main,
        "regime_label": " | ".join([main, STATES["dollar"][dollar],
                                      STATES["credit"][credit], STATES["real_rates"][real_rates]]),
    }
    result.update({f"{dim}_state": STATES[dim][bit]
                   for dim, bit in zip(DIMENSIONS, bits)})
    return result


def build_scores(signals: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=signals.index)
    states = pd.DataFrame(index=signals.index)
    scores = pd.DataFrame(index=signals.index)
    for dim in DIMENSIONS:
        raw = signals[dim]
        scale = raw.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std(ddof=0).shift(1)
        score = (raw / scale.replace(0, np.nan)).clip(-CLIP_Z, CLIP_Z)
        scores[dim] = score
        flat = raw.notna() & raw.abs().le(FLAT_EPSILON)
        direction = np.sign(raw).mask(flat).ffill().where(raw.notna())
        states[dim] = direction
        out[f"{dim}_signal"] = raw
        out[f"{dim}_score"] = score
        out[f"{dim}_score_ema{SMOOTH_SPAN}"] = score.ewm(span=SMOOTH_SPAN, adjust=False).mean().where(score.notna())
        out[f"{dim}_pct_rank"] = expanding_rank(score)
        out[f"{dim}_flat"] = flat
    rows = []
    for date, state in states.iterrows():
        missing = signals.loc[date].isna()
        flat_dims = [d for d in DIMENSIONS if out.at[date, f"{d}_flat"]]
        if missing.any():
            row = {"classification_status": "missing_data", "regime_id": None}
        elif state.isna().any():
            row = {"classification_status": "unresolved_flat", "regime_id": None}
        else:
            row = describe(tuple(int(state[d] > 0) for d in DIMENSIONS))
            row["classification_status"] = "flat_carried" if flat_dims else "complete"
        row["missing_dimensions"] = ",".join(missing.index[missing])
        row["flat_dimensions"] = ",".join(flat_dims)
        rows.append(row)
    out = out.join(pd.DataFrame(rows, index=signals.index))
    out["regime_id"] = out["regime_id"].astype("Int64")
    # Magnitude of weakest axis; not a probability or statistical confidence.
    out["min_axis_strength"] = scores.abs().min(axis=1, skipna=False)
    return out


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--warmup-start", default=WARMUP_START)
    # Exclusive end: deliberately omit the current NY date, even after close,
    # so intraday or not-yet-final ETF bars never receive completed labels.
    parser.add_argument("--end", default=datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    args = parser.parse_args()
    start, warmup, end = map(pd.Timestamp, [args.start, args.warmup_start, args.end])
    today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    if not warmup < start < end or end > today:
        raise ValueError("Require warmup-start < start < end <= today's New York date")
    print("Downloading anchors:", TICKERS)
    prices = download_prices(args.warmup_start, args.end)
    returns = prices.pct_change(fill_method=None)
    signals = daily_signals(returns)
    scores = build_scores(signals)
    mask = (prices.index >= start) & (prices.index < end)
    prices_out, returns_out, scores_out = prices.loc[mask], returns.loc[mask], scores.loc[mask]
    if scores_out.empty:
        raise RuntimeError("No output dates; existing files untouched")
    data, public = args.root / "data", args.root / "public" / "data"
    data.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    for name, frame in [("macro_anchor_prices", prices_out),
                        ("macro_anchor_returns", returns_out),
                        ("macro_dimension_scores", scores_out)]:
        path = data / f"{name}.csv"
        temp = path.with_suffix(".csv.tmp")
        frame.to_csv(temp, index_label="date")
        temp.replace(path)
    # Retain the diagnostic output; only this file uses mean-centered z-scores.
    mean = returns.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean().shift(1)
    std = returns.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).std(ddof=0).shift(1)
    ((returns - mean) / std.replace(0, np.nan)).clip(-CLIP_Z, CLIP_Z).loc[mask].to_csv(
        data / "macro_anchor_zscores.csv", index_label="date")
    daily = scores_out.rename_axis("date").reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    # pandas converts missing values to JSON null; strict encoder checks output.
    rows = json.loads(daily.to_json(orient="records", double_precision=10))
    write_json(public / "macro_dimension_scores.json", rows)
    catalog = [describe(bits) for bits in product((0, 1), repeat=5)]
    counts = scores_out["regime_id"].value_counts()
    for regime in catalog:
        regime["observed_days"] = int(counts.get(regime["regime_id"], 0))
    write_json(public / "macro_regime_catalog.json", catalog)
    write_json(public / "macro_dimension_metadata.json", {
        "model_version": "etf_daily_32_v1", "start_date": args.start,
        "warmup_start": args.warmup_start, "end_exclusive": args.end,
        "last_output_date": daily["date"].iloc[-1],
        "tickers": TICKERS, "bit_order": DIMENSIONS,
        "regime_id_rule": "1 + integer value of five-bit regime_code",
        "duration_assumptions_years": DURATION,
        "formulas": {"inflation": "r_TIP - (D_TIP / D_IEF) * r_IEF",
                     "growth": "r_SPY", "dollar": "r_UUP",
                     "credit": "r_HYG - (D_HYG / D_IEF) * r_IEF",
                     "real_rates": "-r_TIP"},
        "signal_units": "decimal ETF returns or weighted combinations; NOT yield changes",
        "score_method": "signal / preceding 60-session population std; min 20; clip +/-3",
        "percentile_method": "expanding CDF of clipped scores, including warmup and today",
        "flat_policy": "last nonzero direction, explicitly flagged; no state if none exists",
        "missing_policy": "no price filling; incomplete daily classification remains null",
        "same_day_policy": "current New York calendar date excluded",
        "status_counts": scores_out["classification_status"].value_counts().to_dict(),
        "limitations": ["Fixed approximate durations, not historical duration matching",
                        "Income, CPI accrual, curve shifts and ETF pricing affect signals",
                        "Dimensions correlated; TIP reused in inflation and real rates",
                        "SPY measures equity direction, not observed economic growth",
                        "No legacy liquidity_score; update frontend for three replacement axes"],
    })
    print(f"Done: {len(rows)} daily rows; all 32 catalog entries; outputs in {data} and {public}")
    print(scores_out["classification_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
