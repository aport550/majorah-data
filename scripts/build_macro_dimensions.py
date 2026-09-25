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

