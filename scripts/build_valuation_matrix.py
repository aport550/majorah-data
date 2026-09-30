#!/usr/bin/env python3
"""Build five-year historical valuation screens from FMP stable APIs.

Install: python -m pip install pandas numpy requests
Set FMP_API_KEY in your environment; run from the project root.
Example: python build_valuation_matrix.py --tickers AAPL MSFT --frequency quarterly
Default universe: data/universe.csv, column Ticker; optional FMP_Ticker override.
Default frequency: monthly. Outputs: data/valuation_{history,summary,coverage}.csv
and data/valuations.json. --public also writes frontend CSV/JSON to public/data.
Confirm your vendor license permits public redistribution before using --public.

METHOD / LIMITS
- Company market cap from FMP (never dividend-adjusted Yahoo prices).
- EV = market cap + total debt + preferred stock + minority interest - cash.
- Common book equity = stockholders' equity - preferred stock.
- Flow denominators sum four consecutive vendor-normalized standalone quarters.
- Financials available from the day AFTER the latest filing/acceptance date in
  the four-quarter window. Unknown filing dates are excluded, never guessed.
- Latest/restated vendor statements: NOT a vintage point-in-time backtest feed.
- USD quote AND USD financial reporting only in this version. Other currencies
  and cross-currency ADRs are logged as unsupported; no implicit FX conversion.
- Funds skipped. Financial Services EV metrics suppressed; P/B retained.
- Nonpositive denominators and nonpositive EV are null, not 'cheap'.
- A current universe has survivorship bias; delisted names must be supplied.
- Current means the latest available historical-market-cap observation, with
  its actual date exposed. No forward estimates or corporate-action adjustments
  beyond those supplied by FMP. Validate vendor data before acting on outliers.
- Cache is local, 24-hour TTL by default. API key is never stored or logged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://financialmodelingprep.com/stable"
METRICS = ["pb", "ev_ebitda", "ev_sales", "ev_ocf", "ev_fcf"]


class APIError(RuntimeError):
    pass


class Client:
    def __init__(self, key, cache, pause=0.3, ttl=24):
        self.key, self.cache, self.pause, self.ttl = key, Path(cache), pause, ttl
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def get(self, endpoint, **params):
        digest = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        path = self.cache / (digest + ".json")
        if path.exists() and time.time() - path.stat().st_mtime < self.ttl * 3600:
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError):
                pass
        for attempt in range(5):
            time.sleep(self.pause)
            try:
                response = self.session.get(BASE + "/" + endpoint,
                                            params={**params, "apikey": self.key}, timeout=45)
            except requests.RequestException:
                if attempt == 4:
                    raise APIError(f"{endpoint}: network failure") from None
                time.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    raise APIError(f"{endpoint}: HTTP {response.status_code}; retry later")
                time.sleep(min(30, 2 ** (attempt + 1)))
                continue
            if response.status_code != 200:
                raise APIError(f"{endpoint}: HTTP {response.status_code}; check key/plan/access")
            try:
                result = response.json()
            except ValueError:
                raise APIError(f"{endpoint}: invalid JSON") from None
            if not isinstance(result, list) or any(not isinstance(x, dict) for x in result):
                raise APIError(f"{endpoint}: unexpected response; check plan/API schema")
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result))
            temp.replace(path)
            return result


def number(row, key):
    try:
        value = float(row.get(key))
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def filing_day(row):
    # Use all known dates conservatively; acceptedDate may contain a time.
    dates = [pd.to_datetime(str(row.get(k, ""))[:10], errors="coerce")
             for k in ("filingDate", "acceptedDate")]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else pd.NaT


def statements_by_date(rows):
    result = {}
    for row in rows:
        if row.get("period") not in ("Q1", "Q2", "Q3", "Q4"):
            continue
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        # Prefer latest supplied version when duplicates exist; explicitly restated.
        previous = result.get(date)
        if previous is None or str(row.get("acceptedDate", "")) > str(previous.get("acceptedDate", "")):
            result[date] = row
    return result


def build_fundamentals(income, balance, cash):
    inc, bal, cf = map(statements_by_date, (income, balance, cash))
    dates = sorted(set(inc) & set(bal) & set(cf))
    output = []
    for j in range(3, len(dates)):
        window = dates[j-3:j+1]
        gaps = [(b-a).days for a, b in zip(window, window[1:])]
        if not all(65 <= gap <= 115 for gap in gaps):
            continue  # missing quarter / semiannual reporting / transition periods
        records = [source[d] for d in window for source in (inc, bal, cf)]
        if any(r.get("reportedCurrency") != "USD" for r in records):
            continue
        filings = [filing_day(r) for r in records]
        if any(pd.isna(x) for x in filings):
            continue
        end = window[-1]
        available = max(filings) + pd.Timedelta(days=1)
        if available <= end:
            continue
        row = {"fundamental_date": end, "available_date": available}
        for field, source, target in [
            ("revenue", inc, "revenue_ttm"), ("ebitda", inc, "ebitda_ttm"),
            ("operatingCashFlow", cf, "ocf_ttm"), ("freeCashFlow", cf, "fcf_ttm")
        ]:
            values = [number(source[d], field) for d in window]
            row[target] = sum(values) if all(np.isfinite(values)) else np.nan
        b = bal[end]
        preferred = number(b, "preferredStock")
        row.update(debt=number(b, "totalDebt"), cash=number(b, "cashAndCashEquivalents"),
                   preferred=preferred, minority=number(b, "minorityInterest"),
                   book=number(b, "totalStockholdersEquity") - preferred)
        output.append(row)
    return pd.DataFrame(output)


def ratio(numerator, denominator):
    return numerator / denominator if np.isfinite(numerator) and numerator > 0 and np.isfinite(denominator) and denominator > 0 else np.nan


def value_on(cap_date, market_cap, fundamentals, financial_sector, max_age):
    eligible = fundamentals[fundamentals.available_date <= cap_date]
    if eligible.empty:
        return None
    f = eligible.sort_values("fundamental_date").iloc[-1]
    age = (cap_date - f.fundamental_date).days
    ev = market_cap + f.debt + f.preferred + f.minority - f.cash
    row = {"date": cap_date.date().isoformat(), "market_cap": market_cap,
           "fundamental_date": f.fundamental_date.date().isoformat(),
           "available_date": f.available_date.date().isoformat(),
           "fundamental_age_days": age, "enterprise_value": ev,
           "stale_fundamentals": age > max_age,
           **{k: f[k] for k in ("book", "revenue_ttm", "ebitda_ttm", "ocf_ttm", "fcf_ttm")}}
    row["pb"] = ratio(market_cap, f.book)
    for metric, denom in [("ev_ebitda", "ebitda_ttm"), ("ev_sales", "revenue_ttm"),
                          ("ev_ocf", "ocf_ttm"), ("ev_fcf", "fcf_ttm")]:
        row[metric] = np.nan if financial_sector else ratio(ev, f[denom])
    if age > max_age:
        row.update({metric: np.nan for metric in METRICS})
    return row


def summarize(history, current, minimum):
    result = {k: current[k] for k in ("date", "fundamental_date", "fundamental_age_days", "stale_fundamentals")}
    for metric in METRICS:
        # Exclude the current observation if it coincides with a historical snapshot.
        values = np.array([r[metric] for r in history if r["date"] < current["date"] and np.isfinite(r[metric])])
        value = current[metric]
        result.update({metric: value, metric + "_n": len(values),
                       metric + "_median": float(np.median(values)) if len(values) else np.nan,
                       metric + "_percentile": np.nan, metric + "_discount_pct": np.nan})
        if len(values) >= minimum and np.isfinite(value):
            # Midrank ties. Low percentile = low multiple relative to own history.
            result[metric + "_percentile"] = 100 * (np.sum(values < value) + 0.5 * np.sum(values == value)) / len(values)
            result[metric + "_discount_pct"] = 100 * (1 - value / np.median(values))
    return result


def process(client, symbol, args, end, start):
    profiles = client.get("profile", symbol=symbol)
    if not profiles:
        raise ValueError("no_profile")
    profile = profiles[0]
    if profile.get("isEtf") or profile.get("isFund"):
        raise ValueError("fund_not_company")
    if profile.get("currency") != "USD":
        raise ValueError("unsupported_quote_currency")
    statements = [client.get(endpoint, symbol=symbol, period="quarter", limit=(args.years+3)*4)
                  for endpoint in ("income-statement", "balance-sheet-statement", "cash-flow-statement")]
    if any(not rows for rows in statements):
        raise ValueError("missing_statements")
    fundamentals = build_fundamentals(*statements)
    if fundamentals.empty:
        raise ValueError("no_valid_USD_TTM_windows_check_currency_quarters_and_filing_dates")
    caps = []
    # Annual chunks avoid silently receiving only a default short response.
    cursor = start
    while cursor <= end:
        stop = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        records = client.get("historical-market-capitalization", symbol=symbol,
                             **{"from": cursor.date().isoformat(), "to": stop.date().isoformat(), "limit": 1000})
        caps.extend(records)
        cursor = stop + pd.Timedelta(days=1)
    if not caps:
        raise ValueError("no_historical_market_caps")
    caps = pd.DataFrame(caps)
    if not {"date", "marketCap"}.issubset(caps.columns):
        raise ValueError("market_cap_schema_mismatch")
    caps["date"] = pd.to_datetime(caps.date, errors="coerce")
    caps["marketCap"] = pd.to_numeric(caps.marketCap, errors="coerce")
    caps = caps.dropna(subset=["date", "marketCap"]).drop_duplicates("date").sort_values("date")
    caps = caps[(caps.date >= start) & (caps.date <= end) & (caps.marketCap > 0)]
    if caps.empty:
        raise ValueError("no_market_caps_in_requested_window")
    financial = profile.get("sector") == "Financial Services"
    freq = "M" if args.frequency == "monthly" else "Q"
    periods = pd.period_range(start=start, end=end, freq=freq)
    history = []
    for period in periods:
        target = period.end_time.normalize()
        if target >= end:
            continue  # only completed periods; latest snapshot kept separately
        candidates = caps[(caps.date <= target) & (caps.date >= target - pd.Timedelta(days=7))]
        if candidates.empty:
            continue
        c = candidates.iloc[-1]
        row = value_on(c.date, c.marketCap, fundamentals, financial, args.max_age)
        if row:
            row["period_end"] = target.date().isoformat()
            history.append(row)
    latest = caps.iloc[-1]
    current = value_on(latest.date, latest.marketCap, fundamentals, financial, args.max_age)
    if current is None:
        raise ValueError("no_available_financials_for_latest_market_cap")
    summary = summarize(history, current, args.min_observations)
    summary.update(sector=profile.get("sector"), industry=profile.get("industry"),
                   market_cap_age_days=(end-latest.date).days,
                   history_rows=len(history), expected_periods=sum(p.end_time.normalize() < end for p in periods))
    # Stale market prices must not produce seemingly current percentile signals.
    if summary["market_cap_age_days"] > 7:
        for m in METRICS:
            summary[m + "_percentile"] = np.nan
            summary[m + "_discount_pct"] = np.nan
    return history, current, summary


def write_json(path, payload):
    # pandas serialization converts NaN to JSON null and numpy scalars to numbers.
    clean = json.loads(pd.Series([payload]).to_json(orient="values"))[0]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default="data/universe.csv")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--frequency", choices=["monthly", "quarterly"], default="monthly")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument("--max-age", type=int, default=200, help="Maximum age of fiscal period in days")
    parser.add_argument("--pause", type=float, default=0.3, help="Seconds between API requests; adapt to plan")
    parser.add_argument("--cache-hours", type=float, default=24)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--public", action="store_true", help="Also export to public/data; requires suitable data license")
    args = parser.parse_args()
    if args.years < 1 or args.min_observations < 1 or args.pause < 0 or args.cache_hours < 0:
        parser.error("years/min-observations must be positive; pause/cache-hours nonnegative")
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        parser.error("Set FMP_API_KEY environment variable before running")
    if args.tickers:
        universe = pd.DataFrame({"Ticker": args.tickers})
    else:
        universe = pd.read_csv(args.universe, dtype=str)
    if "Ticker" not in universe:
        parser.error("Universe CSV must contain Ticker")
    universe = universe.dropna(subset=["Ticker"]).copy()
    universe["Ticker"] = universe.Ticker.str.strip().str.upper()
    universe = universe[universe.Ticker != ""].drop_duplicates("Ticker")
    end = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    start = end - pd.DateOffset(years=args.years)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(key, out / "fmp_cache", args.pause, args.cache_hours)
    histories, summaries, coverage, per_ticker = [], [], [], {}
    for index, record in enumerate(universe.to_dict("records"), 1):
        ticker = record["Ticker"]
        override = record.get("FMP_Ticker")
        symbol = str(override).strip() if pd.notna(override) and str(override).strip() else ticker.replace(".", "-")
        print(f"[{index}/{len(universe)}] {ticker} ({symbol})", flush=True)
        try:
            history, current, summary = process(client, symbol, args, end, start)
            histories.extend({"Ticker": ticker, **row} for row in history)
            summaries.append({"Ticker": ticker, **summary})
            per_ticker[ticker] = {"history": history, "current": current, "summary": summary}
            coverage.append({"Ticker": ticker, "FMP_Ticker": symbol, "status": "ok",
                             "history_rows": len(history), "expected_periods": summary["expected_periods"],
                             "first_date": history[0]["date"] if history else None,
                             "last_date": current["date"], "reason": ""})
        except (APIError, ValueError, KeyError) as exc:
            # Never include raw requests exceptions (their URLs can contain API keys).
            reason = str(exc)
            print(f"  Skipped: {reason}", flush=True)
            coverage.append({"Ticker": ticker, "FMP_Ticker": symbol, "status": "skipped", "reason": reason})
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(out / "valuation_coverage.csv", index=False)
    if not summaries:
        raise SystemExit("No usable tickers; see valuation_coverage.csv. Existing result files were not overwritten.")
    payload = {"as_of": end.date().isoformat(), "start_date": start.date().isoformat(),
               "frequency": args.frequency, "source": "FMP stable", "currency": "USD",
               "method": "filing-lagged, latest/restated quarterly fundamentals; not vintage point-in-time",
               "percentile": "0-100 midrank; lower means cheaper vs own valid historical observations",
               "requested_tickers": len(universe), "successful_tickers": len(summaries),
               "tickers": per_ticker}
    destinations = [out] + ([Path("public/data")] if args.public else [])
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(histories).to_csv(destination / "valuation_history.csv", index=False)
        pd.DataFrame(summaries).to_csv(destination / "valuation_summary.csv", index=False)
        write_json(destination / "valuations.json", payload)
    print(f"Saved {len(summaries)}/{len(universe)} tickers to {out}; inspect coverage before screening.")


if __name__ == "__main__":
    main()
