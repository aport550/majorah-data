import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCREENER_PATH = ROOT / "public" / "data" / "stock_screener.json"
SMART_MONEY_PATH = ROOT / "public" / "data" / "smart_money.json"
RETURNS_PATH = ROOT / "data" / "daily_returns.csv"

OUT_PATH = ROOT / "public" / "data" / "ranking_model_buys.json"

TOP_N = 10
DOLLARS_PER_PICK = 1000

SCORE_CONFIG = {
    "quality": [
        {"key": "buybackYield", "min": -0.1, "max": 0.1, "direction": "higher", "allowNegative": True, "weight": 0.15},
        {"key": "growthRate", "min": -0.2, "max": 0.2, "direction": "higher", "allowNegative": True, "weight": 0.20},
        {"key": "grossMargin", "min": 0, "max": 1, "direction": "higher", "allowNegative": True, "weight": 0.10},
        {"key": "profitMargin", "min": -0.1, "max": 0.3, "direction": "higher", "allowNegative": True, "weight": 0.20},
        {"key": "currentRatio", "min": 0.01, "max": 3, "direction": "higher", "allowNegative": False, "weight": 0.15},
        {"key": "institutionalOwnership", "min": 0.5, "max": 1, "direction": "higher", "allowNegative": False, "weight": 0.20},
    ],
    "valuation": [
        {"key": "pb", "min": 0.01, "max": 10, "direction": "lower", "allowNegative": False, "weight": 0.20},
        {"key": "ptbv", "min": 0.01, "max": 10, "direction": "lower", "allowNegative": False, "weight": 0.20},
        {"key": "peg", "min": 0.01, "max": 4, "direction": "lower", "allowNegative": False, "weight": 0.20},
        {"key": "forwardPE", "min": 0.01, "max": 50, "direction": "lower", "allowNegative": False, "weight": 0.20},
        {"key": "evMarketCap", "min": 0.01, "max": 2, "direction": "lower", "allowNegative": False, "weight": 0.20},
    ],
    "scale": [
        {"key": "rsi14", "min": 15, "max": 70, "direction": "lower", "allowNegative": False, "weight": 0.30},
        {"key": "pctAbove52WeekLow", "min": 0, "max": 0.5, "direction": "lower", "allowNegative": False, "weight": 0.70},
    ],
    "defensive": [
        {"key": "beta", "min": 0, "max": 2, "direction": "lower", "allowNegative": True, "weight": 1},
        {"key": "pctUpWhenSpyDown", "min": 0, "max": 0.5, "direction": "higher", "allowNegative": False, "weight": 1},
        {"key": "pctUpWhenSpyCrash", "min": 0, "max": 0.35, "direction": "higher", "allowNegative": False, "weight": 1},
    ],
    "smartMoney": [
        {"key": "smartMoneyScoreRaw", "min": 0, "max": 100, "direction": "higher", "allowNegative": True, "weight": 1},
    ],
}

COMBINED_WEIGHTS = {
    "qualityScore": 0.35,
    "valuationScore": 0.25,
    "scaleScore": 0.12,
    "defensiveScore": 0.18,
    "smartMoneyScore": 0.10,
}


def parse_number(value):
    if value is None or value == "":
        return None
    try:
        value = str(value).replace("$", "").replace("%", "").replace(",", "").strip()
        n = float(value)
        return n if n == n else None
    except Exception:
        return None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def normalize_metric_value(raw_value, config):
    n = parse_number(raw_value)
    if n is None:
        return None

    if not config.get("allowNegative", False) and n <= 0:
        return None

    min_value = config["min"]
    max_value = config["max"]
    rng = max_value - min_value

    if rng <= 0:
        return None

    capped = clamp(n, min_value, max_value)

    if config["direction"] == "lower":
        return ((max_value - capped) / rng) * 100

    return ((capped - min_value) / rng) * 100


def compute_score(row, configs):
    valid_scores = []

    for config in configs:
        key = config["key"]
        raw_value = row.get(key)

        if key == "evMarketCap":
            ev = parse_number(row.get("enterpriseValue"))
            market_cap = parse_number(row.get("marketCap"))
            raw_value = ev / market_cap if ev is not None and market_cap and market_cap > 0 else None

        score = normalize_metric_value(raw_value, config)
        weight = float(config.get("weight", 1))

        if score is not None:
            valid_scores.append((score, weight))

    if not valid_scores:
        return None

    total_weight = sum(weight for _, weight in valid_scores)
    if total_weight <= 0:
        return None

    return sum(score * (weight / total_weight) for score, weight in valid_scores)


def is_likely_etf(row):
    ticker = str(row.get("ticker", "")).strip().upper()
    name = str(row.get("name", "")).lower()

    etf_tickers = {
        "SPY", "QQQ", "IWM", "DIA", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP",
        "XLU", "XLY", "ARKK", "VTI", "VOO", "SCHD", "VEA", "EEM", "TLT", "IEF",
        "HYG", "JNK", "BIL", "GLD", "DBC", "XME", "COPX", "PSQ", "SH", "FBTC", "VXX",
    }

    if ticker in etf_tickers:
        return True

    return any(
        phrase in name
        for phrase in [
            " etf",
            "exchange traded",
            "fund",
            "trust",
            "ishares",
            "spdr",
            "vanguard",
            "invesco",
            "proshares",
            "direxion",
        ]
    )


def load_json(path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_spy_down_stats(rows):
    if not RETURNS_PATH.exists():
        return {}

    stock_tickers = {
        str(row.get("ticker", "")).strip().upper()
        for row in rows
        if row.get("ticker") and not is_likely_etf(row)
    }

    stats = {
        ticker: {
            "spyDown": 0,
            "stockUpWhenSpyDown": 0,
            "spyCrash": 0,
            "stockUpWhenSpyCrash": 0,
        }
        for ticker in stock_tickers
    }

    with RETURNS_PATH.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, [])

        headers = [h.strip().upper() for h in headers]

        if "SPY" not in headers:
            return {}

        spy_index = headers.index("SPY")

        ticker_indexes = [
            (ticker, idx)
            for idx, ticker in enumerate(headers)
            if ticker in stock_tickers and idx != spy_index
        ]

        for cols in reader:
            if len(cols) <= spy_index:
                continue

            spy = parse_number(cols[spy_index])
            if spy is None:
                continue

            spy_decimal = spy / 100
            is_spy_down = spy_decimal < 0
            is_spy_crash = spy_decimal <= -0.015

            if not is_spy_down and not is_spy_crash:
                continue

            for ticker, idx in ticker_indexes:
                if idx >= len(cols):
                    continue

                stock = parse_number(cols[idx])
                if stock is None:
                    continue

                stock_decimal = stock / 100

                if is_spy_down:
                    stats[ticker]["spyDown"] += 1
                    if stock_decimal > 0:
                        stats[ticker]["stockUpWhenSpyDown"] += 1

                if is_spy_crash:
                    stats[ticker]["spyCrash"] += 1
                    if stock_decimal > 0:
                        stats[ticker]["stockUpWhenSpyCrash"] += 1

    final = {}

    for ticker, s in stats.items():
        final[ticker] = {
            "pctUpWhenSpyDown": (
                s["stockUpWhenSpyDown"] / s["spyDown"]
                if s["spyDown"] > 0
                else None
            ),
            "pctUpWhenSpyCrash": (
                s["stockUpWhenSpyCrash"] / s["spyCrash"]
                if s["spyCrash"] > 0
                else None
            ),
        }

    return final


def main():
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    screener_json = load_json(SCREENER_PATH, {"data": []})
    smart_money_json = load_json(SMART_MONEY_PATH, {"data": {}})

    rows = screener_json.get("data", [])
    smart_money = smart_money_json.get("data", {})

    smart_money = {
        str(k).strip().upper(): v
        for k, v in smart_money.items()
        if str(k).strip()
    }

    spy_stats = build_spy_down_stats(rows)

    scored = []

    for row in rows:
        if is_likely_etf(row):
            continue

        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue

        enterprise_value = parse_number(row.get("enterpriseValue"))
        market_cap = parse_number(row.get("marketCap"))

        ev_market_cap = (
            enterprise_value / market_cap
            if enterprise_value is not None and market_cap and market_cap > 0
            else None
        )

        sm = smart_money.get(ticker, {})

        enriched = {
            **row,
            "ticker": ticker,
            "evMarketCap": ev_market_cap,
            "smartMoneyScoreRaw": parse_number(sm.get("smartMoneyScore")),
            "numFundsBuying": parse_number(sm.get("numFundsBuying")),
            "numNewPositions": parse_number(sm.get("numNewPositions")),
            "numFundsSelling": parse_number(sm.get("numFundsSelling")),
            "totalSmartMoneyValueAddedUsd": parse_number(sm.get("totalValueAddedUsd")),
            "pctUpWhenSpyDown": spy_stats.get(ticker, {}).get("pctUpWhenSpyDown"),
            "pctUpWhenSpyCrash": spy_stats.get(ticker, {}).get("pctUpWhenSpyCrash"),
        }

        quality_score = compute_score(enriched, SCORE_CONFIG["quality"])
        valuation_score = compute_score(enriched, SCORE_CONFIG["valuation"])
        scale_score = compute_score(enriched, SCORE_CONFIG["scale"])
        defensive_score = compute_score(enriched, SCORE_CONFIG["defensive"])
        smart_money_score = compute_score(enriched, SCORE_CONFIG["smartMoney"])

        score_parts = {
            "qualityScore": quality_score,
            "valuationScore": valuation_score,
            "scaleScore": scale_score,
            "defensiveScore": defensive_score,
            "smartMoneyScore": smart_money_score,
        }

        valid_weight = sum(
            COMBINED_WEIGHTS[key]
            for key, value in score_parts.items()
            if value is not None
        )

        if valid_weight <= 0:
            continue

        combined_score = sum(
            value * COMBINED_WEIGHTS[key]
            for key, value in score_parts.items()
            if value is not None
        ) / valid_weight

        price = parse_number(enriched.get("price"))

        scored.append({
            **enriched,
            **score_parts,
            "combinedScore": combined_score,
            "price": price,
        })

    scored.sort(key=lambda x: x.get("combinedScore") or -999, reverse=True)

    top_picks = []

    for rank, row in enumerate(scored[:TOP_N], start=1):
        price = row.get("price")
        shares = DOLLARS_PER_PICK / price if price and price > 0 else None

        top_picks.append({
            "rank": rank,
            "ticker": row.get("ticker"),
            "name": row.get("name"),
            "action": "BUY",
            "paper_trade": True,
            "buy_price": price,
            "dollars_allocated": DOLLARS_PER_PICK,
            "shares": shares,
            "combinedScore": row.get("combinedScore"),
            "qualityScore": row.get("qualityScore"),
            "valuationScore": row.get("valuationScore"),
            "scaleScore": row.get("scaleScore"),
            "defensiveScore": row.get("defensiveScore"),
            "smartMoneyScore": row.get("smartMoneyScore"),
            "marketCap": parse_number(row.get("marketCap")),
            "forwardPE": parse_number(row.get("forwardPE")),
            "pb": parse_number(row.get("pb")),
            "rsi14": parse_number(row.get("rsi14")),
        })

    existing = load_json(OUT_PATH, {
        "strategy": {
            "name": "combined_score_top_10_daily_paper_buys",
            "top_n": TOP_N,
            "dollars_per_pick": DOLLARS_PER_PICK,
        },
        "buys_by_date": [],
    })

    buys_by_date = existing.get("buys_by_date", [])

    today_entry = {
        "date": today,
        "created_at": now.isoformat(),
        "data_as_of": screener_json.get("as_of"),
        "smart_money_as_of": smart_money_json.get("as_of"),
        "top_n": TOP_N,
        "picks": top_picks,
    }

    # Idempotent: replace today's entry instead of duplicating it.
    buys_by_date = [x for x in buys_by_date if x.get("date") != today]
    buys_by_date.append(today_entry)
    buys_by_date.sort(key=lambda x: x.get("date", ""))

    output = {
        "as_of": now.isoformat(),
        "strategy": {
            "name": "combined_score_top_10_daily_paper_buys",
            "top_n": TOP_N,
            "dollars_per_pick": DOLLARS_PER_PICK,
            "score_weights": COMBINED_WEIGHTS,
        },
        "latest": today_entry,
        "buys_by_date": buys_by_date,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, allow_nan=False)

    print(f"Wrote {OUT_PATH}")
    print("Top picks:")
    for p in top_picks:
        print(
            p["rank"],
            p["ticker"],
            p["buy_price"],
            round(p["combinedScore"], 2) if p["combinedScore"] is not None else None,
        )


if __name__ == "__main__":
    main()
