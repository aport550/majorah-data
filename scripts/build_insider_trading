#!/usr/bin/env python3
"""
Build historical Finviz insider trading dataset.

What it does:
- Scrapes Finviz insider buys:  https://finviz.com/insidertrading?tc=1
- Scrapes Finviz insider sells: https://finviz.com/insidertrading?tc=2
- Captures Finviz "informative" styling when detectable from row HTML/classes/colors.
- Appends new records to a historical JSON file.
- Deduplicates records with a stable hash key.
- Writes:
    data/insider_trading_history.json
    public/data/insider_trading.json
    public/data/insider_trading_summary.json

Install dependencies:
    pip install requests beautifulsoup4

Recommended GitHub Action:
    python scripts/build_insider_trading.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


FINVIZ_BASE = "https://finviz.com"
BUY_URL = "https://finviz.com/insidertrading?tc=1"
SELL_URL = "https://finviz.com/insidertrading?tc=2"

DEFAULT_HISTORY_PATH = Path("data/insider_trading_history.json")
DEFAULT_PUBLIC_PATH = Path("public/data/insider_trading.json")
DEFAULT_SUMMARY_PATH = Path("public/data/insider_trading_summary.json")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

EXPECTED_HEADERS = [
    "Ticker",
    "Owner",
    "Relationship",
    "Date",
    "Transaction",
    "Cost",
    "#Shares",
    "Value ($)",
    "#Shares Total",
    "SEC Form 4",
]


@dataclass
class InsiderRecord:
    id: str
    ticker: str
    owner: str
    relationship: str
    transaction_date: str
    transaction: str
    transaction_side: str
    cost: Optional[float]
    shares: Optional[int]
    value: Optional[int]
    shares_total: Optional[int]
    sec_form_4_datetime: str
    sec_form_4_url: str
    informative: bool
    informative_reason: str
    source: str
    source_url: str
    captured_at: str


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_float(value: str) -> Optional[float]:
    s = clean_text(value)
    if not s or s in {"-", "—", "N/A"}:
        return None

    s = s.replace("$", "").replace(",", "").replace("%", "")
    s = s.replace("(", "-").replace(")", "")

    try:
        return float(s)
    except ValueError:
        return None


def parse_int(value: str) -> Optional[int]:
    n = parse_float(value)
    if n is None:
        return None
    return int(round(n))


def normalize_ticker(value: str) -> str:
    return clean_text(value).upper()


def parse_finviz_transaction_date(value: str) -> str:
    """
    Finviz shows dates like: May 08 '26
    Return YYYY-MM-DD when possible.
    """
    s = clean_text(value)
    if not s:
        return ""

    for fmt in ("%b %d '%y", "%b %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return s


def parse_sec_form_datetime(value: str) -> str:
    """
    Finviz shows SEC form timestamp like: May 08 06:48 PM
    It usually omits the year. Use current year as a practical default.
    """
    s = clean_text(value)
    if not s:
        return ""

    current_year = datetime.now(timezone.utc).year

    for candidate in (f"{s} {current_year}", s):
        for fmt in ("%b %d %I:%M %p %Y", "%b %d '%y %I:%M %p", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                pass

    return s


def transaction_side_from_url_or_text(source_kind: str, transaction: str) -> str:
    t = clean_text(transaction).lower()

    if "buy" in t:
        return "Buy"
    if "sale" in t or "sell" in t:
        return "Sale"

    if source_kind == "buy":
        return "Buy"
    if source_kind == "sell":
        return "Sale"

    return "Unknown"


def make_record_id(
    ticker: str,
    owner: str,
    relationship: str,
    transaction_date: str,
    transaction: str,
    cost: Optional[float],
    shares: Optional[int],
    value: Optional[int],
    shares_total: Optional[int],
    sec_form_4_url: str,
) -> str:
    """
    Stable dedupe ID. Include SEC URL where available because it is usually the strongest unique signal.
    """
    raw = "|".join(
        [
            normalize_ticker(ticker),
            clean_text(owner).lower(),
            clean_text(relationship).lower(),
            clean_text(transaction_date),
            clean_text(transaction).lower(),
            "" if cost is None else f"{cost:.6f}",
            "" if shares is None else str(shares),
            "" if value is None else str(value),
            "" if shares_total is None else str(shares_total),
            clean_text(sec_form_4_url),
        ]
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def row_looks_informative(row: Any, transaction_side: str, transaction: str, value: Optional[int]) -> tuple[bool, str]:
    """
    Finviz visually highlights informative rows. The exact implementation can change,
    so this function checks several possible signals:
    - HTML class names containing "informative"
    - inline styles / row background colors that look like bright green
    - conservative fallback heuristic for meaningful open-market buys.

    Note: fallback heuristic is intentionally marked separately so you know whether
    the signal came from Finviz styling or inference.
    """
    row_html = str(row).lower()
    row_class = " ".join(row.get("class", [])).lower() if hasattr(row, "get") else ""
    row_style = str(row.get("style", "")).lower() if hasattr(row, "get") else ""

    combined = f"{row_class} {row_style} {row_html}"

    explicit_terms = [
        "informative",
        "insider-buy-row",
        "is-green",
        "background-color:#d",
        "background:#d",
        "rgb(220",
        "rgb(209",
        "rgba(34,197,94",
        "#dcfce7",
        "#d1fae5",
        "#bbf7d0",
    ]

    if any(term in combined for term in explicit_terms):
        return True, "finviz_row_style"

    t = clean_text(transaction).lower()

    uninformative_terms = [
        "option",
        "exercise",
        "gift",
        "award",
        "grant",
        "automatic",
        "tax",
        "withholding",
        "conversion",
        "disposition",
    ]

    if any(term in t for term in uninformative_terms):
        return False, "transaction_type_noise"

    # Conservative fallback: sizable open-market buys are often the most informative class.
    if transaction_side == "Buy" and value is not None and value >= 100_000:
        return True, "fallback_large_open_market_buy"

    return False, "not_flagged"


def extract_cells(row: Any) -> List[str]:
    cells = row.find_all("td")
    return [clean_text(c.get_text(" ", strip=True)) for c in cells]


def find_insider_rows(soup: BeautifulSoup) -> Iterable[Any]:
    """
    Find data rows in the Finviz insider table.

    The table may not expose a stable id/class, so this detects rows by shape:
    exactly 10 cells and first cell is a ticker-looking symbol.
    """
    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 10:
            continue

        texts = [clean_text(c.get_text(" ", strip=True)) for c in cells]
        ticker = normalize_ticker(texts[0])

        if not ticker or ticker in {"TICKER", "FILTER"}:
            continue

        if not re.match(r"^[A-Z0-9.\-]{1,12}$", ticker):
            continue

        yield row


def scrape_finviz_page(url: str, source_kind: str, sleep_seconds: float = 0.8) -> List[InsiderRecord]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": FINVIZ_BASE,
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    if "<html" not in response.text.lower():
        raise RuntimeError(f"Unexpected non-HTML response from {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    captured_at = now_utc_iso()
    records: List[InsiderRecord] = []

    for row in find_insider_rows(soup):
        cells = row.find_all("td")
        texts = extract_cells(row)

        if len(texts) != 10:
            continue

        ticker = normalize_ticker(texts[0])
        owner = texts[1]
        relationship = texts[2]
        transaction_date = parse_finviz_transaction_date(texts[3])
        transaction = texts[4]
        cost = parse_float(texts[5])
        shares = parse_int(texts[6])
        value = parse_int(texts[7])
        shares_total = parse_int(texts[8])

        sec_cell = cells[9]
        sec_link_tag = sec_cell.find("a")
        sec_form_text = texts[9]
        sec_form_url = (
            urljoin(FINVIZ_BASE, sec_link_tag.get("href"))
            if sec_link_tag and sec_link_tag.get("href")
            else ""
        )
        sec_form_dt = parse_sec_form_datetime(sec_form_text)

        transaction_side = transaction_side_from_url_or_text(source_kind, transaction)
        informative, informative_reason = row_looks_informative(
            row=row,
            transaction_side=transaction_side,
            transaction=transaction,
            value=value,
        )

        record_id = make_record_id(
            ticker=ticker,
            owner=owner,
            relationship=relationship,
            transaction_date=transaction_date,
            transaction=transaction,
            cost=cost,
            shares=shares,
            value=value,
            shares_total=shares_total,
            sec_form_4_url=sec_form_url,
        )

        records.append(
            InsiderRecord(
                id=record_id,
                ticker=ticker,
                owner=owner,
                relationship=relationship,
                transaction_date=transaction_date,
                transaction=transaction,
                transaction_side=transaction_side,
                cost=cost,
                shares=shares,
                value=value,
                shares_total=shares_total,
                sec_form_4_datetime=sec_form_dt,
                sec_form_4_url=sec_form_url,
                informative=informative,
                informative_reason=informative_reason,
                source="finviz",
                source_url=url,
                captured_at=captured_at,
            )
        )

    time.sleep(sleep_seconds)
    return records


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]

        return []
    except json.JSONDecodeError:
        print(f"Warning: {path} is invalid JSON. Starting from empty history.")
        return []


def merge_and_dedupe(existing: List[Dict[str, Any]], new_records: List[InsiderRecord]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}

    for item in existing:
        record_id = item.get("id")
        if not record_id:
            record_id = make_record_id(
                ticker=item.get("ticker", ""),
                owner=item.get("owner", ""),
                relationship=item.get("relationship", ""),
                transaction_date=item.get("transaction_date", ""),
                transaction=item.get("transaction", ""),
                cost=item.get("cost"),
                shares=item.get("shares"),
                value=item.get("value"),
                shares_total=item.get("shares_total"),
                sec_form_4_url=item.get("sec_form_4_url", ""),
            )
            item["id"] = record_id

        by_id[record_id] = item

    for record in new_records:
        incoming = asdict(record)

        # Preserve first captured_at if already present, but update fields that might improve.
        if record.id in by_id:
            prior = by_id[record.id]
            incoming["captured_at"] = prior.get("captured_at") or incoming["captured_at"]

            # If either version detected informative, keep true.
            incoming["informative"] = bool(prior.get("informative")) or bool(incoming.get("informative"))

            if prior.get("informative") and not incoming.get("informative_reason"):
                incoming["informative_reason"] = prior.get("informative_reason", "")

        by_id[record.id] = incoming

    merged = list(by_id.values())

    def sort_key(x: Dict[str, Any]) -> tuple:
        return (
            x.get("transaction_date") or "",
            x.get("sec_form_4_datetime") or "",
            x.get("value") or 0,
        )

    merged.sort(key=sort_key, reverse=True)
    return merged


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_summary(records: List[Dict[str, Any]], new_count: int) -> Dict[str, Any]:
    buys = [r for r in records if r.get("transaction_side") == "Buy"]
    sells = [r for r in records if r.get("transaction_side") == "Sale"]
    informative = [r for r in records if r.get("informative")]

    latest_date = max((r.get("transaction_date") or "" for r in records), default="")

    value_by_side = {
        "Buy": sum(int(r.get("value") or 0) for r in buys),
        "Sale": sum(int(r.get("value") or 0) for r in sells),
    }

    top_informative = sorted(
        informative,
        key=lambda r: int(r.get("value") or 0),
        reverse=True,
    )[:25]

    return {
        "generated_at": now_utc_iso(),
        "latest_transaction_date": latest_date,
        "record_count": len(records),
        "new_records_seen_this_run": new_count,
        "buy_count": len(buys),
        "sale_count": len(sells),
        "informative_count": len(informative),
        "value_by_side": value_by_side,
        "top_informative_by_value": top_informative,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-path", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--public-path", default=str(DEFAULT_PUBLIC_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--sleep-seconds", type=float, default=0.8)
    parser.add_argument("--max-public-records", type=int, default=2500)
    args = parser.parse_args()

    history_path = Path(args.history_path)
    public_path = Path(args.public_path)
    summary_path = Path(args.summary_path)

    print("Scraping Finviz insider buys...")
    buy_records = scrape_finviz_page(BUY_URL, "buy", sleep_seconds=args.sleep_seconds)
    print(f"Fetched buy records: {len(buy_records)}")

    print("Scraping Finviz insider sells...")
    sell_records = scrape_finviz_page(SELL_URL, "sell", sleep_seconds=args.sleep_seconds)
    print(f"Fetched sell records: {len(sell_records)}")

    new_records = buy_records + sell_records

    existing = load_json_array(history_path)
    before_count = len(existing)

    merged = merge_and_dedupe(existing, new_records)
    after_count = len(merged)
    new_unique_count = max(0, after_count - before_count)

    print(f"Existing records: {before_count}")
    print(f"Merged records: {after_count}")
    print(f"New unique records: {new_unique_count}")

    write_json(history_path, merged)

    public_records = merged[: args.max_public_records]
    write_json(
        public_path,
        {
            "generated_at": now_utc_iso(),
            "source": "finviz",
            "source_urls": {
                "buys": BUY_URL,
                "sells": SELL_URL,
            },
            "record_count": len(public_records),
            "history_record_count": len(merged),
            "data": public_records,
        },
    )

    summary = build_summary(merged, new_unique_count)
    write_json(summary_path, summary)

    print(f"Wrote history: {history_path}")
    print(f"Wrote public data: {public_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
