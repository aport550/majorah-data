#!/usr/bin/env python3
"""
build_smart_money.py

Builds a ticker-level "smart money" 13F accumulation dataset for Majorah.

What it does:
1. Pulls latest and prior 13F-HR filings for a curated list of institutional managers.
2. Parses the SEC 13F information table XML.
3. Compares current quarter vs prior quarter holdings.
4. Scores stocks based on active accumulation:
   - new positions
   - shares added
   - value added
   - number of curated funds buying
   - top-holder / concentrated-position style signals
5. Writes:
   - public/data/smart_money.json
   - data/smart_money_raw.csv
   - data/smart_money_fund_holdings.csv
   - data/smart_money_unmapped_cusips.csv

Important:
- SEC 13F data usually reports CUSIP, not ticker.
- For ticker-level output, add a CSV at:
    data/cusip_ticker_map.csv

Expected cusip_ticker_map.csv columns:
    cusip,ticker

Example:
    594918104,MSFT
    037833100,AAPL

GitHub Actions notes:
- SEC asks automated requests to include a real User-Agent identifying your app/contact.
- Set this secret/env var:
    SEC_USER_AGENT="Majorah smart money builder contact: your-email@example.com"

Run:
    python build_smart_money.py

Optional env vars:
    SEC_USER_AGENT
    SMART_MONEY_MIN_VALUE_USD
    SMART_MONEY_MAX_FUNDS
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DATA_DIR = ROOT / "data"
PUBLIC_DATA_DIR = ROOT / "public" / "data"

CUSIP_TICKER_MAP_PATH = DATA_DIR / "cusip_ticker_map.csv"

OUT_JSON = PUBLIC_DATA_DIR / "smart_money.json"
OUT_RAW_CSV = DATA_DIR / "smart_money_raw.csv"
OUT_HOLDINGS_CSV = DATA_DIR / "smart_money_fund_holdings.csv"
OUT_UNMAPPED_CUSIPS_CSV = DATA_DIR / "smart_money_unmapped_cusips.csv"

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Majorah smart money builder contact: aport550@gmail.com",
)

REQUEST_SLEEP_SECONDS = 0.12
MIN_VALUE_USD = float(os.environ.get("SMART_MONEY_MIN_VALUE_USD", "1000000"))
MAX_FUNDS = int(os.environ.get("SMART_MONEY_MAX_FUNDS", "0") or "0")

# Starter list. You should curate this over time.
# Avoid passive giants like BlackRock/Vanguard/State Street for scoring.
FUND_CIKS: Dict[str, str] = {
    "Berkshire Hathaway": "0001067983",
    "Pershing Square": "0001336528",
    "Baupost Group": "0001061768",
    "Greenlight Capital": "0001079114",
    "Third Point": "0001040273",
    "Appaloosa": "0001006438",
    "Scion Asset Management": "0001649339",
    "Coatue Management": "0001135730",
    "Tiger Global Management": "0001167483",
    "Lone Pine Capital": "0001061165",
    "D1 Capital Partners": "0001747057",
    "Duquesne Family Office": "0001536411",
    "Soros Fund Management": "0001029160",
    "Elliott Investment Management": "0001791786",
    "Starboard Value": "0001517137",
    "Akre Capital Management": "0001050470",
    "Polen Capital Management": "0001034524",
    "Miller Value Partners": "0001314173",
}


@dataclass
class FilingRef:
    cik: str
    fund_name: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str
    form: str


@dataclass
class Holding:
    fund_name: str
    fund_cik: str
    accession: str
    filing_date: str
    report_date: str
    cusip: str
    ticker: Optional[str]
    issuer: str
    shares: float
    value_usd: float
    put_call: str


def sec_get(url: str, *, expect_json: bool = False) -> Any:
    # Do NOT manually request gzip here. urllib does not automatically decompress
    # compressed SEC responses, which can cause json.loads to fail with:
    # "Expecting value: line 1 column 1 (char 0)".
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                text = raw.decode("utf-8", errors="replace").strip()
                time.sleep(REQUEST_SLEEP_SECONDS)

                if expect_json:
                    if not text:
                        raise ValueError(f"Empty response from SEC: {url}")

                    # SEC may sometimes return an HTML rate-limit/error page.
                    if text.startswith("<"):
                        preview = text[:300].replace("\n", " ")
                        raise ValueError(f"Expected JSON but received HTML from SEC: {preview}")

                    return json.loads(text)

                return text

        except urllib.error.HTTPError as e:
            if e.code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, ValueError, json.JSONDecodeError):
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise


def normalize_cik(cik: str) -> str:
    digits = re.sub(r"\D", "", str(cik or ""))
    return digits.zfill(10)


def compact_cik(cik: str) -> str:
    return str(int(normalize_cik(cik)))


def clean_cusip(cusip: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(cusip or "")).upper()


def clean_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper().replace(".", "-")


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        n = float(str(value).replace(",", "").strip())
        return n if math.isfinite(n) else 0.0
    except Exception:
        return 0.0


def load_cusip_ticker_map(path: Path = CUSIP_TICKER_MAP_PATH) -> Dict[str, str]:
    if not path.exists():
        print(f"WARNING: Missing {path}. Ticker-level smart_money.json will be limited.")
        print("Create data/cusip_ticker_map.csv with columns: cusip,ticker")
        return {}

    out: Dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        cusip_col = cols.get("cusip")
        ticker_col = cols.get("ticker") or cols.get("symbol")

        if not cusip_col or not ticker_col:
            raise ValueError("cusip_ticker_map.csv must contain columns: cusip,ticker")

        for row in reader:
            cusip = clean_cusip(row.get(cusip_col, ""))
            ticker = clean_ticker(row.get(ticker_col, ""))
            if cusip and ticker:
                out[cusip] = ticker

    print(f"Loaded {len(out):,} CUSIP -> ticker mappings.")
    return out


def get_recent_13f_filings(cik: str, fund_name: str, limit: int = 2) -> List[FilingRef]:
    cik10 = normalize_cik(cik)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    data = sec_get(url, expect_json=True)

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])

    filings: List[FilingRef] = []
    for i, form in enumerate(forms):
        form_upper = str(form or "").upper()
        if form_upper not in {"13F-HR", "13F-HR/A"}:
            continue

        filings.append(
            FilingRef(
                cik=cik10,
                fund_name=fund_name,
                accession=accession_numbers[i],
                filing_date=filing_dates[i] if i < len(filing_dates) else "",
                report_date=report_dates[i] if i < len(report_dates) else "",
                primary_document=primary_documents[i] if i < len(primary_documents) else "",
                form=form_upper,
            )
        )

        if len(filings) >= limit:
            break

    return filings


def filing_index_json(filing: FilingRef) -> Dict[str, Any]:
    cik_no_zeros = compact_cik(filing.cik)
    accession_no_dash = filing.accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_no_dash}/index.json"
    )
    return sec_get(url, expect_json=True)


def find_information_table_document(filing: FilingRef) -> Optional[str]:
    index = filing_index_json(filing)
    items = index.get("directory", {}).get("item", [])

    candidates: List[str] = []
    for item in items:
        name = item.get("name", "")
        lower = name.lower()
        if lower.endswith(".xml") and (
            "infotable" in lower
            or "informationtable" in lower
            or "form13f" in lower
            or "primary_doc" not in lower
        ):
            candidates.append(name)

    for name in candidates:
        lower = name.lower()
        if "info" in lower or "table" in lower:
            return name

    for name in candidates:
        if name != filing.primary_document:
            return name

    return None


def download_filing_document(filing: FilingRef, document_name: str) -> str:
    cik_no_zeros = compact_cik(filing.cik)
    accession_no_dash = filing.accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_zeros}/{accession_no_dash}/{document_name}"
    )
    return sec_get(url, expect_json=False)


def strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def child_text_any_namespace(node: ET.Element, child_name: str) -> str:
    for child in list(node):
        if strip_namespace(child.tag).lower() == child_name.lower():
            return (child.text or "").strip()
    return ""


def find_info_table_nodes(root: ET.Element) -> List[ET.Element]:
    return [node for node in root.iter() if strip_namespace(node.tag).lower() == "infotable"]


def parse_13f_xml(
    xml_text: str,
    filing: FilingRef,
    cusip_to_ticker: Dict[str, str],
) -> List[Holding]:
    xml_text = xml_text.strip()
    first_xml = xml_text.find("<")
    if first_xml > 0:
        xml_text = xml_text[first_xml:]

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"XML parse failed for {filing.fund_name} {filing.accession}: {e}")
        return []

    holdings: List[Holding] = []

    for node in find_info_table_nodes(root):
        issuer = child_text_any_namespace(node, "nameOfIssuer")
        cusip = clean_cusip(child_text_any_namespace(node, "cusip"))
        value_thousands = safe_float(child_text_any_namespace(node, "value"))
        put_call = child_text_any_namespace(node, "putCall").upper()

        shares = 0.0
        for sub in node.iter():
            if strip_namespace(sub.tag).lower() == "sshprnamt":
                shares = safe_float(sub.text)
                break

        value_usd = value_thousands * 1000.0

        if not cusip:
            continue

        # Ignore options for the first version.
        if put_call in {"PUT", "CALL"}:
            continue

        if value_usd < MIN_VALUE_USD:
            continue

        ticker = cusip_to_ticker.get(cusip)

        holdings.append(
            Holding(
                fund_name=filing.fund_name,
                fund_cik=filing.cik,
                accession=filing.accession,
                filing_date=filing.filing_date,
                report_date=filing.report_date,
                cusip=cusip,
                ticker=ticker,
                issuer=issuer,
                shares=shares,
                value_usd=value_usd,
                put_call=put_call,
            )
        )

    return holdings


def percentile_rank(values: Dict[str, float]) -> Dict[str, float]:
    items = [(k, v) for k, v in values.items() if math.isfinite(v)]
    if not items:
        return {}
    items.sort(key=lambda x: x[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 100.0}
    return {k: (i / (n - 1)) * 100.0 for i, (k, _) in enumerate(items)}


def compute_smart_money_scores(
    latest_holdings_by_fund: Dict[str, List[Holding]],
    prior_holdings_by_fund: Dict[str, List[Holding]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    raw_rows: List[Dict[str, Any]] = []

    ticker_agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "ticker": "",
            "numFundsBuying": 0,
            "numNewPositions": 0,
            "numFundsSelling": 0,
            "totalValueAddedUsd": 0.0,
            "totalSharesAdded": 0.0,
            "totalCurrentValueUsd": 0.0,
            "maxPositionWeightPct": 0.0,
            "buyers": [],
            "sellers": [],
            "newPositionFunds": [],
            "topFunds": [],
        }
    )

    for fund_name, latest_holdings in latest_holdings_by_fund.items():
        prior_holdings = prior_holdings_by_fund.get(fund_name, [])
        latest_total_value = sum(h.value_usd for h in latest_holdings if h.value_usd > 0)
        prior_by_cusip = {h.cusip: h for h in prior_holdings}

        for h in latest_holdings:
            if not h.ticker:
                continue

            prior = prior_by_cusip.get(h.cusip)
            prior_shares = prior.shares if prior else 0.0
            prior_value = prior.value_usd if prior else 0.0

            shares_added = h.shares - prior_shares
            value_added = h.value_usd - prior_value
            shares_added_pct = shares_added / prior_shares if prior_shares > 0 else None
            is_new_position = prior is None or prior_shares <= 0
            is_buying = is_new_position or shares_added > 0 or value_added > 0
            is_selling = (not is_new_position) and (shares_added < 0 or value_added < 0)

            position_weight_pct = h.value_usd / latest_total_value if latest_total_value > 0 else 0.0

            raw_rows.append(
                {
                    "ticker": h.ticker,
                    "cusip": h.cusip,
                    "issuer": h.issuer,
                    "fundName": fund_name,
                    "fundCik": h.fund_cik,
                    "latestReportDate": h.report_date,
                    "latestFilingDate": h.filing_date,
                    "latestValueUsd": round(h.value_usd, 2),
                    "priorValueUsd": round(prior_value, 2),
                    "valueAddedUsd": round(value_added, 2),
                    "latestShares": round(h.shares, 4),
                    "priorShares": round(prior_shares, 4),
                    "sharesAdded": round(shares_added, 4),
                    "sharesAddedPct": round(shares_added_pct, 6) if shares_added_pct is not None else None,
                    "positionWeightPct": round(position_weight_pct, 6),
                    "isNewPosition": bool(is_new_position),
                    "isBuying": bool(is_buying),
                    "isSelling": bool(is_selling),
                }
            )

            agg = ticker_agg[h.ticker]
            agg["ticker"] = h.ticker
            agg["totalCurrentValueUsd"] += h.value_usd
            agg["maxPositionWeightPct"] = max(agg["maxPositionWeightPct"], position_weight_pct)

            if is_buying:
                agg["numFundsBuying"] += 1
                agg["totalValueAddedUsd"] += max(value_added, h.value_usd if is_new_position else 0.0)
                agg["totalSharesAdded"] += max(shares_added, h.shares if is_new_position else 0.0)
                agg["buyers"].append(fund_name)

            if is_selling:
                agg["numFundsSelling"] += 1
                agg["sellers"].append(fund_name)

            if is_new_position:
                agg["numNewPositions"] += 1
                agg["newPositionFunds"].append(fund_name)

            if position_weight_pct >= 0.01:
                agg["topFunds"].append(
                    {
                        "fund": fund_name,
                        "positionWeightPct": round(position_weight_pct, 4),
                        "valueUsd": round(h.value_usd, 2),
                    }
                )

    value_added_rank = percentile_rank({ticker: agg["totalValueAddedUsd"] for ticker, agg in ticker_agg.items()})
    current_value_rank = percentile_rank({ticker: agg["totalCurrentValueUsd"] for ticker, agg in ticker_agg.items()})
    concentration_rank = percentile_rank({ticker: agg["maxPositionWeightPct"] for ticker, agg in ticker_agg.items()})

    scores: Dict[str, Dict[str, Any]] = {}

    for ticker, agg in ticker_agg.items():
        num_buyers = agg["numFundsBuying"]
        num_new = agg["numNewPositions"]
        num_sellers = agg["numFundsSelling"]

        breadth_score = min(100.0, num_buyers * 14.0)
        new_position_score = min(100.0, num_new * 22.0)
        value_score = value_added_rank.get(ticker, 0.0)
        existing_ownership_score = current_value_rank.get(ticker, 0.0)
        concentration_score = concentration_rank.get(ticker, 0.0)
        selling_penalty = min(35.0, num_sellers * 8.0)

        smart_money_score = (
            0.30 * breadth_score
            + 0.25 * new_position_score
            + 0.25 * value_score
            + 0.10 * concentration_score
            + 0.10 * existing_ownership_score
            - selling_penalty
        )
        smart_money_score = max(0.0, min(100.0, smart_money_score))

        top_funds = sorted(
            agg["topFunds"],
            key=lambda x: (x["positionWeightPct"], x["valueUsd"]),
            reverse=True,
        )[:8]

        scores[ticker] = {
            "smartMoneyScore": round(smart_money_score, 2),
            "numFundsBuying": int(num_buyers),
            "numNewPositions": int(num_new),
            "numFundsSelling": int(num_sellers),
            "totalValueAddedUsd": round(agg["totalValueAddedUsd"], 2),
            "totalSharesAdded": round(agg["totalSharesAdded"], 4),
            "totalCurrentValueUsd": round(agg["totalCurrentValueUsd"], 2),
            "maxPositionWeightPct": round(agg["maxPositionWeightPct"], 6),
            "buyers": sorted(set(agg["buyers"])),
            "sellers": sorted(set(agg["sellers"])),
            "newPositionFunds": sorted(set(agg["newPositionFunds"])),
            "topFunds": top_funds,
        }

    return scores, raw_rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def holding_to_row(h: Holding) -> Dict[str, Any]:
    return {
        "fundName": h.fund_name,
        "fundCik": h.fund_cik,
        "accession": h.accession,
        "filingDate": h.filing_date,
        "reportDate": h.report_date,
        "cusip": h.cusip,
        "ticker": h.ticker or "",
        "issuer": h.issuer,
        "shares": h.shares,
        "valueUsd": h.value_usd,
        "putCall": h.put_call,
    }


def main() -> None:
    print("Building smart money 13F dataset...")
    print(f"Using User-Agent: {SEC_USER_AGENT}")
    print(f"Minimum 13F position value: ${MIN_VALUE_USD:,.0f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cusip_to_ticker = load_cusip_ticker_map()

    fund_items = list(FUND_CIKS.items())
    if MAX_FUNDS > 0:
        fund_items = fund_items[:MAX_FUNDS]
        print(f"SMART_MONEY_MAX_FUNDS active: only processing {MAX_FUNDS} funds.")

    latest_holdings_by_fund: Dict[str, List[Holding]] = {}
    prior_holdings_by_fund: Dict[str, List[Holding]] = {}
    all_holdings_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for fund_name, cik in fund_items:
        print(f"\nProcessing {fund_name} ({cik})...")

        try:
            filings = get_recent_13f_filings(cik, fund_name, limit=2)

            if len(filings) < 1:
                print("  No recent 13F-HR filings found.")
                failures.append({"fundName": fund_name, "cik": cik, "error": "No 13F filings found"})
                continue

            for idx, filing in enumerate(filings[:2]):
                label = "latest" if idx == 0 else "prior"
                doc = find_information_table_document(filing)

                if not doc:
                    print(f"  Could not find information table XML for {label} filing {filing.accession}.")
                    failures.append(
                        {
                            "fundName": fund_name,
                            "cik": cik,
                            "accession": filing.accession,
                            "error": "No information table XML found",
                        }
                    )
                    continue

                xml_text = download_filing_document(filing, doc)
                holdings = parse_13f_xml(xml_text, filing, cusip_to_ticker)

                print(
                    f"  {label}: {filing.form} report={filing.report_date} "
                    f"filed={filing.filing_date} holdings={len(holdings):,}"
                )

                if idx == 0:
                    latest_holdings_by_fund[fund_name] = holdings
                else:
                    prior_holdings_by_fund[fund_name] = holdings

                all_holdings_rows.extend(holding_to_row(h) for h in holdings)

        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append({"fundName": fund_name, "cik": cik, "error": str(e)})

    ticker_scores, raw_rows = compute_smart_money_scores(
        latest_holdings_by_fund,
        prior_holdings_by_fund,
    )

    unmapped: Dict[str, Dict[str, Any]] = {}
    for h_rows in latest_holdings_by_fund.values():
        for h in h_rows:
            if not h.ticker:
                item = unmapped.setdefault(
                    h.cusip,
                    {
                        "cusip": h.cusip,
                        "issuer": h.issuer,
                        "fundCount": 0,
                        "totalValueUsd": 0.0,
                        "exampleFunds": set(),
                    },
                )
                item["fundCount"] += 1
                item["totalValueUsd"] += h.value_usd
                item["exampleFunds"].add(h.fund_name)

    unmapped_rows = []
    for item in unmapped.values():
        unmapped_rows.append(
            {
                "cusip": item["cusip"],
                "issuer": item["issuer"],
                "fundCount": item["fundCount"],
                "totalValueUsd": round(item["totalValueUsd"], 2),
                "exampleFunds": "; ".join(sorted(item["exampleFunds"])[:10]),
            }
        )

    unmapped_rows.sort(key=lambda r: r["totalValueUsd"], reverse=True)

    as_of_report_dates = sorted(
        {
            h.report_date
            for holdings in latest_holdings_by_fund.values()
            for h in holdings
            if h.report_date
        },
        reverse=True,
    )

    payload = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_report_dates": as_of_report_dates[:5],
        "source": "SEC EDGAR 13F-HR filings",
        "notes": [
            "13F filings are delayed and do not show shorts or full hedge context.",
            "Raw SEC 13F reports securities by CUSIP; ticker output depends on data/cusip_ticker_map.csv.",
            "Passive index managers should generally be excluded from FUND_CIKS.",
        ],
        "fund_count_configured": len(FUND_CIKS),
        "fund_count_processed_latest": len(latest_holdings_by_fund),
        "ticker_count": len(ticker_scores),
        "failure_count": len(failures),
        "failures": failures,
        "data": ticker_scores,
    }

    write_json(OUT_JSON, payload)
    write_csv(OUT_RAW_CSV, raw_rows)
    write_csv(OUT_HOLDINGS_CSV, all_holdings_rows)
    write_csv(OUT_UNMAPPED_CUSIPS_CSV, unmapped_rows)

    print("\nDone.")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_RAW_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_HOLDINGS_CSV.relative_to(ROOT)}")
    print(f"Wrote {OUT_UNMAPPED_CUSIPS_CSV.relative_to(ROOT)}")
    print(f"Ticker scores: {len(ticker_scores):,}")
    print(f"Unmapped CUSIPs: {len(unmapped_rows):,}")

    if not cusip_to_ticker:
        print("\nIMPORTANT:")
        print("No CUSIP map found, so smart_money.json may have little/no ticker-level data.")
        print("Add data/cusip_ticker_map.csv with columns: cusip,ticker")


if __name__ == "__main__":
    main()
