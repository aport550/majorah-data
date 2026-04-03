import os
import math
import traceback
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()

NY_TZ = ZoneInfo("America/New_York")


def get_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


SUPABASE_URL = get_env("SUPABASE_URL")
# Support either old service_role or newer secret key naming.
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_SECRET_KEY")
    or os.getenv("SUPABASE_SERVICE_KEY")
)

if not SUPABASE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SECRET_KEY in environment."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def today_ny() -> date:
    return datetime.now(NY_TZ).date()


def parse_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    return None


def parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def is_finite_number(value) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except Exception:
        return False


def safe_float(value) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def fetch_history(ticker: str, start_date: date, end_date: date):
    """
    Yahoo end is exclusive, so caller should pass end_date + 1 day
    when they want to include end_date itself.
    """
    if end_date <= start_date:
        end_date = start_date + timedelta(days=1)

    df = yf.download(
        tickers=ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return df


def first_trading_open_on_or_after(ticker: str, target_date: date) -> tuple[date | None, float | None]:
    """
    Find the first trading day on or after target_date and return its open.
    Searches forward up to 14 calendar days to handle weekends/holidays.
    """
    search_end = target_date + timedelta(days=14)
    df = fetch_history(ticker, target_date, search_end + timedelta(days=1))

    if df is None or df.empty:
        return None, None

    for idx, row in df.iterrows():
        row_date = idx.date()
        open_price = safe_float(row.get("Open"))
        if row_date >= target_date and open_price is not None:
            return row_date, open_price

    return None, None


def last_close_on_or_before(ticker: str, target_date: date) -> tuple[date | None, float | None]:
    """
    Find the last trading day on or before target_date and return its close.
    Searches backward 14 calendar days to handle weekends/holidays.
    """
    search_start = target_date - timedelta(days=14)
    df = fetch_history(ticker, search_start, target_date + timedelta(days=1))

    if df is None or df.empty:
        return None, None

    df = df.sort_index()
    valid_rows = []

    for idx, row in df.iterrows():
        row_date = idx.date()
        close_price = safe_float(row.get("Close"))
        if row_date <= target_date and close_price is not None:
            valid_rows.append((row_date, close_price))

    if not valid_rows:
        return None, None

    return valid_rows[-1]


def most_recent_close(ticker: str) -> tuple[date | None, float | None]:
    """
    Get the most recent available daily close from the last ~10 calendar days.
    """
    end_date = today_ny()
    start_date = end_date - timedelta(days=10)
    df = fetch_history(ticker, start_date, end_date + timedelta(days=1))

    if df is None or df.empty:
        return None, None

    df = df.sort_index()
    for idx in reversed(df.index):
        row = df.loc[idx]
        close_price = safe_float(row.get("Close"))
        if close_price is not None:
            return idx.date(), close_price

    return None, None


def load_convictions() -> list[dict]:
    response = (
        supabase.table("convictions")
        .select(
            "id,user_id,ticker,confidence,timeline,rationale,created_at,start_date,end_date,start_price,latest_price,is_closed,closed_at"
        )
        .execute()
    )
    return response.data or []


def update_conviction(conviction_id: str, payload: dict) -> None:
    if not payload:
        return
    supabase.table("convictions").update(payload).eq("id", conviction_id).execute()


def process_conviction(row: dict) -> dict:
    conviction_id = row["id"]
    ticker = (row.get("ticker") or "").strip().upper()
    start_date = parse_date(row.get("start_date"))
    end_date = parse_date(row.get("end_date"))
    start_price = safe_float(row.get("start_price"))
    latest_price = safe_float(row.get("latest_price"))
    is_closed = bool(row.get("is_closed"))
    closed_at = parse_datetime(row.get("closed_at"))
    today = today_ny()

    result = {
        "id": conviction_id,
        "ticker": ticker,
        "updated": False,
        "notes": [],
        "error": None,
    }

    if not ticker:
        result["error"] = "Missing ticker"
        return result

    if start_date is None or end_date is None:
        result["error"] = "Missing start_date or end_date"
        return result

    payload = {}

    # 1) Ensure start_price is set from the first trading day's OPEN on/after start_date,
    # and normalize start_date to that actual trading date if needed.
    if start_price is None:
        actual_start_date, actual_open = first_trading_open_on_or_after(ticker, start_date)
        if actual_start_date is not None and actual_open is not None:
            payload["start_price"] = actual_open
            if actual_start_date != start_date:
                payload["start_date"] = actual_start_date.isoformat()
                start_date = actual_start_date
                result["notes"].append(
                    f"Adjusted start_date to actual trading day {actual_start_date.isoformat()}"
                )
            start_price = actual_open
            result["notes"].append(f"Set start_price={actual_open:.4f}")
        else:
            result["notes"].append("Could not find first trading day/open yet")

    # 2) If conviction has expired, freeze latest_price to final close on/before end_date
    # and mark closed.
    if end_date < today:
        final_date, final_close = last_close_on_or_before(ticker, end_date)

        if final_date is not None and final_close is not None:
            payload["latest_price"] = final_close
            latest_price = final_close
            result["notes"].append(
                f"Set final latest_price={final_close:.4f} using close from {final_date.isoformat()}"
            )
        else:
            result["notes"].append("Could not find final close on/before end_date")

        if not is_closed:
            payload["is_closed"] = True
            if closed_at is None:
                payload["closed_at"] = datetime.now(NY_TZ).isoformat()
            result["notes"].append("Marked conviction closed")

    # 3) If conviction is still open and has started, update latest_price to most recent close.
    elif start_date <= today:
        recent_date, recent_close = most_recent_close(ticker)
        if recent_date is not None and recent_close is not None:
            payload["latest_price"] = recent_close
            latest_price = recent_close
            result["notes"].append(
                f"Updated latest_price={recent_close:.4f} using close from {recent_date.isoformat()}"
            )
        else:
            result["notes"].append("Could not get recent close")

    # 4) Persist any updates.
    if payload:
        update_conviction(conviction_id, payload)
        result["updated"] = True

    return result


def main():
    print("Starting conviction update job...")
    convictions = load_convictions()
    print(f"Loaded {len(convictions)} convictions")

    updated_count = 0
    error_count = 0

    for row in convictions:
        conviction_id = row.get("id")
        ticker = row.get("ticker")
        try:
            result = process_conviction(row)
            prefix = f"[{ticker} | {conviction_id}]"
            if result["error"]:
                error_count += 1
                print(f"{prefix} ERROR: {result['error']}")
            else:
                if result["updated"]:
                    updated_count += 1
                    print(f"{prefix} UPDATED")
                else:
                    print(f"{prefix} NO CHANGE")

                for note in result["notes"]:
                    print(f"  - {note}")

        except Exception as exc:
            error_count += 1
            print(f"[{ticker} | {conviction_id}] EXCEPTION: {exc}")
            traceback.print_exc()

    print("Done.")
    print(f"Updated: {updated_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
