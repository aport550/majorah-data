import os
import math
import requests
import yfinance as yf
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

if not SUPABASE_URL:
    raise Exception("Missing SUPABASE_URL")

if not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_SERVICE_ROLE_KEY")

BASE_URL = f"{SUPABASE_URL}/rest/v1/convictions"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Prefer": "return=representation",
}

TIMEOUT = 30


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except Exception:
        return None


def load_open_convictions():
    url = (
        f"{BASE_URL}"
        "?select=id,ticker,start_date,end_date,start_price,latest_price,close_price,is_closed,closed_at,notes"
        "&is_closed=eq.false"
    )
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    print("Load status:", r.status_code)
    print("Load response preview:", r.text[:500])
    r.raise_for_status()
    return r.json()


def update_conviction(conviction_id, payload):
    encoded_id = quote(str(conviction_id), safe="")
    url = f"{BASE_URL}?id=eq.{encoded_id}"

    r = requests.patch(url, headers=HEADERS, json=payload, timeout=TIMEOUT)

    print(f"Update {conviction_id} status:", r.status_code)
    print("Update response preview:", r.text[:500])

    r.raise_for_status()
    rows = r.json()
    if not rows:
        print(f"WARNING: no row matched id={conviction_id}")
    return rows


def fetch_latest_price_live(ticker):
    """
    Current/live-ish price for marking open convictions.
    NOT used for entry or expiry fills.
    """
    try:
        t = yf.Ticker(ticker)

        try:
            fast = getattr(t, "fast_info", None)
            if fast:
                price = fast.get("lastPrice") or fast.get("last_price")
                if price is not None and math.isfinite(float(price)):
                    return float(price)
        except Exception as e:
            print(f"{ticker}: fast_info failed: {e}")

        for period, interval in [
            ("1d", "1m"),
            ("5d", "15m"),
            ("1mo", "1d"),
        ]:
            try:
                hist = t.history(period=period, interval=interval, auto_adjust=False)
                if hist is not None and not hist.empty:
                    closes = hist["Close"].dropna()
                    if not closes.empty:
                        price = float(closes.iloc[-1])
                        if math.isfinite(price):
                            return price
            except Exception as e:
                print(f"{ticker}: history {period}/{interval} failed: {e}")

        try:
            info = t.info
            price = info.get("regularMarketPrice")
            if price is not None and math.isfinite(float(price)):
                return float(price)
        except Exception as e:
            print(f"{ticker}: info failed: {e}")

    except Exception as e:
        print(f"{ticker}: fetch_latest_price_live fatal error: {e}")

    return None


def fetch_open_price_on_or_after_date(ticker, target_date_str, max_days_forward=7):
    """
    Returns the first available daily OPEN on or after target_date.
    This makes the script resilient if target_date lands on a weekend/holiday.

    Returns:
        (open_price: float | None, actual_trade_date: str | None)
    """
    target_date = parse_date(target_date_str)
    if not target_date:
        print(f"{ticker}: invalid target date {target_date_str}")
        return None, None

    fetch_start = target_date.isoformat()
    fetch_end = (target_date + timedelta(days=max_days_forward)).isoformat()

    try:
        t = yf.Ticker(ticker)
        hist = t.history(
            start=fetch_start,
            end=fetch_end,
            interval="1d",
            auto_adjust=False,
        )

        if hist is None or hist.empty:
            print(f"{ticker}: no history returned for window {fetch_start} -> {fetch_end}")
            return None, None

        opens = hist["Open"].dropna()
        if opens.empty:
            print(f"{ticker}: history returned but no valid open prices")
            return None, None

        first_ts = opens.index[0]
        actual_trade_date = first_ts.date().isoformat()
        open_price = float(opens.iloc[0])

        if math.isfinite(open_price):
            return open_price, actual_trade_date

    except Exception as e:
        print(f"{ticker}: fetch_open_price_on_or_after_date failed: {e}")

    return None, None


def main():
    print("Starting conviction update job...")
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    print("UTC now:", now_utc.isoformat())

    convictions = load_open_convictions()
    print(f"Loaded {len(convictions)} open convictions")

    for c in convictions:
        cid = c["id"]
        ticker = (c.get("ticker") or "").strip().upper()

        if not ticker:
            print(f"Skipping {cid}: missing ticker")
            continue

        start_date = parse_date(c.get("start_date"))
        end_date = parse_date(c.get("end_date"))
        existing_start_price = to_float(c.get("start_price"))
        existing_close_price = to_float(c.get("close_price"))

        print(f"\nProcessing {ticker} ({cid})")
        print(
            f"{ticker}: start_date={start_date}, end_date={end_date}, "
            f"start_price={existing_start_price}, close_price={existing_close_price}"
        )

        payload = {}

        # 1) ENTRY FILL
        # Fill ONLY from historical OPEN on/after stored start_date.
        # Never use today's live price as a substitute.
        if existing_start_price is None:
            if start_date is None:
                print(f"{ticker}: no start_date, cannot fill entry")
            elif today_utc >= start_date:
                entry_open, actual_entry_trade_date = fetch_open_price_on_or_after_date(
                    ticker, start_date.isoformat()
                )
                if entry_open is not None:
                    payload["start_price"] = entry_open
                    existing_start_price = entry_open
                    print(
                        f"{ticker}: entry filled at {entry_open} "
                        f"using market open on {actual_entry_trade_date}"
                    )
                else:
                    print(f"{ticker}: entry open not available yet for start_date={start_date}")
            else:
                print(f"{ticker}: entry not due yet")

        # 2) EXIT FILL
        # If expiry date has arrived/passed, close ONLY from the
        # historical OPEN on/after stored end_date.
        should_attempt_close = end_date is not None and today_utc >= end_date

        if should_attempt_close and existing_close_price is None:
            close_open, actual_close_trade_date = fetch_open_price_on_or_after_date(
                ticker, end_date.isoformat()
            )
            if close_open is not None:
                payload["close_price"] = close_open
                payload["latest_price"] = close_open
                payload["is_closed"] = True
                payload["closed_at"] = now_utc.isoformat()

                print(
                    f"{ticker}: conviction closed at {close_open} "
                    f"using market open on {actual_close_trade_date}"
                )
            else:
                print(f"{ticker}: close open not available yet for end_date={end_date}")

        # 3) LIVE MARK UPDATE FOR STILL-OPEN POSITIONS
        # Only do this if entry has already been filled and the conviction
        # is not being closed in this run.
        is_closing_now = payload.get("is_closed") is True

        if existing_start_price is not None and not is_closing_now:
            live_price = fetch_latest_price_live(ticker)
            if live_price is not None:
                payload["latest_price"] = live_price
                print(f"{ticker}: latest live mark set to {live_price}")
            else:
                print(f"{ticker}: could not fetch live latest price")

        # 4) WRITE IF THERE IS SOMETHING TO UPDATE
        if payload:
            payload["notes"] = f"Updated at {now_utc.isoformat()}"

            try:
                rows = update_conviction(cid, payload)
                if rows:
                    row = rows[0]
                    print(
                        f"{ticker}: updated row -> "
                        f"start_price={row.get('start_price')}, "
                        f"latest_price={row.get('latest_price')}, "
                        f"close_price={row.get('close_price')}, "
                        f"is_closed={row.get('is_closed')}"
                    )
            except Exception as e:
                print(f"{ticker}: update failed: {e}")
        else:
            print(f"{ticker}: nothing to update this run")

    print("\nDone.")


if __name__ == "__main__":
    main()
