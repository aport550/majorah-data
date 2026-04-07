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


def load_open_convictions():
    url = (
        f"{BASE_URL}"
        "?select=id,ticker,start_date,end_date,start_price,latest_price,is_closed,notes"
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


def fetch_latest_price(ticker):
    try:
        t = yf.Ticker(ticker)

        # Try fast_info first
        try:
            fast = getattr(t, "fast_info", None)
            if fast:
                price = fast.get("lastPrice") or fast.get("last_price")
                if price is not None and math.isfinite(float(price)):
                    return float(price)
        except Exception as e:
            print(f"{ticker}: fast_info failed: {e}")

        # Then recent history
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

        # Fallback to info
        try:
            info = t.info
            price = info.get("regularMarketPrice")
            if price is not None and math.isfinite(float(price)):
                return float(price)
        except Exception as e:
            print(f"{ticker}: info failed: {e}")

    except Exception as e:
        print(f"{ticker}: fetch_latest_price fatal error: {e}")

    return None


def fetch_start_price_for_date(ticker, start_date_str):
    """
    Gets the market close on or just after the conviction start_date.
    """
    if not start_date_str:
        return None

    try:
        start_dt = datetime.fromisoformat(start_date_str).date()
    except Exception:
        print(f"{ticker}: invalid start_date {start_date_str}")
        return None

    # yfinance end date is exclusive, so add a few days to survive weekends/holidays
    fetch_start = start_dt.isoformat()
    fetch_end = (start_dt + timedelta(days=7)).isoformat()

    try:
        t = yf.Ticker(ticker)
        hist = t.history(
            start=fetch_start,
            end=fetch_end,
            interval="1d",
            auto_adjust=False,
        )

        if hist is None or hist.empty:
            print(f"{ticker}: no history returned for start window {fetch_start} -> {fetch_end}")
            return None

        closes = hist["Close"].dropna()
        if closes.empty:
            print(f"{ticker}: history returned but no valid close prices")
            return None

        # first available trading day on/after start_date
        price = float(closes.iloc[0])
        if math.isfinite(price):
            return price

    except Exception as e:
        print(f"{ticker}: fetch_start_price_for_date failed: {e}")

    return None


def main():
    print("Starting conviction update job...")
    now_utc = datetime.now(timezone.utc).isoformat()
    print("UTC now:", now_utc)

    convictions = load_open_convictions()
    print(f"Loaded {len(convictions)} open convictions")

    for c in convictions:
        cid = c["id"]
        ticker = (c.get("ticker") or "").strip().upper()
        start_date = c.get("start_date")
        existing_start_price = to_float(c.get("start_price"))

        if not ticker:
            print(f"Skipping {cid}: missing ticker")
            continue

        print(f"\nProcessing {ticker} ({cid})")

        payload = {
            "notes": f"Updated at {datetime.now(timezone.utc).isoformat()}",
        }

        # Set start_price only if missing
        if existing_start_price is None:
            start_price = fetch_start_price_for_date(ticker, start_date)
            if start_price is not None:
                payload["start_price"] = start_price
                print(f"{ticker}: setting start_price = {start_price} from start_date {start_date}")
            else:
                print(f"{ticker}: could not determine start_price from start_date {start_date}")
        else:
            print(f"{ticker}: existing start_price = {existing_start_price}")

        # Always refresh latest_price
        latest_price = fetch_latest_price(ticker)
        if latest_price is not None:
            payload["latest_price"] = latest_price
            print(f"{ticker}: setting latest_price = {latest_price}")
        else:
            print(f"{ticker}: could not fetch latest_price")

        # Only update if we have more than just notes
        if len(payload) > 1:
            try:
                rows = update_conviction(cid, payload)
                if rows:
                    row = rows[0]
                    print(
                        f"{ticker}: updated row start_price={row.get('start_price')} "
                        f"latest_price={row.get('latest_price')}"
                    )
            except Exception as e:
                print(f"{ticker}: update failed: {e}")
        else:
            print(f"{ticker}: no price fields available to update")

    print("\nDone.")


if __name__ == "__main__":
    main()
