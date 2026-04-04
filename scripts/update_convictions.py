import os
import requests
from datetime import datetime, timezone

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
}

TIMEOUT = 30


def load_convictions():
    url = f"{BASE_URL}?select=*"
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

    print("Load status:", r.status_code)
    print("Load response preview:", r.text[:300])

    r.raise_for_status()
    return r.json()


def update_conviction(conviction_id, payload):
    url = f"{BASE_URL}?id=eq.{conviction_id}"
    r = requests.patch(url, headers=HEADERS, json=payload, timeout=TIMEOUT)

    print(f"Update {conviction_id} status:", r.status_code)
    if r.status_code not in (200, 204):
        print("Update failed response:", r.text[:300])
        r.raise_for_status()


def main():
    print("Starting conviction update job...")

    print("URL present:", bool(SUPABASE_URL))
    print("KEY present:", bool(SUPABASE_KEY))
    print("KEY first char:", repr(SUPABASE_KEY[:1]))
    print("KEY last char:", repr(SUPABASE_KEY[-1:]))

    convictions = load_convictions()
    print(f"Loaded {len(convictions)} convictions")

    for c in convictions:
        cid = c["id"]
        ticker = c.get("ticker", "UNKNOWN")

        print(f"Processing {ticker} ({cid})")

        update_conviction(
            cid,
            {
                "notes": f"Updated at {datetime.now(timezone.utc).isoformat()}"
            },
        )

    print("Done.")


if __name__ == "__main__":
    main()
