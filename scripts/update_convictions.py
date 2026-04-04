import os
import requests
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing Supabase env vars")

BASE_URL = f"{SUPABASE_URL}/rest/v1/convictions"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def load_convictions():
    url = f"{BASE_URL}?select=*"
    r = requests.get(url, headers=HEADERS)

    print("Status:", r.status_code)
    print("Response:", r.text[:300])

    r.raise_for_status()
    return r.json()


def update_conviction(conviction_id, payload):
    url = f"{BASE_URL}?id=eq.{conviction_id}"
    r = requests.patch(url, headers=HEADERS, json=payload)

    if r.status_code not in [200, 204]:
        print("Update failed:", r.text)
        raise Exception("Update failed")


def main():
    print("Starting conviction update job...")

    convictions = load_convictions()
    print(f"Loaded {len(convictions)} convictions")

    for c in convictions:
        cid = c["id"]
        ticker = c["ticker"]

        print(f"Processing {ticker} ({cid})")

        # TEMP TEST UPDATE (just to verify system works)
        update_conviction(cid, {
            "notes": f"Updated at {datetime.utcnow().isoformat()}"
        })

    print("Done.")


if __name__ == "__main__":
    main()
