"""
Uploads the current season's match results to Supabase's `results` table, where
the leaderboard (supabase/leaderboard.sql) scores everyone's logged bets.

Reads the results from the site build (generated_site/index.html), so run it after
build_site.py. Needs the project's secret key in SUPABASE_SECRET_KEY (a GitHub
secret -- it bypasses row level security, so it must never be in the page); without
it the step is skipped. Re-uploading is harmless: rows are upserted by match.

    python src/site/sync_results.py
"""

import json
import os
import re
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, "generated_site", "index.html")
SUPABASE_URL = "https://rstembhxuncyttzdmssl.supabase.co"


def site_data():
    html = open(SITE, encoding="utf-8").read()
    raw = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    return json.loads(raw.replace("<\\/", "</"))


def result_rows(data):
    rows = []
    for lg in data["leagues"]:
        for m in lg["matches"]:
            rows.append({
                "league": lg["id"], "season": lg["season"], "home": m["home"], "away": m["away"],
                "kickoff": pd.Timestamp(m["kickoff"]).tz_localize("Europe/London").tz_convert("UTC").isoformat(),
                "result": m["result"], "home_goals": m["score"][0], "away_goals": m["score"][1],
            })
    return rows


def main():
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not key:
        print("Results sync: SUPABASE_SECRET_KEY not set, skipping")
        return 0
    rows = result_rows(site_data())
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/results",
        params={"on_conflict": "league,season,home,away"},
        headers={"apikey": key, "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows, timeout=60,
    )
    if resp.status_code >= 300:
        print(f"::warning::Results sync failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return 1
    print(f"Results sync: uploaded {len(rows)} results to Supabase")
    return 0


if __name__ == "__main__":
    sys.exit(main())
