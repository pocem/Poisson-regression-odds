"""
Uploads to Supabase, after each site build:
  - the current season's results (`results`), which the leaderboard
    (supabase/leaderboard.sql) scores everyone's logged bets against;
  - the Bet365 odds and model chances of the upcoming matches exactly as the site
    shows them (`match_odds`): new bets are locked to these odds
    (supabase/locked_odds.sql).

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


def odds_rows(data):
    rows = []
    for lg in data["leagues"]:
        for g in (lg["next"]["games"] if lg["next"] else []):
            if not g["b365"] or any(o is None for o in g["b365"]):
                continue
            kickoff = pd.Timestamp(g["kickoff_utc"])
            # A "kickoff TBC" fixture stays open until its weekend is over, as on the page.
            locks = kickoff + pd.Timedelta(days=4) if g["kickoff_tbc"] else kickoff
            rows.append({
                "league": lg["id"], "season": lg["season"], "home": g["home"], "away": g["away"],
                "round": g["round"], "kickoff": kickoff.isoformat(), "locks_at": locks.isoformat(),
                "odds_h": g["b365"][0], "odds_d": g["b365"][1], "odds_a": g["b365"][2],
                "model_h": round(g["p_model"][0], 4), "model_d": round(g["p_model"][1], 4),
                "model_a": round(g["p_model"][2], 4),
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            })
    return rows


def upsert(key, table, rows):
    if not rows:
        return True
    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params={"on_conflict": "league,season,home,away"},
        headers={"apikey": key, "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=rows, timeout=60,
    )
    if resp.status_code >= 300:
        print(f"::warning::Upload to {table} failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return False
    print(f"Supabase sync: uploaded {len(rows)} rows to {table}")
    return True


def main():
    key = os.environ.get("SUPABASE_SECRET_KEY")
    if not key:
        print("Results sync: SUPABASE_SECRET_KEY not set, skipping")
        return 0
    data = site_data()
    ok = upsert(key, "results", result_rows(data))
    ok = upsert(key, "match_odds", odds_rows(data)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
