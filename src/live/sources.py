"""
Everything the round-by-round pipeline pulls fresh on each run:

  - fixture schedule + round numbers   fixturedownload.com (no auth)
  - played-match stats                 football-data.co.uk, same format as data/raw/pl*.csv
  - xG / xGA / deep                    Understat getLeagueData, same JSON as data/cache/leaguedata_*.json
  - Elo ratings                        ClubElo API, cached per date in data/external/elo_df.csv

Each fetch writes to the same cache location the historical data lives in,
and falls back to that cached copy (with a warning) if the source is down,
so one flaky site never blocks a prediction run.
"""

import io
import json
import os
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from process_season_data import load_data  # noqa: E402
from rebuild_rolling_as_ewma import TEAM_MAP as UNDERSTAT_TO_CANONICAL  # noqa: E402

RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "cache")
ELO_FILE = os.path.join(ROOT, "data", "external", "elo_df.csv")

HEADERS = {"User-Agent": "Mozilla/5.0"}
FIXTURES_URL = "https://fixturedownload.com/feed/json/epl-{year}"
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
UNDERSTAT_URL = "https://understat.com/getLeagueData/EPL/{year}"
CLUBELO_URL = "http://api.clubelo.com/{date}"

# Canonical naming is football-data.co.uk's (same as every pl*.csv file).
# Understat names are mapped in rebuild_rolling_as_ewma.TEAM_MAP.
FIXTUREDOWNLOAD_TO_CANONICAL = {"Man Utd": "Man United", "Spurs": "Tottenham"}
CLUBELO_TO_CANONICAL = {"Forest": "Nott'm Forest"}


def now_uk():
    """Current UK wall-clock time (naive), the same clock as fixture kickoffs --
    independent of the machine's timezone (GitHub Actions runs on UTC)."""
    return pd.Timestamp.now(tz="Europe/London").tz_localize(None)


def season_start_year(season):
    """'26-27' -> 2026"""
    return 2000 + int(season.split("-")[0])


def fetch_fixtures(season):
    """All 380 fixtures of `season`, played and unplayed, with round numbers.
    Kickoff is converted from UTC to UK local time so Date/Time line up with
    football-data.co.uk and Understat."""
    url = FIXTURES_URL.format(year=season_start_year(season))
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json())
    kickoff = pd.to_datetime(df["DateUtc"], utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None)
    df["Kickoff"] = kickoff
    df["Date"] = kickoff.dt.normalize()
    df["Time"] = kickoff.dt.strftime("%H:%M")
    df["HomeTeam"] = df["HomeTeam"].replace(FIXTUREDOWNLOAD_TO_CANONICAL)
    df["AwayTeam"] = df["AwayTeam"].replace(FIXTUREDOWNLOAD_TO_CANONICAL)
    df["Played"] = df["HomeTeamScore"].notna() & df["AwayTeamScore"].notna()
    return df[["RoundNumber", "Kickoff", "Date", "Time", "HomeTeam", "AwayTeam",
               "HomeTeamScore", "AwayTeamScore", "Played"]].sort_values(
        ["RoundNumber", "Kickoff"]).reset_index(drop=True)


def fetch_played_matches(season):
    """load_data()-shaped frame of this season's played matches, or None if
    football-data.co.uk hasn't published the file yet (normal before the
    first match)."""
    a, b = season.split("-")
    url = FOOTBALL_DATA_URL.format(code=f"{a}{b}")
    cache_path = os.path.join(RAW_DIR, f"pl{season}_live.csv")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        text = resp.content.decode("utf-8-sig")
        if resp.status_code != 200 or "HomeTeam" not in text[:1000]:
            print(f"  No match file published yet at {url}")
            return None
        with open(cache_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    except requests.RequestException as e:
        if not os.path.exists(cache_path):
            raise
        print(f"  WARNING: football-data.co.uk unreachable ({e}); using cached {cache_path}")

    df = load_data(cache_path)
    print(f"  football-data.co.uk: {len(df)} played matches")
    return df


def fetch_understat(season):
    """Refreshes data/cache/leaguedata_<year>.json for the live season.
    Returns DataFrame[Team, Date_str] of every team-match Understat has xG
    for (canonical team names), so the caller can check it's up to date."""
    year = season_start_year(season)
    cache_path = os.path.join(CACHE_DIR, f"leaguedata_{year}.json")
    try:
        resp = requests.get(UNDERSTAT_URL.format(year=year),
                            headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except (requests.RequestException, ValueError) as e:
        if not os.path.exists(cache_path):
            print(f"  WARNING: Understat unreachable ({e}) and no cache -- live-season xG will be missing")
            return pd.DataFrame(columns=["Team", "Date_str"])
        print(f"  WARNING: Understat unreachable ({e}); using cached {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

    rows = [
        {"Team": UNDERSTAT_TO_CANONICAL.get(t["title"], t["title"]),
         "Date_str": pd.to_datetime(m["date"]).strftime("%Y-%m-%d")}
        for t in data["teams"].values() for m in t["history"]
    ]
    print(f"  Understat: xG for {len(rows) // 2} played matches")
    return pd.DataFrame(rows, columns=["Team", "Date_str"])


def _fetch_clubelo_day(date_str):
    resp = requests.get(CLUBELO_URL.format(date=date_str), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    day = pd.read_csv(io.StringIO(resp.text))
    day = day[day["Country"] == "ENG"]
    return pd.DataFrame({
        "team": day["Club"].replace(CLUBELO_TO_CANONICAL),
        "rank": pd.to_numeric(day["Rank"], errors="coerce"),
        "country": day["Country"],
        "level": day["Level"],
        "elo": day["Elo"],
        "from": day["From"],
        "to": day["To"],
        "league": pd.NA,
        "QueryDate": date_str,
    })


def update_elo_cache(dates):
    """Fetches ClubElo ratings (all English clubs) for every date in `dates`
    not already in elo_df.csv, appends them, and returns the full cache.
    Ratings for a past date never change, so cached dates are not refetched.
    A failed date is simply left out (and retried next run) -- features.py
    then falls back to each team's last known rating."""
    elo = pd.read_csv(ELO_FILE)
    missing = sorted(set(dates) - set(elo["QueryDate"]))
    if not missing:
        print("  ClubElo: all needed dates already cached")
        return elo

    frames, failed = [], []
    for d in missing:
        try:
            frames.append(_fetch_clubelo_day(d))
        except (requests.RequestException, pd.errors.ParserError, KeyError) as e:
            failed.append(d)
            if len(failed) == 1:
                print(f"  ClubElo fetch failed for {d}: {e}")
            # Server down rather than one bad date -- don't wait out a timeout per date.
            if len(failed) >= 2 and not frames:
                failed += [x for x in missing if x not in failed and x > d]
                break

    if frames:
        elo = pd.concat([elo] + frames, ignore_index=True)
        elo.to_csv(ELO_FILE, index=False)
    print(f"  ClubElo: fetched {len(frames)}/{len(missing)} new dates"
          + (f"; FAILED {len(failed)} (falling back to last known ratings)" if failed else ""))
    return elo
