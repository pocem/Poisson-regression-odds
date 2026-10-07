"""
Everything the round-by-round pipeline pulls fresh on each run, per league
(league settings and file locations: src/leagues.py):

  - fixture schedule + round numbers   fixturedownload.com (no auth)
  - played-match stats                 football-data.co.uk, same format as the league's raw/ season files
  - xG / xGA / deep                    Understat getLeagueData, cached as cache/understat_<year>.json
  - Elo ratings                        ClubElo API, cached per date in external/elo_df.csv

Each fetch writes to the league's own folder and falls back to that cached
copy (with a warning) if the source is down, so one flaky site never blocks a
prediction run.
"""

import io
import json
import os
import sys

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from process_season_data import load_data  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0"}
FIXTURES_URL = "https://fixturedownload.com/feed/json/{slug}"
# Until a league announces exact kickoff times (the Bundesliga does so only a
# few weeks ahead), fixturedownload.com lists the round at 00:00 UTC on the
# Friday of its weekend. Such fixtures are flagged KickoffTBC and count as
# upcoming until TBC_WINDOW_DAYS after that date.
TBC_WINDOW_DAYS = 4
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/{code}/{division}.csv"
UNDERSTAT_URL = "https://understat.com/getLeagueData/{league}/{year}"
CLUBELO_URL = "http://api.clubelo.com/{date}"


def now_uk():
    """Current UK wall-clock time (naive), the same clock as fixture kickoffs --
    independent of the machine's timezone (GitHub Actions runs on UTC)."""
    return pd.Timestamp.now(tz="Europe/London").tz_localize(None)


def season_start_year(season):
    """'26-27' -> 2026"""
    return 2000 + int(season.split("-")[0])


def fetch_fixtures(league, season):
    """Every fixture of `season`, played and unplayed, with round numbers.
    Kickoff is converted from UTC to UK time -- the clock football-data.co.uk
    uses for every league -- so Date/Time line up across sources."""
    url = FIXTURES_URL.format(slug=league.fixtures_slug.format(year=season_start_year(season)))
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json())
    utc = pd.to_datetime(df["DateUtc"], utc=True)
    kickoff = utc.dt.tz_convert("Europe/London").dt.tz_localize(None)
    df["Kickoff"] = kickoff
    df["Date"] = kickoff.dt.normalize()
    df["Time"] = kickoff.dt.strftime("%H:%M")
    df["HomeTeam"] = df["HomeTeam"].replace(league.fixturedownload_names)
    df["AwayTeam"] = df["AwayTeam"].replace(league.fixturedownload_names)
    df["Played"] = df["HomeTeamScore"].notna() & df["AwayTeamScore"].notna()
    df["KickoffTBC"] = ~df["Played"] & (utc.dt.hour == 0) & (utc.dt.minute == 0)
    return df[["RoundNumber", "Kickoff", "KickoffTBC", "Date", "Time", "HomeTeam", "AwayTeam",
               "HomeTeamScore", "AwayTeamScore", "Played"]].sort_values(
        ["RoundNumber", "Kickoff"]).reset_index(drop=True)


def download_season_file(league, season, path):
    """Downloads a football-data.co.uk season file to `path`. Returns False if
    it isn't published (yet)."""
    a, b = season.split("-")
    url = FOOTBALL_DATA_URL.format(code=f"{a}{b}", division=league.football_data_code)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    text = resp.content.decode("utf-8-sig")
    if resp.status_code != 200 or "HomeTeam" not in text[:1000]:
        print(f"  No match file published yet at {url}")
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


def fetch_played_matches(league, season):
    """load_data()-shaped frame of this season's played matches, or None if
    football-data.co.uk hasn't published the file yet (normal before the
    first match)."""
    cache_path = league.live_raw_file(season)
    try:
        if not download_season_file(league, season, cache_path):
            return None
    except requests.RequestException as e:
        if not os.path.exists(cache_path):
            raise
        print(f"  WARNING: football-data.co.uk unreachable ({e}); using cached {cache_path}")

    df = load_data(cache_path)
    print(f"  football-data.co.uk: {len(df)} played matches")
    return df


def understat_rows(league, data):
    """Understat league JSON -> [{Team (canonical), Date_str, xG, xGA, deep, deep_allowed}]."""
    return [
        {"Team": league.understat_names.get(t["title"], t["title"]),
         "Date_str": pd.to_datetime(m["date"]).strftime("%Y-%m-%d"),
         "xG": m["xG"], "xGA": m["xGA"], "deep": m["deep"], "deep_allowed": m["deep_allowed"]}
        for t in data["teams"].values() for m in t["history"]
    ]


XG_DATE_TOLERANCE_DAYS = 2


def align_xg_dates(xg, match_keys):
    """Understat occasionally files a match under a neighbouring date (e.g.
    St. Pauli v Holstein Kiel, played Friday 2024-11-29, is listed on the 30th).
    Moves such an xG row onto the team's football-data.co.uk match date when the
    two are at most XG_DATE_TOLERANCE_DAYS apart and neither has a partner.
    xg: DataFrame with Team, Date_str; match_keys: iterable of (team, 'YYYY-MM-DD')."""
    match_keys = set(match_keys)
    have = set(zip(xg["Team"], xg["Date_str"]))
    unmatched = {}
    for team, date in match_keys - have:
        unmatched.setdefault(team, []).append(pd.Timestamp(date))
    if not unmatched:
        return xg
    xg = xg.copy()
    for idx, team, date in zip(xg.index, xg["Team"], xg["Date_str"]):
        if (team, date) in match_keys or team not in unmatched:
            continue
        d = pd.Timestamp(date)
        near = [c for c in unmatched[team] if abs((c - d).days) <= XG_DATE_TOLERANCE_DAYS]
        if near:
            best = min(near, key=lambda c: abs((c - d).days))
            xg.at[idx, "Date_str"] = best.strftime("%Y-%m-%d")
            unmatched[team].remove(best)
    return xg


def download_understat(league, year):
    resp = requests.get(UNDERSTAT_URL.format(league=league.understat_name, year=year),
                        headers={**HEADERS, "X-Requested-With": "XMLHttpRequest"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    path = league.understat_cache(year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_understat(league, season):
    """Refreshes the live season's Understat cache. Returns DataFrame[Team,
    Date_str] of every team-match Understat has xG for (canonical names), so
    the caller can check it's up to date."""
    year = season_start_year(season)
    cache_path = league.understat_cache(year)
    try:
        data = download_understat(league, year)
    except (requests.RequestException, ValueError) as e:
        if not os.path.exists(cache_path):
            print(f"  WARNING: Understat unreachable ({e}) and no cache -- live-season xG will be missing")
            return pd.DataFrame(columns=["Team", "Date_str"])
        print(f"  WARNING: Understat unreachable ({e}); using cached {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

    rows = understat_rows(league, data)
    print(f"  Understat: xG for {len(rows) // 2} played matches")
    return pd.DataFrame(rows, columns=["Team", "Date_str"])


def _fetch_clubelo_day(league, date_str):
    resp = requests.get(CLUBELO_URL.format(date=date_str), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    day = pd.read_csv(io.StringIO(resp.text))
    day = day[day["Country"] == league.clubelo_country]
    return pd.DataFrame({
        "team": day["Club"].replace(league.clubelo_names),
        "rank": pd.to_numeric(day["Rank"], errors="coerce"),
        "country": day["Country"],
        "level": day["Level"],
        "elo": day["Elo"],
        "from": day["From"],
        "to": day["To"],
        "league": pd.NA,
        "QueryDate": date_str,
    })


def update_elo_cache(league, dates):
    """Fetches ClubElo ratings (every club in the league's country) for each
    date in `dates` not already in the league's elo_df.csv, appends them, and
    returns the full cache. Ratings for a past date never change, so cached
    dates are not refetched. A failed date is simply left out (and retried
    next run) -- seeding then falls back to the latest cached rating."""
    elo = pd.read_csv(league.elo_file)
    missing = sorted(set(dates) - set(elo["QueryDate"]))
    if not missing:
        return elo

    frames, failed = [], []
    for d in missing:
        try:
            frames.append(_fetch_clubelo_day(league, d))
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
        elo.to_csv(league.elo_file, index=False)
    print(f"  ClubElo: fetched {len(frames)}/{len(missing)} new dates"
          + (f"; {len(failed)} unavailable (using the latest cached ratings)" if failed else ""))
    return elo
