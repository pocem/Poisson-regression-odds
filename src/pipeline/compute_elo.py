"""
Self-computed Elo ratings, replacing per-match-date ClubElo lookups.

Each team is seeded at the start of every SPELL it has in our data -- its
first match, and again whenever it comes back after relegation (no match for
over SPELL_GAP_DAYS). E.g. Man City once on 2023-08-11; Ipswich at the start
of 24-25 and again at the start of 26-27. Between seeds every rating is
computed here, match by match, with the standard Elo update:

    E_home = 1 / (1 + 10 ** ((R_away - (R_home + HOME_ADVANTAGE)) / 400))
    S_home = 1 win / 0.5 draw / 0 loss
    R_home += K * (S_home - E_home)
    R_away -= K * (S_home - E_home)

Home_Elo / Away_Elo of a match are the ratings BEFORE it is played.

Seed = the team's ClubElo rating going into its first match of the spell --
the same rule in every league. Sources, in order (files are per league, in
data/leagues/<league>/external/):
  1. elo_seeds_manual.csv (team, season, elo) -- hand-entered ratings, always win,
  2. the ClubElo cache (elo_df.csv): the rating whose validity range (from..to)
     covers the day before the first match -- ClubElo only changes a rating
     when the team plays, so this is exactly its pre-match rating, whichever
     date the table was downloaded on,
  3. the cache row downloaded on the first match date itself,
  4. that date fetched from ClubElo (only if a fetch function is passed in),
  5. FALLBACK, reported as a warning: the team's latest cached rating before
     that date (it may have changed since, e.g. through cup matches),
  6. FALLBACK: the average seed of the promoted spells, so a team with no
     rating at all still gets a sensible value.

Run directly to recompute Home_Elo / Away_Elo in a league's processed dataset:
    python src/pipeline/compute_elo.py --league premier_league     (or: all)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from process_season_data import load_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leagues import leagues_from_arg  # noqa: E402

K = 20
HOME_ADVANTAGE = 0
SPELL_GAP_DAYS = 200  # longer than a summer break, shorter than a season away


def season_of(date_str):
    """'2026-08-21' -> '26-27' (seasons run August to May)."""
    d = pd.Timestamp(date_str)
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y % 100:02d}-{(y + 1) % 100:02d}"


def spell_starts(matches):
    """[(team, 'YYYY-MM-DD')] -- each team's first match, plus its first match
    after every gap longer than SPELL_GAP_DAYS (a return from relegation)."""
    long = pd.concat([
        matches[["Date", "HomeTeam"]].rename(columns={"HomeTeam": "Team"}),
        matches[["Date", "AwayTeam"]].rename(columns={"AwayTeam": "Team"}),
    ]).drop_duplicates().sort_values(["Team", "Date"])
    gap = long.groupby("Team")["Date"].diff()
    starts = long[gap.isna() | (gap > pd.Timedelta(days=SPELL_GAP_DAYS))]
    return list(zip(starts["Team"], starts["Date"].dt.strftime("%Y-%m-%d")))


def _clean_cache(elo_cache):
    cache = elo_cache[["team", "QueryDate", "elo", "from", "to"]].dropna(subset=["team", "QueryDate", "elo"]).copy()
    cache["QueryDate"] = pd.to_datetime(cache["QueryDate"], format="mixed").dt.strftime("%Y-%m-%d")
    cache["from"] = pd.to_datetime(cache["from"], format="mixed", errors="coerce")
    cache["to"] = pd.to_datetime(cache["to"], format="mixed", errors="coerce")
    return cache.drop_duplicates(["team", "QueryDate"], keep="last")


def _valid_on(cache, team, day):
    """ClubElo rating of `team` in force on `day` (its from..to range covers it), or None."""
    rows = cache[(cache["team"] == team) & (cache["from"] <= day) & (cache["to"] >= day)]
    return None if rows.empty else float(rows["elo"].iloc[-1])


def _load_manual(league):
    if not os.path.exists(league.manual_seeds_file):
        return {}
    m = pd.read_csv(league.manual_seeds_file, dtype={"season": str})
    return {(r.team, r.season): float(r.elo) for r in m.itertuples()}


def seed_ratings(league, matches, fetch=None):
    """{(team, spell start date): starting rating} for every spell in
    `matches` (played or not). fetch: optional callable(list of
    'YYYY-MM-DD') -> updated elo cache, used for seeds not cached yet."""
    spells = spell_starts(matches)
    manual = _load_manual(league)
    cache = _clean_cache(pd.read_csv(league.elo_file))

    def pre_match(team, date):
        valid = _valid_on(cache, team, pd.Timestamp(date) - pd.Timedelta(days=1))
        if valid is not None:
            return valid
        same_day = cache[(cache["team"] == team) & (cache["QueryDate"] == date)]
        return None if same_day.empty else float(same_day["elo"].iloc[-1])

    missing = [d for t, d in spells if (t, season_of(d)) not in manual and pre_match(t, d) is None]
    if missing and fetch is not None:
        cache = _clean_cache(fetch(sorted(set(missing))))

    seeds, source = {}, {}
    for team, date in spells:
        key = (team, date)
        exact = pre_match(team, date)
        if (team, season_of(date)) in manual:
            seeds[key], source[key] = manual[(team, season_of(date))], "manual"
        elif exact is not None:
            seeds[key], source[key] = exact, "ClubElo"
        else:
            before = cache[(cache["team"] == team) & (cache["QueryDate"] < date)]
            if not before.empty:
                row = before.sort_values("QueryDate").iloc[-1]
                days = (pd.Timestamp(date) - pd.Timestamp(row["QueryDate"])).days
                seeds[key] = float(row["elo"])
                source[key] = (f"FALLBACK: ClubElo {row['QueryDate']}, {days} days before its first match -- "
                               f"add the pre-match rating to {os.path.basename(league.manual_seeds_file)}")

    # Promoted = spells starting well after the first season kicked off (not
    # just a team missing the opening day).
    cutoff = (pd.Timestamp(min(d for _, d in spells)) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    promoted = [v for (t, d), v in seeds.items() if d > cutoff]
    default = float(np.mean(promoted)) if promoted else float(np.mean(list(seeds.values())))
    for key in spells:
        if key not in seeds:
            seeds[key], source[key] = default, "promoted-team average (no rating found)"

    for (team, date), src in sorted(source.items(), key=lambda kv: kv[0][1]):
        if src != "ClubElo":
            print(f"  Elo seed {team} {season_of(date)}: {seeds[(team, date)]:.0f} from {src}")
    return seeds


def compute_elo(matches, seeds, k=K, home_advantage=HOME_ADVANTAGE):
    """Adds pre-match Home_Elo / Away_Elo / Elo_Difference to a copy of
    `matches` (needs Date, Time, HomeTeam, AwayTeam, FTHG, FTAG). A team's
    rating is (re)set from `seeds` at each spell start. Rows with NaN goals
    (unplayed fixtures) get the current ratings but don't update them."""
    out = matches.copy()
    ratings = {}
    home_elo = np.full(len(out), np.nan)
    away_elo = np.full(len(out), np.nan)
    date_str = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d").values

    order = np.lexsort((out["Time"].astype(str).values, out["Date"].values))
    for i in order:
        row = out.iloc[i]
        home, away = row["HomeTeam"], row["AwayTeam"]
        for team in (home, away):
            if (team, date_str[i]) in seeds:
                ratings[team] = seeds[(team, date_str[i])]
        r_home, r_away = ratings[home], ratings[away]
        home_elo[i], away_elo[i] = r_home, r_away

        hg, ag = row["FTHG"], row["FTAG"]
        if pd.isna(hg) or pd.isna(ag):
            continue
        expected = 1 / (1 + 10 ** ((r_away - (r_home + home_advantage)) / 400))
        score = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = k * (score - expected)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    out["Home_Elo"] = home_elo
    out["Away_Elo"] = away_elo
    out["Elo_Difference"] = out["Home_Elo"] - out["Away_Elo"]
    out.attrs["final_ratings"] = ratings  # every team's rating after the last played match
    return out


def completed_seasons(league):
    """['23-24', '24-25', '25-26'] from the league's raw/ folder (live files excluded)."""
    names = [os.path.basename(p) for p in glob.glob(league.path("raw", "*.csv"))]
    return sorted(m.group(1) for n in names if (m := re.fullmatch(r"(\d\d-\d\d)\.csv", n)))


def load_history_matches(league, seasons=None):
    seasons = seasons or completed_seasons(league)
    return pd.concat([load_data(league.raw_file(s)) for s in seasons], ignore_index=True)


def refresh_processed_elo(league, seeds, history_matches=None):
    """Overwrites Home_Elo / Away_Elo in the league's processed dataset with
    ratings computed from the raw match files, so training data and live
    features always use the same Elo definition."""
    history_matches = history_matches if history_matches is not None else load_history_matches(league)
    elo = compute_elo(history_matches, seeds)[["Date", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]]
    elo["Date"] = elo["Date"].astype("datetime64[ns]")
    if not os.path.exists(league.processed_file):
        return
    df = pd.read_csv(league.processed_file, parse_dates=["Date"])
    df["Date"] = df["Date"].astype("datetime64[ns]")
    cols = df.columns.tolist()
    df = df.drop(columns=["Home_Elo", "Away_Elo"]).merge(elo, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    df[cols].to_csv(league.processed_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="all", help="league id, or 'all'")
    for league in leagues_from_arg(parser.parse_args().league):
        history = load_history_matches(league)
        refresh_processed_elo(league, seed_ratings(league, history), history)
        print(f"{league.name}: recomputed Elo (K={K}, home advantage={HOME_ADVANTAGE}) for {len(history)} matches")


if __name__ == "__main__":
    main()
