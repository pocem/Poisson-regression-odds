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

Seed sources, in order:
  1. data/external/elo_seeds_manual.csv (team, season, elo) -- hand-entered
     ratings, always win,
  2. the ClubElo cache (data/external/elo_df.csv) on the spell's first match date,
  3. that date fetched from ClubElo (only if a fetch function is passed in),
  4. the team's latest cached rating before that date,
  5. the average seed of the promoted spells, so a team with no rating at all
     still gets a sensible value.

Run directly to recompute Home_Elo / Away_Elo in the processed datasets:
    python src/pipeline/compute_elo.py
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from process_season_data import load_data

K = 20
HOME_ADVANTAGE = 0
SPELL_GAP_DAYS = 200  # longer than a summer break, shorter than a season away

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "raw")
ELO_FILE = os.path.join(ROOT, "data", "external", "elo_df.csv")
MANUAL_SEEDS_FILE = os.path.join(ROOT, "data", "external", "elo_seeds_manual.csv")
PROCESSED_FILES = [
    os.path.join(ROOT, "data", "processed", "all_seasons_14window_ppg.csv"),
    os.path.join(ROOT, "data", "processed", "all_seasons_14window_ppg_full.csv"),
]


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
    cache = elo_cache[["team", "QueryDate", "elo"]].dropna().copy()
    cache["QueryDate"] = pd.to_datetime(cache["QueryDate"]).dt.strftime("%Y-%m-%d")
    return cache.drop_duplicates(["team", "QueryDate"], keep="last")


def _load_manual():
    if not os.path.exists(MANUAL_SEEDS_FILE):
        return {}
    m = pd.read_csv(MANUAL_SEEDS_FILE, dtype={"season": str})
    return {(r.team, r.season): float(r.elo) for r in m.itertuples()}


def seed_ratings(matches, elo_cache, fetch=None):
    """{(team, spell start date): starting rating} for every spell in
    `matches` (played or not). fetch: optional callable(list of
    'YYYY-MM-DD') -> updated elo cache, used for seeds not cached yet."""
    spells = spell_starts(matches)
    manual = _load_manual()
    cache = _clean_cache(elo_cache)
    exact = cache.set_index(["team", "QueryDate"])["elo"]

    missing = [d for t, d in spells if (t, season_of(d)) not in manual and (t, d) not in exact.index]
    if missing and fetch is not None:
        cache = _clean_cache(fetch(sorted(set(missing))))
        exact = cache.set_index(["team", "QueryDate"])["elo"]

    seeds, source = {}, {}
    for team, date in spells:
        key = (team, date)
        if (team, season_of(date)) in manual:
            seeds[key], source[key] = manual[(team, season_of(date))], "manual"
        elif key in exact.index:
            seeds[key], source[key] = float(exact[key]), "ClubElo"
        else:
            before = cache[(cache["team"] == team) & (cache["QueryDate"] < date)]
            if not before.empty:
                row = before.sort_values("QueryDate").iloc[-1]
                seeds[key], source[key] = float(row["elo"]), f"ClubElo {row['QueryDate']} (latest cached before)"

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
    return out


def completed_seasons():
    names = [os.path.basename(p) for p in glob.glob(os.path.join(RAW_DIR, "pl*.csv"))]
    return sorted(m.group(1) for n in names if (m := re.fullmatch(r"pl(\d\d-\d\d)\.csv", n)))


def load_history_matches(seasons=None):
    seasons = seasons or completed_seasons()
    return pd.concat([load_data(os.path.join(RAW_DIR, f"pl{s}.csv")) for s in seasons], ignore_index=True)


def refresh_processed_elo(seeds, history_matches=None):
    """Overwrites Home_Elo / Away_Elo in the processed datasets with ratings
    computed from the raw match files, so training data and live features
    always use the same Elo definition."""
    history_matches = history_matches if history_matches is not None else load_history_matches()
    elo = compute_elo(history_matches, seeds)[["Date", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]]
    elo["Date"] = elo["Date"].astype("datetime64[ns]")
    for path in PROCESSED_FILES:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = df["Date"].astype("datetime64[ns]")
        cols = df.columns.tolist()
        df = df.drop(columns=["Home_Elo", "Away_Elo"]).merge(elo, on=["Date", "HomeTeam", "AwayTeam"], how="left")
        df[cols].to_csv(path, index=False)


def main():
    history = load_history_matches()
    seeds = seed_ratings(history, pd.read_csv(ELO_FILE))
    refresh_processed_elo(seeds, history)
    print(f"Recomputed Elo (K={K}, home advantage={HOME_ADVANTAGE}) for {len(history)} matches "
          f"in {', '.join(os.path.relpath(p, ROOT) for p in PROCESSED_FILES)}")


if __name__ == "__main__":
    main()
