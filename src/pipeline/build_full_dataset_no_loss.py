"""
Rebuilds the same feature set as data/processed/all_seasons_14window_ppg.csv, but
without the legacy per-season loss.

The current pipeline is:
    process_season_data.export_dataset() [PER SEASON, dropna()] -> data/processed/{season}.csv
    all_df.py [aggregate]                                       -> all_seasons.csv
    team_elo_api.py / add_bookie_odds.py / add_bet365_odds.py /
    add_xg_features.py / add_team_rolling_and_rest.py           -> all_seasons_with_bookies.csv
    build_14window_dataset.py                                   -> all_seasons_14window.csv
    build_ppg_dataset.py                                        -> all_seasons_14window_ppg.csv

export_dataset()'s dropna() runs on EACH SEASON IN ISOLATION, before
concatenation. Table position and every Rolling5 feature were computed
fresh per season back then, so every team's first home match and first
away match of every season had no prior within-season history (Rolling5
= NaN) and got silently dropped -- about 20 of 380 matches/season, ~245
total across the 12-season dataset (measured directly: data/processed/{s}.csv
consistently has ~20 fewer rows than the matching raw data/raw/pl{s}.csv).

None of the LATER stages drop rows -- they're all left-merges keyed on
(Date, Time, HomeTeam, AwayTeam) onto whatever row set already exists.
So the loss is entirely a property of the very first stage, and it is
already fixed in spirit for the *rolling features themselves* --
rebuild_rolling_as_ewma.build_team_centric() and this script's own
add_team_rolling_and_rest.py already recompute every Rolling5/RollingTeam7
feature continuously across season boundaries (no reset), and use
min_periods=1 EWMA -- but that fix was grafted on AFTER the row set was
already capped by the original per-season dropna(). Fixing the rolling
math doesn't un-drop rows that were removed three stages earlier.

This script builds ONE continuous multi-season frame directly from the
raw pl*.csv files from the start -- table position, PPG, and every
rolling feature all carry forward across season boundaries from the very
first row, and dropna() is never called. The only rows that can still
have a genuine NaN afterwards are a team's literal first-ever tracked
match in the full 2014-2026 window (there's nothing prior to carry over
from) -- not every season's opening fixtures.

Reuses the exact same functions/constants the existing pipeline already
uses (process_season_data.load_data/add_table_positions/
add_points_per_game/create_team_df, rebuild_rolling_as_ewma.TEAM_MAP/
VENUE_ROLLING_COLS/TEAM_ROLLING_COLS, add_bookie_odds.build_odds_frame,
add_bet365_odds.build_bet365_frame) so the feature definitions are
identical to the current dataset -- only the row set changes.

Writes to a NEW file (does not touch the dataset currently used by the
models/report) so results can be compared side by side first.
"""

import json

import pandas as pd

from process_season_data import (
    load_data,
    add_table_positions,
    add_points_per_game,
    create_team_df,
)
from rebuild_rolling_as_ewma import TEAM_MAP, VENUE_ROLLING_COLS, TEAM_ROLLING_COLS
from add_bookie_odds import build_odds_frame
from add_bet365_odds import build_bet365_frame
from compute_elo import compute_elo, seed_ratings

# Trimmed to 23-24 onward for this project. Note: the processed CSVs in
# data/processed were filtered from the original 14-15..25-26 build, so their
# early-23-24 rolling/PPG features still carry over from 22-23. Re-running this
# script with only these seasons restarts that carryover at 23-24.
SEASONS = ["23-24", "24-25", "25-26"]
SPAN = 14
SCRATCH_DIR = "data/cache"
ELO_FILE = "data/external/elo_df.csv"
OUTPUT_FILE = "data/processed/all_seasons_14window_ppg_full.csv"

# Columns present in the current all_seasons_14window_ppg.csv that this
# script should match exactly, for apples-to-apples comparison.
REFERENCE_FILE = "data/processed/all_seasons_14window_ppg.csv"


def load_xg_long():
    rows = []
    for year in range(2023, 2026):
        with open(f"{SCRATCH_DIR}/leaguedata_{year}.json", encoding="utf-8") as f:
            data = json.load(f)
        for team in data["teams"].values():
            title = TEAM_MAP.get(team["title"], team["title"])
            for m in team["history"]:
                rows.append({
                    "Date_str": pd.to_datetime(m["date"]).strftime("%Y-%m-%d"),
                    "Team": title,
                    "xG": m["xG"], "xGA": m["xGA"],
                    "deep": m["deep"], "deep_allowed": m["deep_allowed"],
                })
    return pd.DataFrame(rows)


def build_base_matches():
    """Every raw match, all seasons, continuous table-position/PPG carryover."""
    frames = []
    prior_positions = None
    prior_ppg = None
    for season in SEASONS:
        raw = load_data(f"data/raw/pl{season}.csv")
        raw, prior_positions = add_table_positions(raw, prior_positions=prior_positions)
        raw, prior_ppg = add_points_per_game(raw, prior_ppg=prior_ppg)
        raw["Season"] = season
        frames.append(raw)

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values(["Date", "Time"]).reset_index(drop=True)
    matches = matches.rename(columns={
        "HomePPG": "Home_PPG", "AwayPPG": "Away_PPG", "PPGDiff": "PPG_Difference",
    })
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")
    return matches


def add_elo(matches):
    """Self-computed Elo (compute_elo.py): seeded once per team from ELO_FILE,
    then updated from results."""
    seeds = seed_ratings(matches, pd.read_csv(ELO_FILE))
    return compute_elo(matches, seeds)


def add_odds(matches):
    odds_frames = [build_odds_frame(f"data/raw/pl{s}.csv") for s in SEASONS]
    odds_df = pd.concat(odds_frames, ignore_index=True)
    matches = matches.merge(odds_df, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    bet365_frames = [build_bet365_frame(f"data/raw/pl{s}.csv") for s in SEASONS]
    bet365 = pd.concat(bet365_frames, ignore_index=True)
    matches = matches.merge(bet365, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    return matches


def build_team_centric(matches):
    team_frames = []
    prior_positions = None
    for season in SEASONS:
        raw = load_data(f"data/raw/pl{season}.csv")
        raw, prior_positions = add_table_positions(raw, prior_positions=prior_positions)
        team_df = create_team_df(raw)
        team_frames.append(team_df)
    team_all = pd.concat(team_frames, ignore_index=True)

    team_all["ShotAccuracy"] = team_all["ShotsOnTarget"] / team_all["Shots"].replace(0, 1)
    team_all["GoalDifference"] = team_all["GoalsFor"] - team_all["GoalsAgainst"]
    team_all["ShotDifference"] = team_all["Shots"] - team_all["ShotsAgainst"]
    team_all["ShotOnTargetDifference"] = team_all["ShotsOnTarget"] - team_all["ShotsOnTargetAgainst"]
    team_all["CornerDifference"] = team_all["Corners"] - team_all["CornersAgainst"]
    team_all["FoulDifference"] = team_all["FoulsAgainst"] - team_all["Fouls"]
    team_all["YellowCardDifference"] = team_all["YellowCardsAgainst"] - team_all["YellowCards"]
    team_all["Win"] = team_all["Result"].map({"W": 1, "D": 0.5, "L": 0})
    team_all["Date_str"] = team_all["Date"].dt.strftime("%Y-%m-%d")

    elo = matches[["Date_str", "Time", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]].copy()
    home_elo_lookup = elo.rename(columns={"HomeTeam": "Team", "Away_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    away_elo_lookup = elo.rename(columns={"AwayTeam": "Team", "Home_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    opponent_elo_lookup = pd.concat([home_elo_lookup, away_elo_lookup], ignore_index=True)
    team_all = team_all.merge(opponent_elo_lookup, on=["Date_str", "Time", "Team"], how="left")

    xg_long = load_xg_long()
    team_all = team_all.merge(xg_long, on=["Date_str", "Team"], how="left")

    return team_all.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)


def add_ewma_rolling(matches):
    team_all = build_team_centric(matches)

    for col in VENUE_ROLLING_COLS:
        team_all[f"{col}_Rolling5"] = (
            team_all.groupby(["Team", "Venue"])[col]
            .transform(lambda x: x.shift(1).ewm(span=SPAN, min_periods=1).mean())
        )
    for col in TEAM_ROLLING_COLS:
        team_all[f"{col}_RollingTeam7"] = (
            team_all.groupby("Team")[col]
            .transform(lambda x: x.shift(1).ewm(span=SPAN, min_periods=1).mean())
        )

    venue_cols = [f"{c}_Rolling5" for c in VENUE_ROLLING_COLS]
    team_cols = [f"{c}_RollingTeam7" for c in TEAM_ROLLING_COLS]
    all_new = venue_cols + team_cols

    home_roll = (
        team_all[team_all["Venue"] == "H"][["Date", "Time", "Team"] + all_new]
        .rename(columns={"Team": "HomeTeam", **{c: f"Home_{c}" for c in all_new}})
    )
    away_roll = (
        team_all[team_all["Venue"] == "A"][["Date", "Time", "Team"] + all_new]
        .rename(columns={"Team": "AwayTeam", **{c: f"Away_{c}" for c in all_new}})
    )

    matches = matches.merge(home_roll, on=["Date", "Time", "HomeTeam"], how="left")
    matches = matches.merge(away_roll, on=["Date", "Time", "AwayTeam"], how="left")
    return matches


def main():
    matches = build_base_matches()
    print(f"Base matches (all raw rows, all seasons, no dropna): {matches.shape}")

    matches = add_elo(matches)
    matches = add_odds(matches)
    matches = add_ewma_rolling(matches)

    matches = matches.drop(columns=["Date_str"])
    matches = matches.sort_values(["Date", "Time"]).reset_index(drop=True)

    reference_cols = pd.read_csv(REFERENCE_FILE, nrows=0).columns.tolist()
    missing_from_new = [c for c in reference_cols if c not in matches.columns]
    extra_in_new = [c for c in matches.columns if c not in reference_cols]
    print(f"\nColumns in reference but missing here: {missing_from_new}")
    print(f"Columns here but not in reference: {extra_in_new}")

    matches = matches[reference_cols]

    print(f"\nFinal shape: {matches.shape}")
    nan_rows = matches.isna().any(axis=1).sum()
    print(f"Rows with any remaining NaN: {nan_rows} / {len(matches)}")
    if nan_rows:
        nan_mask = matches.isna().any(axis=1)
        print(matches[nan_mask].groupby("Season").size())

    matches.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {matches.shape} to {OUTPUT_FILE}")

    old = pd.read_csv(REFERENCE_FILE)
    print(f"\nOld dataset: {old.shape[0]} rows. New dataset: {matches.shape[0]} rows. "
          f"Recovered: {matches.shape[0] - old.shape[0]} rows.")


if __name__ == "__main__":
    main()
