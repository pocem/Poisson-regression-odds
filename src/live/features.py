"""
Builds model features for the live season, the same way
build_full_dataset_no_loss.py builds the historical dataset: one continuous
frame from the first raw season through the live one, so PPG carryover and
every EWMA rolling feature continue across season boundaries.

Unplayed fixtures of the round being predicted are appended as placeholder
rows (all stats NaN). shift(1) inside the EWMA means a placeholder row picks
up each team's latest feature values without its own (unknown) stats, so no
separate "carry forward" logic is needed.
"""

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "pipeline"))
from process_season_data import load_data, add_points_per_game, create_team_df  # noqa: E402
from rebuild_rolling_as_ewma import TEAM_MAP, VENUE_ROLLING_COLS, TEAM_ROLLING_COLS  # noqa: E402
from build_full_dataset_no_loss import SPAN  # noqa: E402
from compute_elo import compute_elo  # noqa: E402

from sources import CACHE_DIR, RAW_DIR, season_start_year  # noqa: E402

RAW_STAT_COLS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]


def load_xg_long(seasons):
    rows = []
    for season in seasons:
        path = os.path.join(CACHE_DIR, f"leaguedata_{season_start_year(season)}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
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


def _placeholder_raw(upcoming_df):
    df = upcoming_df[["Date", "Time", "HomeTeam", "AwayTeam"]].copy()
    for c in RAW_STAT_COLS:
        df[c] = float("nan")
    return df


def build_live_frames(history_seasons, current_season, played_df, upcoming_df, elo_seeds):
    """
    history_seasons: completed seasons with a data/raw/pl{s}.csv, oldest first.
    played_df: load_data()-shaped frame of the live season so far, or None.
    upcoming_df: DataFrame[Date, Time, HomeTeam, AwayTeam] to predict.
    elo_seeds: {(team, spell start date): starting rating} from compute_elo.seed_ratings().

    Returns (played_rows, predict_rows) -- both live-season only, with every
    covariate the model needs.
    """
    all_seasons = list(history_seasons) + [current_season]

    raw_frames = {s: load_data(os.path.join(RAW_DIR, f"pl{s}.csv")) for s in history_seasons}
    current_raw = played_df if played_df is not None else pd.DataFrame(
        columns=["Date", "Time", "HomeTeam", "AwayTeam"] + RAW_STAT_COLS)
    raw_frames[current_season] = (
        pd.concat([current_raw, _placeholder_raw(upcoming_df)], ignore_index=True)
        .sort_values(["Date", "Time"]).reset_index(drop=True)
    )

    # PPG, carried over season to season
    frames = []
    prior_ppg = None
    for season in all_seasons:
        raw, prior_ppg = add_points_per_game(raw_frames[season].copy(), prior_ppg=prior_ppg)
        raw["Season"] = season
        frames.append(raw)
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.rename(columns={"HomePPG": "Home_PPG", "AwayPPG": "Away_PPG", "PPGDiff": "PPG_Difference"})
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")

    matches = compute_elo(matches, elo_seeds)

    # Team-centric frame + EWMA rolling features, continuous across seasons
    team_frames = []
    for season in all_seasons:
        raw = raw_frames[season].copy()
        raw["TablePosDiff"] = 0.0  # not a model covariate; create_team_df just needs the column
        team_frames.append(create_team_df(raw))
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

    elo_lookup = matches[["Date_str", "Time", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]]
    opponent_elo = pd.concat([
        elo_lookup.rename(columns={"HomeTeam": "Team", "Away_Elo": "OpponentElo"})[["Date_str", "Time", "Team", "OpponentElo"]],
        elo_lookup.rename(columns={"AwayTeam": "Team", "Home_Elo": "OpponentElo"})[["Date_str", "Time", "Team", "OpponentElo"]],
    ], ignore_index=True)
    team_all = team_all.merge(opponent_elo, on=["Date_str", "Time", "Team"], how="left")
    team_all = team_all.merge(load_xg_long(all_seasons), on=["Date_str", "Team"], how="left")
    team_all = team_all.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)

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

    all_new = [f"{c}_Rolling5" for c in VENUE_ROLLING_COLS] + [f"{c}_RollingTeam7" for c in TEAM_ROLLING_COLS]
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
    matches = matches.drop(columns=["Date_str"])

    live = matches[matches["Season"] == current_season]
    return live[live["FTHG"].notna()].copy(), live[live["FTHG"].isna()].copy()
