"""
Builds the model's features for any set of a league's seasons as one
continuous frame, so PPG carryover, Elo and every EWMA rolling feature continue
across season boundaries. Used both for the historical training dataset
(src/pipeline/build_dataset.py) and for the live season (build_live_frames).

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
from rebuild_rolling_as_ewma import VENUE_ROLLING_COLS, TEAM_ROLLING_COLS  # noqa: E402
from compute_elo import compute_elo  # noqa: E402

from sources import align_xg_dates, season_start_year, understat_rows  # noqa: E402

SPAN = 14  # EWMA span of every rolling feature

RAW_STAT_COLS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
]


def load_xg_long(league, seasons):
    rows = []
    for season in seasons:
        path = league.understat_cache(season_start_year(season))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows += understat_rows(league, json.load(f))
    return pd.DataFrame(rows, columns=["Team", "Date_str", "xG", "xGA", "deep", "deep_allowed"])


def _placeholder_raw(upcoming_df):
    df = upcoming_df[["Date", "Time", "HomeTeam", "AwayTeam"]].copy()
    for c in RAW_STAT_COLS:
        df[c] = float("nan")
    return df


def build_feature_frame(league, season_frames, elo_seeds):
    """season_frames: {season: load_data()-shaped frame}, oldest season first.
    Returns every match of every season with Season, PPG, Elo and all rolling
    features (the model's covariates among them)."""
    seasons = list(season_frames)

    # PPG, carried over season to season
    frames = []
    prior_ppg = None
    for season in seasons:
        raw, prior_ppg = add_points_per_game(season_frames[season].copy(), prior_ppg=prior_ppg)
        raw["Season"] = season
        frames.append(raw)
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.rename(columns={"HomePPG": "Home_PPG", "AwayPPG": "Away_PPG", "PPGDiff": "PPG_Difference"})
    matches["Date_str"] = matches["Date"].dt.strftime("%Y-%m-%d")

    matches = compute_elo(matches, elo_seeds)

    # Team-centric frame + EWMA rolling features, continuous across seasons
    team_frames = []
    for season in seasons:
        raw = season_frames[season].copy()
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
    xg = align_xg_dates(load_xg_long(league, seasons), zip(team_all["Team"], team_all["Date_str"]))
    team_all = team_all.merge(xg, on=["Date_str", "Team"], how="left")
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
    return matches.drop(columns=["Date_str"])


def build_live_frames(league, history_seasons, current_season, played_df, upcoming_df, elo_seeds):
    """
    history_seasons: completed seasons with a file in the league's raw/ folder, oldest first.
    played_df: load_data()-shaped frame of the live season so far, or None.
    upcoming_df: DataFrame[Date, Time, HomeTeam, AwayTeam] to predict.
    elo_seeds: {(team, spell start date): starting rating} from compute_elo.seed_ratings().

    Returns (played_rows, predict_rows) -- both live-season only, with every
    covariate the model needs.
    """
    season_frames = {s: load_data(league.raw_file(s)) for s in history_seasons}
    current_raw = played_df if played_df is not None else pd.DataFrame(
        columns=["Date", "Time", "HomeTeam", "AwayTeam"] + RAW_STAT_COLS)
    season_frames[current_season] = (
        pd.concat([current_raw, _placeholder_raw(upcoming_df)], ignore_index=True)
        .sort_values(["Date", "Time"]).reset_index(drop=True)
    )
    matches = build_feature_frame(league, season_frames, elo_seeds)
    live = matches[matches["Season"] == current_season]
    return live[live["FTHG"].notna()].copy(), live[live["FTHG"].isna()].copy()
