"""
Replaces every flat (equal-weight) rolling-average feature in
all_seasons_with_bookies.csv with an exponentially-weighted moving average
(EWMA) instead -- same column names, so nothing downstream breaks.

A flat N-match rolling mean has a hard cliff: match N+1 back has full
weight, then the instant it exits the window it's worth exactly zero.
EWMA weights recent matches more and decays older ones smoothly instead,
which is a more realistic model of "recent form fading."

Covers all three families currently in the dataset:
  - Venue-grouped (Team+Venue) stats -> span=5, matching the old _Rolling5
  - OpponentElo (Team+Venue)         -> span=5, matching the old _Rolling5
  - Team-overall (Team only) stats   -> span=7, matching the old _RollingTeam7

Also fixes a pre-existing inconsistency: the original _Rolling5 core stats
were computed per-season (reset every August), while OpponentElo/xG rolling
already spanned continuously across season boundaries. Everything here is
now continuous, consistent with the newer features.
"""

import json

import pandas as pd

from process_season_data import load_data, add_table_positions, create_team_df

MATCHES_FILE = "data/processed/all_seasons_with_bookies.csv"
SCRATCH_DIR = "data/cache"
VENUE_SPAN = 10
TEAM_SPAN = 10

TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

CORE_ROLLING_COLS = [
    "TablePosDiff", "GoalsFor", "GoalsAgainst", "GoalDifference",
    "Shots", "ShotsAgainst", "ShotDifference",
    "ShotsOnTarget", "ShotsOnTargetAgainst", "ShotOnTargetDifference",
    "ShotAccuracy", "Corners", "CornerDifference", "Fouls", "FoulDifference",
    "YellowCards", "YellowCardDifference",
    "Points", "Win",
]
XG_STATS = ["xG", "xGA", "deep", "deep_allowed"]
VENUE_ROLLING_COLS = CORE_ROLLING_COLS + XG_STATS + ["OpponentElo"]
TEAM_ROLLING_COLS = CORE_ROLLING_COLS + XG_STATS


def build_team_centric(matches, seasons):
    team_frames = []
    prior_positions = None
    for season in seasons:
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

    # --- OpponentElo: pull Home_Elo/Away_Elo (already point-in-time, non-rolling,
    #     computed earlier) from `matches` and assign each side its opponent's. ---
    elo = matches[["Date", "Time", "HomeTeam", "AwayTeam", "Home_Elo", "Away_Elo"]].copy()
    elo["Date_str"] = elo["Date"].dt.strftime("%Y-%m-%d")

    home_elo_lookup = elo.rename(columns={"HomeTeam": "Team", "Away_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    away_elo_lookup = elo.rename(columns={"AwayTeam": "Team", "Home_Elo": "OpponentElo"})[
        ["Date_str", "Time", "Team", "OpponentElo"]
    ]
    opponent_elo_lookup = pd.concat([home_elo_lookup, away_elo_lookup], ignore_index=True)

    team_all = team_all.merge(opponent_elo_lookup, on=["Date_str", "Time", "Team"], how="left")

    # --- xG stats from the Understat data already fetched into scratch/ ---
    xg_rows = []
    for year in range(2014, 2026):
        with open(f"{SCRATCH_DIR}/leaguedata_{year}.json", encoding="utf-8") as f:
            data = json.load(f)
        for team in data["teams"].values():
            title = TEAM_MAP.get(team["title"], team["title"])
            for m in team["history"]:
                xg_rows.append({
                    "Date_str": pd.to_datetime(m["date"]).strftime("%Y-%m-%d"),
                    "Team": title,
                    "xG": m["xG"], "xGA": m["xGA"],
                    "deep": m["deep"], "deep_allowed": m["deep_allowed"],
                })
    xg_long = pd.DataFrame(xg_rows)
    team_all = team_all.merge(xg_long, on=["Date_str", "Team"], how="left")

    return team_all.sort_values(["Team", "Date", "Time"]).reset_index(drop=True)


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])

    old_rolling_cols = [c for c in matches.columns if c.endswith("_Rolling5") or "RollingTeam7" in c]
    print(f"Dropping {len(old_rolling_cols)} old flat-rolling columns")
    matches = matches.drop(columns=old_rolling_cols)

    seasons = sorted(matches["Season"].unique())
    team_all = build_team_centric(matches, seasons)

    # --- Venue-grouped EWMA (replaces old _Rolling5) ---
    for col in VENUE_ROLLING_COLS:
        team_all[f"{col}_Rolling5"] = (
            team_all.groupby(["Team", "Venue"])[col]
            .transform(lambda x: x.shift(1).ewm(span=VENUE_SPAN, min_periods=1).mean())
        )

    # --- Team-overall EWMA (replaces old _RollingTeam7) ---
    for col in TEAM_ROLLING_COLS:
        team_all[f"{col}_RollingTeam7"] = (
            team_all.groupby("Team")[col]
            .transform(lambda x: x.shift(1).ewm(span=TEAM_SPAN, min_periods=1).mean())
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

    new_cols = [f"Home_{c}" for c in all_new] + [f"Away_{c}" for c in all_new]
    print(f"Added {len(new_cols)} EWMA columns")
    print("\nNaN counts by season (rows with any NaN among new columns):")
    nan_mask = matches[new_cols].isna().any(axis=1)
    print(matches[nan_mask].groupby("Season").size())

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
