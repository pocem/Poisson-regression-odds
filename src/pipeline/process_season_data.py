import pandas as pd

def load_data(path):
    keep_columns = [
        'Date', 'Time', 'HomeTeam', 'AwayTeam', 
        'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 
        'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
        'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
    ]

    available_cols = pd.read_csv(path, nrows=0).columns.tolist()
    cols_to_use = [c for c in keep_columns if c in available_cols]
    df = pd.read_csv(path, usecols=cols_to_use)

    # Some season files have a trailing blank row (all commas, no data)
    # which read_csv parses as a row of NaNs instead of skipping it.
    df = df.dropna(subset=['HomeTeam', 'AwayTeam']).reset_index(drop=True)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # If Time is missing (older seasons), fill with a constant placeholder
    # so downstream sort/merge logic doesn't need special-casing everywhere.
    if 'Time' not in df.columns:
        df['Time'] = '00:00'
        print(f"Note: {path} has no Time column — filled with placeholder '00:00'")

    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    return df

def create_team_df(df):
    ### Team-centric dataframe, extra column for points ###
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)
    if 'TablePosDiff' not in df.columns:
        raise ValueError("df is missing 'TablePosDiff' — make sure to call add_table_positions(df) first.")
    # -------------------------
    # HOME TEAM PERSPECTIVE
    # -------------------------

    home_df = pd.DataFrame({
        'Date': df['Date'],
        'Time': df['Time'],
        'Team': df['HomeTeam'],
        'Opponent': df['AwayTeam'],
        'Venue': 'H',
        
        # Pull TablePosDiff from the main match dataframe
        'TablePosDiff': df['TablePosDiff'],

        'GoalsFor': df['FTHG'],
        'GoalsAgainst': df['FTAG'],

        'HTGoalsFor': df['HTHG'],
        'HTGoalsAgainst': df['HTAG'],

        'Shots': df['HS'],
        'ShotsAgainst': df['AS'],

        'ShotsOnTarget': df['HST'],
        'ShotsOnTargetAgainst': df['AST'],

        'Corners': df['HC'],
        'CornersAgainst': df['AC'],

        'Fouls': df['HF'],
        'FoulsAgainst': df['AF'],

        'YellowCards': df['HY'],
        'YellowCardsAgainst': df['AY'],

        'RedCards': df['HR'],
        'RedCardsAgainst': df['AR']
    })

    # Result from home team's perspective
    home_df['Result'] = df['FTR'].map({
        'H': 'W',
        'D': 'D',
        'A': 'L'
    })

    # -------------------------
    # AWAY TEAM PERSPECTIVE
    # -------------------------

    away_df = pd.DataFrame({
        'Date': df['Date'],
        'Time': df['Time'],
        'Team': df['AwayTeam'],
        'Opponent': df['HomeTeam'],
        'Venue': 'A',
        
        # Invert the difference for the away team's perspective
        'TablePosDiff': -df['TablePosDiff'],

        'GoalsFor': df['FTAG'],
        'GoalsAgainst': df['FTHG'],

        'HTGoalsFor': df['HTAG'],
        'HTGoalsAgainst': df['HTHG'],

        'Shots': df['AS'],
        'ShotsAgainst': df['HS'],

        'ShotsOnTarget': df['AST'],
        'ShotsOnTargetAgainst': df['HST'],

        'Corners': df['AC'],
        'CornersAgainst': df['HC'],

        'Fouls': df['AF'],
        'FoulsAgainst': df['HF'],

        'YellowCards': df['AY'],
        'YellowCardsAgainst': df['HY'],

        'RedCards': df['AR'],
        'RedCardsAgainst': df['HR']
    })

    # Result from away team's perspective
    away_df['Result'] = df['FTR'].map({
        'A': 'W',
        'D': 'D',
        'H': 'L'
    })

    # -------------------------
    # COMBINE
    # -------------------------

    team_df = pd.concat([home_df, away_df], ignore_index=True)

    # Points column
    team_df['Points'] = team_df['Result'].map({
        'W': 3,
        'D': 1,
        'L': 0
    })

    # Sort chronologically
    team_df = team_df.sort_values(
        ['Team', 'Date', 'Time']
    ).reset_index(drop=True)

    team_df["GoalDifference"] = (
    team_df["GoalsFor"] -
    team_df["GoalsAgainst"]
)

    team_df["ShotDifference"] = (
        team_df["Shots"] -
        team_df["ShotsAgainst"]
    )

    team_df["ShotOnTargetDifference"] = (
        team_df["ShotsOnTarget"] -
        team_df["ShotsOnTargetAgainst"]
    )

    team_df["CornerDifference"] = (
        team_df["Corners"] -
        team_df["CornersAgainst"]
    )

    team_df["FoulDifference"] = (
        team_df["FoulsAgainst"] -
        team_df["Fouls"]
    )

    return team_df

def add_table_positions(df, prior_positions=None, new_team_default=15):
    """Adds HomeTablePos/AwayTablePos/TablePosDiff.

    For each team's first 3 matches of the season (before the live table is
    trustworthy), the position used is:
      - prior_positions[team] -- that team's FINAL position in the previous
        season, if prior_positions was supplied and the team is in it.
      - new_team_default (15) -- if prior_positions was supplied but this
        particular team isn't in it (newly promoted, no top-flight history).
      - 10 (flat neutral guess) -- if prior_positions is None, i.e. there is
        no previous season at all to draw from (the first season in a
        multi-season build, or a standalone single-season call).

    Returns (df, final_positions) -- final_positions is THIS season's final
    standings as {team: position}, meant to be passed as next season's
    prior_positions so position estimates carry across season boundaries.
    """
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    teams = sorted(
        set(df['HomeTeam']).union(set(df['AwayTeam']))
    )

    table = pd.DataFrame(index=teams)
    table['Pts'] = 0
    table['GD'] = 0
    table['GF'] = 0

    played = {team: 0 for team in teams}

    home_pos = []
    away_pos = []

    for _, row in df.iterrows():

        home = row['HomeTeam']
        away = row['AwayTeam']

        standings = (
            table
            .sort_values(['Pts', 'GD', 'GF'], ascending=False)
            .reset_index()
        )

        standings['Position'] = standings.index + 1
        pos_lookup = dict(zip(standings['index'], standings['Position']))

        # Early-season estimate: carry over last season's final position
        # instead of a flat guess, now that this spans multiple seasons.
        if prior_positions is None:
            home_default = 10
            away_default = 10
        else:
            home_default = prior_positions.get(home, new_team_default)
            away_default = prior_positions.get(away, new_team_default)

        home_pos.append(home_default if played[home] < 3 else pos_lookup[home])
        away_pos.append(away_default if played[away] < 3 else pos_lookup[away])

        # update table AFTER recording positions
        hg = row['FTHG']
        ag = row['FTAG']

        table.loc[home, 'GF'] += hg
        table.loc[home, 'GD'] += hg - ag

        table.loc[away, 'GF'] += ag
        table.loc[away, 'GD'] += ag - hg

        if hg > ag:
            table.loc[home, 'Pts'] += 3
        elif hg < ag:
            table.loc[away, 'Pts'] += 3
        else:
            table.loc[home, 'Pts'] += 1
            table.loc[away, 'Pts'] += 1

        played[home] += 1
        played[away] += 1

    df['HomeTablePos'] = home_pos
    df['AwayTablePos'] = away_pos
    df['TablePosDiff'] = df['AwayTablePos'] - df['HomeTablePos']

    final_standings = (
        table
        .sort_values(['Pts', 'GD', 'GF'], ascending=False)
        .reset_index()
    )
    final_positions = dict(zip(final_standings['index'], final_standings.index + 1))

    return df, final_positions


def add_points_per_game(df, prior_ppg=None, new_team_default_ppg=1.0):
    """Adds HomePPG/AwayPPG/PPGDiff -- points-per-game instead of
    add_table_positions()'s ordinal rank. Rank has a real distortion: the
    gap between 1st and 2nd isn't remotely the same as the gap between 19th
    and 20th (a title race and a relegation battle can both compress to "1
    position" while representing very different actual quality gaps). PPG
    is a genuine rate/interval measure, so a straight difference between two
    teams' PPG doesn't have that problem. Points-PER-GAME rather than raw
    cumulative points, so early-season and late-season values stay directly
    comparable (10 points after 5 games is not the same team strength
    signal as 10 points after 20 games).

    Same early-season carryover idea as add_table_positions: for each
    team's first 3 matches of a season, the value used is:
      - prior_ppg[team] -- that team's final PPG (points/38) from the
        previous season, if prior_ppg was supplied and the team is in it.
      - new_team_default_ppg (1.0, ~a 38-point pace -- roughly a
        newly-promoted side's expected debut level) -- if prior_ppg was
        supplied but this team isn't in it (newly promoted).
      - new_team_default_ppg for everyone -- if prior_ppg is None, i.e.
        there's no previous season at all to draw from.

    PPGDiff = HomePPG - AwayPPG, so positive means the home team has been
    accumulating points faster (same "positive = home advantage" direction
    as TablePosDiff, for a like-for-like swap in any covariate list).

    Returns (df, final_ppg) -- final_ppg is THIS season's final PPG as
    {team: ppg}, to hand off as next season's prior_ppg.
    """
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    teams = sorted(
        set(df['HomeTeam']).union(set(df['AwayTeam']))
    )

    pts = {team: 0 for team in teams}
    played = {team: 0 for team in teams}

    home_ppg = []
    away_ppg = []

    for _, row in df.iterrows():

        home = row['HomeTeam']
        away = row['AwayTeam']

        if prior_ppg is None:
            home_default = new_team_default_ppg
            away_default = new_team_default_ppg
        else:
            home_default = prior_ppg.get(home, new_team_default_ppg)
            away_default = prior_ppg.get(away, new_team_default_ppg)

        home_live = pts[home] / played[home] if played[home] > 0 else home_default
        away_live = pts[away] / played[away] if played[away] > 0 else away_default

        home_ppg.append(home_default if played[home] < 3 else home_live)
        away_ppg.append(away_default if played[away] < 3 else away_live)

        # update AFTER recording, same as add_table_positions
        hg = row['FTHG']
        ag = row['FTAG']

        if hg > ag:
            pts[home] += 3
        elif hg < ag:
            pts[away] += 3
        else:
            pts[home] += 1
            pts[away] += 1

        played[home] += 1
        played[away] += 1

    df['HomePPG'] = home_ppg
    df['AwayPPG'] = away_ppg
    df['PPGDiff'] = df['HomePPG'] - df['AwayPPG']

    final_ppg = {
        team: (pts[team] / played[team] if played[team] > 0 else new_team_default_ppg)
        for team in teams
    }

    return df, final_ppg


def add_rolling_features(team_df):
    """Compute all rolling features"""

    # -------------------------------------------------
    # Create additional features FIRST
    # -------------------------------------------------

    # Offensive efficiency
    team_df["ShotAccuracy"] = (
        team_df["ShotsOnTarget"] /
        team_df["Shots"].replace(0, 1)
    )

    # Net performance
    team_df["GoalDifference"] = (
        team_df["GoalsFor"] -
        team_df["GoalsAgainst"]
    )

    team_df["ShotDifference"] = (
        team_df["Shots"] -
        team_df["ShotsAgainst"]
    )

    team_df["ShotOnTargetDifference"] = (
        team_df["ShotsOnTarget"] -
        team_df["ShotsOnTargetAgainst"]
    )

    team_df["CornerDifference"] = (
        team_df["Corners"] -
        team_df["CornersAgainst"]
    )

    team_df["FoulDifference"] = (
        team_df["FoulsAgainst"] -
        team_df["Fouls"]
    )

    # -------------------------------------------------
    # Columns to compute rolling averages for
    # -------------------------------------------------

    rolling_cols = [

        "TablePosDiff",

        "GoalsFor",
        "GoalsAgainst",
        "GoalDifference",

        "Shots",
        "ShotsAgainst",
        "ShotDifference",

        "ShotsOnTarget",
        "ShotsOnTargetAgainst",
        "ShotOnTargetDifference",

        "ShotAccuracy",

        "Corners",
        "CornerDifference",

        "Fouls",
        "FoulDifference"
    ]

    # -------------------------------------------------
    # Rolling 5 (home/away grouped only)
    # -------------------------------------------------

    for col in rolling_cols:

        team_df[f"{col}_Rolling5"] = (
            team_df
            .groupby(["Team", "Venue"])[col]
            .transform(
                lambda x:
                    x.shift(1)
                     .rolling(5, min_periods=1)
                     .mean()
            )
        )

    # -------------------------------------------------
    # Win indicator
    # -------------------------------------------------

    team_df["Win"] = team_df["Result"].map({
        "W": 1,
        "D": 0.5,
        "L": 0
    })

    # -------------------------------------------------
    # Rolling 5 Points + Win (home/away grouped only)
    # -------------------------------------------------

    for col in ["Points", "Win"]:

        team_df[f"{col}_Rolling5"] = (
            team_df
            .groupby(["Team", "Venue"])[col]
            .transform(
                lambda x:
                    x.shift(1)
                     .rolling(5, min_periods=1)
                     .mean()
            )
        )
    # matches["Elo_Diff"] = matches["Home_Elo"] - matches["Away_Elo"]

    return team_df

def build_match_dataset(team_df, df):

    # -------------------------------------------------
    # Automatically keep all usable pre-match features
    # -------------------------------------------------

    keep_cols = [
        "Date",
        "Time",
        "Team",
        "TablePosDiff"
    ]

    # Automatically include every rolling feature
    keep_cols += [
        c for c in team_df.columns
        if "Rolling" in c
    ]

    # Remove duplicates while preserving order
    keep_cols = list(dict.fromkeys(keep_cols))

    # -------------------------------------------------
    # HOME FEATURES
    # -------------------------------------------------

    home_features = (
        team_df[team_df["Venue"] == "H"][keep_cols]
        .rename(columns={"Team": "HomeTeam"})
    )

    home_features = home_features.rename(
        columns={
            c: f"Home_{c}"
            for c in home_features.columns
            if c not in ["Date", "Time", "HomeTeam"]
        }
    )

    # -------------------------------------------------
    # AWAY FEATURES
    # -------------------------------------------------

    away_features = (
        team_df[team_df["Venue"] == "A"][keep_cols]
        .rename(columns={"Team": "AwayTeam"})
    )

    away_features = away_features.rename(
        columns={
            c: f"Away_{c}"
            for c in away_features.columns
            if c not in ["Date", "Time", "AwayTeam"]
        }
    )

    # -------------------------------------------------
    # Base match data
    # -------------------------------------------------

    matches = df[
        [
            "Date",
            "Time",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "FTHG",
            "FTAG",
        ]
    ].copy()

    # -------------------------------------------------
    # Merge home features
    # -------------------------------------------------

    matches = matches.merge(
        home_features,
        on=["Date", "Time", "HomeTeam"],
        how="left"
    )

    # -------------------------------------------------
    # Merge away features
    # -------------------------------------------------

    matches = matches.merge(
        away_features,
        on=["Date", "Time", "AwayTeam"],
        how="left"
    )

    # -------------------------------------------------
    # Final cleanup
    # -------------------------------------------------

    matches = (
        matches
        .sort_values(["Date", "Time"])
        .reset_index(drop=True)
    )

    return matches


def export_dataset(matches, path="data/processed/13-14.csv"):
    # optional safety cleanup
    df = matches.copy()

    # sort for reproducibility
    df = df.sort_values(['Date', 'Time']).reset_index(drop=True)

    # remove rows that still have missing features (optional but common)
    df = df.dropna()

    # export
    df.to_csv(path, index=False)

    print(f"Dataset exported to {path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    raw_df = load_data("data/raw/pl13-14.csv")
    df, _ = add_table_positions(raw_df)
    team_df = create_team_df(df)       
    team_df = add_rolling_features(team_df)
    matches = build_match_dataset(team_df, df)
    export_dataset(matches)