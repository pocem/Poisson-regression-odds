import glob

import pandas as pd

MATCHES_FILE = "data/processed/all_seasons_with_bookies.csv"


def build_bet365_frame(path):
    df = pd.read_csv(path, usecols=["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df.rename(columns={
        "B365H": "B365HomeOdds",
        "B365D": "B365DrawOdds",
        "B365A": "B365AwayOdds",
    })


def main():
    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])
    seasons = sorted(matches["Season"].unique())

    frames = []
    for season in seasons:
        path = f"data/raw/pl{season}.csv"
        frames.append(build_bet365_frame(path))
        print(f"Loaded {path}")

    bet365 = pd.concat(frames, ignore_index=True)

    matches = matches.merge(bet365, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    missing = matches["B365HomeOdds"].isna().sum()
    print(f"\nMatches with no Bet365 odds: {missing} / {len(matches)}")
    if missing:
        print(matches[matches["B365HomeOdds"].isna()].groupby("Season").size())

    matches.to_csv(MATCHES_FILE, index=False)
    print(f"\nSaved {matches.shape} to {MATCHES_FILE}")


if __name__ == "__main__":
    main()
