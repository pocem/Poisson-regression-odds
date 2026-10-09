"""
Builds a league's training dataset, data/leagues/<league>/processed/all_seasons.csv,
from every completed season file in its raw/ folder + Understat xG + Elo.

One continuous frame over all those seasons -- PPG, Elo and every rolling
feature carry over across season boundaries, and no rows are dropped -- built
with exactly the same code as the live season's features (src/live/features.py),
so training and prediction inputs always match. Only the columns the model and
its Bet365 evaluation use are kept (Bivariate_Poisson.DATASET_COLUMNS).

    python src/pipeline/build_dataset.py --league bundesliga     (or: all)
    python src/pipeline/build_dataset.py --league bundesliga --overwrite   # replace an existing one

Note: the Premier League's all_seasons.csv came from the ML project's 2014+
build, so its early-23-24 rolling features carry over from 22-23. Rebuilding it
here starts the carryover at 23-24 instead (the first season in its raw/ folder).
That's why an existing dataset is only replaced with --overwrite.
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
sys.path.insert(0, os.path.join(ROOT, "src", "football_odds"))
from leagues import leagues_from_arg  # noqa: E402
from features import build_feature_frame  # noqa: E402
from compute_elo import completed_seasons, seed_ratings  # noqa: E402
from process_season_data import load_data  # noqa: E402
from add_bet365_odds import build_bet365_frame  # noqa: E402
from Bivariate_Poisson import DATASET_COLUMNS  # noqa: E402


def build_dataset(league):
    seasons = completed_seasons(league)
    frames = {s: load_data(league.raw_file(s)) for s in seasons}
    seeds = seed_ratings(league, pd.concat(frames.values(), ignore_index=True))
    matches = build_feature_frame(league, frames, seeds)

    odds = pd.concat([build_bet365_frame(league.raw_file(s)) for s in seasons], ignore_index=True)
    matches = matches.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    out = matches.sort_values(["Date", "Time"]).reset_index(drop=True)[DATASET_COLUMNS]

    os.makedirs(os.path.dirname(league.processed_file), exist_ok=True)
    out.to_csv(league.processed_file, index=False)
    nan_rows = out.drop(columns=["FTR"]).isna().any(axis=1)
    print(f"{league.name}: {len(out)} matches ({', '.join(seasons)}) -> {os.path.relpath(league.processed_file, ROOT)}")
    print(f"  rows with a missing covariate (a team's first match in the data; mean-imputed by the model): "
          f"{int(nan_rows.sum())}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True, help="league id, or 'all'")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing all_seasons.csv")
    args = parser.parse_args()
    for league in leagues_from_arg(args.league):
        if os.path.exists(league.processed_file) and not args.overwrite:
            print(f"{league.name}: {os.path.relpath(league.processed_file, ROOT)} already exists -- skipped "
                  f"(pass --overwrite to rebuild it)")
            continue
        build_dataset(league)


if __name__ == "__main__":
    main()
