"""
Downloads a league's completed seasons into its folder (src/leagues.py):
football-data.co.uk match files -> raw/<season>.csv and Understat xG ->
cache/understat_<year>.json. Used once when adding a league; afterwards the
live pipeline keeps the current season up to date by itself.

    python src/pipeline/download_history.py --league bundesliga --seasons 22-23 23-24 24-25 25-26

Then add the league's ClubElo ratings to external/elo_df.csv (or starting Elos
to external/elo_seeds_manual.csv) and build the training data:

    python src/pipeline/build_dataset.py --league bundesliga
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from leagues import get_league  # noqa: E402
from sources import download_season_file, download_understat, season_start_year  # noqa: E402
from process_season_data import load_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", required=True)
    parser.add_argument("--seasons", nargs="+", required=True, help="completed seasons, e.g. 23-24 24-25")
    args = parser.parse_args()
    league = get_league(args.league)

    for season in args.seasons:
        path = league.raw_file(season)
        if not download_season_file(league, season, path):
            raise SystemExit(f"football-data.co.uk has no {league.football_data_code} file for {season}")
        n = len(load_data(path))
        year = season_start_year(season)
        data = download_understat(league, year)
        n_xg = sum(len(t["history"]) for t in data["teams"].values()) // 2
        print(f"{league.name} {season}: {n} matches -> {os.path.relpath(path, ROOT)}; "
              f"Understat xG for {n_xg} matches -> {os.path.relpath(league.understat_cache(year), ROOT)}")
        if n != league.teams * (league.teams - 1):
            print(f"  NOTE: expected {league.teams * (league.teams - 1)} matches for a full season")


if __name__ == "__main__":
    main()
