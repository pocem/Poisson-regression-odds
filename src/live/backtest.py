"""
Backtest of a league's live season, exactly as the live pipeline runs: the
model is frozen -- trained once on the WINDOW completed seasons -- and predicts
every round with that same fit. Each match's features (Elo, xG, PPG, ...) only
use data from before that match, so nothing is known in hindsight.

    python src/live/backtest.py --league premier_league     (or: all)
        -> data/leagues/<league>/predictions/<season>_model_odds.csv
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "football_odds"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from leagues import leagues_from_arg  # noqa: E402
from Bivariate_Poisson import PoissonRegressionGoalsMeanImpute  # noqa: E402
from sources import fetch_fixtures  # noqa: E402
from predict_round import WINDOW, next_season  # noqa: E402
from compute_elo import completed_seasons  # noqa: E402


def season_backtest(league, fixtures, history_seasons=None):
    """One row per played live-season match: match info, result, Bet365 odds,
    and the frozen model's H/D/A probabilities."""
    history_seasons = history_seasons or completed_seasons(league)
    hist = pd.read_csv(league.processed_file, parse_dates=["Date"])
    prior = hist[hist["Season"].isin(history_seasons[-WINDOW:])]
    live = pd.read_csv(league.live_season_file, parse_dates=["Date"]).merge(
        fixtures[["RoundNumber", "Kickoff", "HomeTeam", "AwayTeam"]], on=["HomeTeam", "AwayTeam"], how="left",
    )
    # The same pairing can appear twice in a fixture feed (a feed error, or a
    # rearranged match): keep the fixture whose kickoff is closest to the match date.
    live["_gap"] = (live["Kickoff"].dt.normalize() - live["Date"]).abs()
    live = (live.sort_values("_gap").drop_duplicates(["Date", "HomeTeam", "AwayTeam"])
            .drop(columns="_gap"))
    missing = live[live["RoundNumber"].isna()]
    if not missing.empty:
        raise ValueError(f"played matches not in the fixture feed: {missing[['HomeTeam', 'AwayTeam']].values.tolist()}")

    model = PoissonRegressionGoalsMeanImpute().fit(prior)
    live[["p_home", "p_draw", "p_away"]] = model.predict_proba(live)
    return live.sort_values(["RoundNumber", "Kickoff"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="all", help="league id, or 'all' (default)")
    for league in leagues_from_arg(parser.parse_args().league):
        season = next_season(completed_seasons(league)[-1])
        bt = season_backtest(league, fetch_fixtures(league, season))
        out = pd.DataFrame({
            "round": bt["RoundNumber"].astype(int), "kickoff": bt["Kickoff"],
            "home_team": bt["HomeTeam"], "away_team": bt["AwayTeam"],
            "p_home": bt["p_home"].round(4), "p_draw": bt["p_draw"].round(4), "p_away": bt["p_away"].round(4),
            "odds_home": (1 / bt["p_home"]).round(2), "odds_draw": (1 / bt["p_draw"]).round(2),
            "odds_away": (1 / bt["p_away"]).round(2),
            "b365_home": bt["B365HomeOdds"], "b365_draw": bt["B365DrawOdds"], "b365_away": bt["B365AwayOdds"],
            "home_goals": bt["FTHG"], "away_goals": bt["FTAG"], "result": bt["FTR"],
        })
        path = league.model_odds_csv(season)
        out.to_csv(path, index=False)
        print(f"{league.name}: wrote {len(out)} matches to {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
