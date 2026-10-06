"""
Backtest of the live season, exactly as the live pipeline runs: the model is
frozen -- trained once on the WINDOW completed seasons -- and predicts every
round with that same fit. Each match's features (Elo, xG, PPG, ...) only use
data from before that match, so nothing is known in hindsight.

    python src/live/backtest.py     # writes data/predictions/{season}_model_odds.csv
"""

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "football_odds"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from Bivariate_Poisson import PoissonRegressionGoalsMeanImpute  # noqa: E402
from sources import fetch_fixtures  # noqa: E402
from predict_round import PROCESSED_FILE, LIVE_SEASON_FILE, WINDOW, next_season  # noqa: E402
from compute_elo import completed_seasons  # noqa: E402


def season_backtest(fixtures, history_seasons=None):
    """One row per played live-season match: match info, result, bookmaker
    odds, and the frozen model's H/D/A probabilities."""
    history_seasons = history_seasons or completed_seasons()
    hist = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"])
    prior = hist[hist["Season"].isin(history_seasons[-WINDOW:])]
    live = pd.read_csv(LIVE_SEASON_FILE, parse_dates=["Date"]).merge(
        fixtures[["RoundNumber", "Kickoff", "HomeTeam", "AwayTeam"]], on=["HomeTeam", "AwayTeam"], how="left",
    )
    missing = live[live["RoundNumber"].isna()]
    if not missing.empty:
        raise ValueError(f"played matches not in the fixture feed: {missing[['HomeTeam', 'AwayTeam']].values.tolist()}")

    model = PoissonRegressionGoalsMeanImpute().fit(prior)
    live[["p_home", "p_draw", "p_away"]] = model.predict_proba(live)
    return live.sort_values(["RoundNumber", "Kickoff"]).reset_index(drop=True)


def main():
    season = next_season(completed_seasons()[-1])
    bt = season_backtest(fetch_fixtures(season))
    out = pd.DataFrame({
        "round": bt["RoundNumber"].astype(int), "kickoff": bt["Kickoff"],
        "home_team": bt["HomeTeam"], "away_team": bt["AwayTeam"],
        "p_home": bt["p_home"].round(4), "p_draw": bt["p_draw"].round(4), "p_away": bt["p_away"].round(4),
        "odds_home": (1 / bt["p_home"]).round(2), "odds_draw": (1 / bt["p_draw"]).round(2),
        "odds_away": (1 / bt["p_away"]).round(2),
        "b365_home": bt["B365HomeOdds"], "b365_draw": bt["B365DrawOdds"], "b365_away": bt["B365AwayOdds"],
        "home_goals": bt["FTHG"], "away_goals": bt["FTAG"], "result": bt["FTR"],
    })
    path = os.path.join(ROOT, "data", "predictions", f"{season}_model_odds.csv")
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} matches to {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
