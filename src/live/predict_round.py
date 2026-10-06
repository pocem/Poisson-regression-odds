"""
Round-by-round odds maker. Every run:

  1. pulls fresh fixtures, played-match stats and Understat xG (sources.py),
  2. recomputes Elo for every match from each team's seed rating
     (src/pipeline/compute_elo.py -- ClubElo is only contacted to seed a team
     we have no rating for) and rebuilds features for the live season (features.py),
  3. predicts the next round with the FROZEN bivariate Poisson model -- trained
     once on the 3 most recent completed seasons (WINDOW = 3) and saved in
     data/models/; it is not refit during the season. Only the features (Elo,
     xG, form, PPG) move from round to round. The saved model is refit only
     when its training data changes (new season, or an edited manual Elo seed),
  4. appends the predictions to data/predictions/predictions_log.csv.

Usage, from anywhere:
    python src/live/predict_round.py              # next upcoming round (+ rearranged games before it)
    python src/live/predict_round.py --round 7    # a specific round (its unplayed fixtures)
    python src/live/predict_round.py --force      # predict even if a source is behind
    python src/live/predict_round.py --skip-if-unchanged   # no new row if the inputs haven't changed

Before predicting, every played match in the fixture feed must already be in
football-data.co.uk (stats) and Understat (xG), and team names must line up
across sources -- otherwise the run stops (exit code 3) rather than predict
from stale features. football-data usually lags results by a day or two.

Safe to re-run: each run appends a new timestamped set of rows, so you can
run it again after late team news and compare.

Season rollover: once a season is finished, save its final football-data.co.uk
file as data/raw/pl{season}.csv (e.g. pl26-27.csv). The live season is always
the one after the newest completed raw file, and training uses the 3 newest.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "football_odds"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from Bivariate_Poisson import (  # noqa: E402
    PoissonRegressionGoalsMeanImpute, _bivpois_logpmf, HOME_COVARIATES, AWAY_COVARIATES, DATASET_COLUMNS,
)
from sources import (  # noqa: E402
    RAW_DIR, ELO_FILE, now_uk, fetch_fixtures, fetch_played_matches, fetch_understat, update_elo_cache,
)
from features import build_live_frames  # noqa: E402
from process_season_data import load_data  # noqa: E402
from add_bookie_odds import build_odds_frame  # noqa: E402
from add_bet365_odds import build_bet365_frame  # noqa: E402
from compute_elo import (  # noqa: E402
    MANUAL_SEEDS_FILE, completed_seasons, load_history_matches, seed_ratings, refresh_processed_elo,
)

WINDOW = 3
PROCESSED_FILE = os.path.join(ROOT, "data", "processed", "all_seasons_14window_ppg_full.csv")
# Played live-season matches with features + bookmaker odds, rewritten every
# run -- Bivariate_Poisson.py appends it to the history to evaluate round by round.
LIVE_SEASON_FILE = os.path.join(ROOT, "data", "processed", "live_season.csv")
MODEL_DIR = os.path.join(ROOT, "data", "models")
LOG_CSV = os.path.join(ROOT, "data", "predictions", "predictions_log.csv")
# Outcome of the latest run (predicted / not ready / unchanged ...), shown on the site.
STATUS_JSON = os.path.join(ROOT, "data", "predictions", "last_run.json")
MAX_GOALS_GRID = 6
NOT_READY_EXIT = 3  # distinct from a crash (1), so automation can treat "wait for data" as normal
REARRANGED_DAYS = 5  # a round's games span Fri-Mon; further out = rearranged


def next_season(season):
    a = int(season.split("-")[0]) + 1
    return f"{a:02d}-{a + 1:02d}"


def select_fixtures(fixtures, requested, now):
    """(target round, fixtures to predict).

    A fixture is "rearranged" if it kicks off more than REARRANGED_DAYS from
    its round's median kickoff (postponed and re-dated, or brought forward).
    Target = the round of the earliest upcoming fixture that isn't rearranged
    -- so mid-round it's the rest of the current round, never the next one.
    Rearranged fixtures kicking off before the target round ends are predicted
    with it. Postponed games with no new date (kickoff in the past, unplayed)
    are skipped."""
    unplayed = fixtures[~fixtures["Played"]]
    if requested is not None:
        return requested, unplayed[unplayed["RoundNumber"] == requested]
    upcoming = unplayed[unplayed["Kickoff"] >= now].sort_values("Kickoff")
    if upcoming.empty:
        return None, upcoming
    round_median = fixtures.groupby("RoundNumber")["Kickoff"].median()
    rearranged = (upcoming["Kickoff"] - upcoming["RoundNumber"].map(round_median)).abs() > pd.Timedelta(days=REARRANGED_DAYS)
    regular = upcoming[~rearranged]
    target = int((regular if not regular.empty else upcoming).iloc[0]["RoundNumber"])
    last_kickoff = upcoming.loc[upcoming["RoundNumber"] == target, "Kickoff"].max()
    return target, upcoming[(upcoming["RoundNumber"] == target) | (rearranged & (upcoming["Kickoff"] <= last_kickoff))]


def data_problems(fixtures, played, xg, history):
    """Reasons the inputs aren't ready to predict from. Empty list = ready."""
    problems = []
    feed_teams = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    fd_teams = set() if played is None else set(played["HomeTeam"]) | set(played["AwayTeam"])

    # Team names: every source must use the fixture feed's (canonical) names.
    for t in sorted(fd_teams - feed_teams):
        problems.append(f"football-data.co.uk team '{t}' isn't in the fixture feed "
                        f"-- add the feed's spelling to FIXTUREDOWNLOAD_TO_CANONICAL in src/live/sources.py")
    for t in sorted(set(xg["Team"]) - feed_teams):
        problems.append(f"Understat team '{t}' isn't in the fixture feed "
                        f"-- add it to TEAM_MAP in src/pipeline/rebuild_rolling_as_ewma.py")
    known = (set(history["HomeTeam"]) | set(history["AwayTeam"]) | fd_teams
             | set(pd.read_csv(ELO_FILE)["team"]) | set(pd.read_csv(MANUAL_SEEDS_FILE)["team"]))
    for t in sorted(feed_teams - known):
        problems.append(f"fixture-feed team '{t}' isn't known to football-data.co.uk, ClubElo or the manual seeds "
                        f"-- a spelling mismatch (fix FIXTUREDOWNLOAD_TO_CANONICAL) or a new team "
                        f"(add its starting Elo to {os.path.relpath(MANUAL_SEEDS_FILE, ROOT)})")

    # Freshness: every result the fixture feed has must be in football-data
    # (stats) and Understat (xG), or the features are a round out of date.
    feed_played = fixtures[fixtures["Played"]]
    fd_pairs = set() if played is None else set(zip(played["HomeTeam"], played["AwayTeam"]))
    behind = [f"{r.HomeTeam} v {r.AwayTeam} (round {r.RoundNumber})"
              for r in feed_played.itertuples() if (r.HomeTeam, r.AwayTeam) not in fd_pairs]
    if behind:
        problems.append(f"football-data.co.uk is missing {len(behind)} played match(es): {', '.join(behind)}")

    if played is not None:
        xg_keys = set(zip(xg["Team"], xg["Date_str"]))
        no_xg = [f"{r.HomeTeam} v {r.AwayTeam} ({r.Date:%Y-%m-%d})" for r in played.itertuples()
                 if (r.HomeTeam, f"{r.Date:%Y-%m-%d}") not in xg_keys
                 or (r.AwayTeam, f"{r.Date:%Y-%m-%d}") not in xg_keys]
        if no_xg:
            problems.append(f"Understat has no xG yet for {len(no_xg)} played match(es): {', '.join(no_xg)}")

        both = feed_played.merge(played, on=["HomeTeam", "AwayTeam"])
        bad = both[(both["HomeTeamScore"] != both["FTHG"]) | (both["AwayTeamScore"] != both["FTAG"])]
        for r in bad.itertuples():
            problems.append(f"score mismatch {r.HomeTeam} v {r.AwayTeam}: fixture feed "
                            f"{int(r.HomeTeamScore)}-{int(r.AwayTeamScore)}, football-data {int(r.FTHG)}-{int(r.FTAG)}")
    return problems


def training_history(train_seasons, history_seasons, elo_seeds):
    """Completed-season training rows from the processed dataset (built from
    the full 2014+ history, so early-23-24 features keep their carryover).
    Seasons not in that file yet (e.g. after a rollover) are rebuilt from raw."""
    processed = pd.read_csv(PROCESSED_FILE, parse_dates=["Date"])
    hist = processed[processed["Season"].isin(train_seasons)]
    missing = [s for s in train_seasons if s not in set(hist["Season"])]
    for season in missing:
        played = load_data(os.path.join(RAW_DIR, f"pl{season}.csv"))
        prior = [h for h in history_seasons if h < season]
        rebuilt, _ = build_live_frames(prior, season, played, played.iloc[0:0], elo_seeds)
        hist = pd.concat([hist, rebuilt], ignore_index=True)
    return hist


def frozen_model(train_df, train_seasons):
    """The season's frozen model: loaded from data/models/ if it was trained on
    exactly this data, otherwise trained now and saved. Returns (model, meta)."""
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + list(dict.fromkeys(HOME_COVARIATES + AWAY_COVARIATES))
    fingerprint = hashlib.sha1(train_df[cols].to_csv(index=False).encode()).hexdigest()[:12]
    name = f"bivariate_poisson_{train_seasons[0]}_to_{train_seasons[-1]}"
    npz_path, meta_path = os.path.join(MODEL_DIR, name + ".npz"), os.path.join(MODEL_DIR, name + ".json")

    if os.path.exists(meta_path) and os.path.exists(npz_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("training_fingerprint") == fingerprint:
            npz = np.load(npz_path)
            model = PoissonRegressionGoalsMeanImpute()
            for k in ["beta_home", "beta_away", "home_mean", "home_std", "away_mean", "away_std"]:
                setattr(model, k, npz[k])
            model.home_adv, model.theta = float(npz["home_adv"]), float(npz["theta"])
            print(f"Using frozen model {name} (trained {meta['trained_at']} on {meta['n_train']} matches)")
            return model, meta
        print(f"Training data for {name} changed since it was saved -- retraining.")

    print(f"Training frozen model on {len(train_df)} matches ({train_seasons[0]}..{train_seasons[-1]})...")
    model = PoissonRegressionGoalsMeanImpute().fit(train_df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(npz_path, beta_home=model.beta_home, beta_away=model.beta_away,
             home_adv=model.home_adv, theta=model.theta,
             home_mean=model.home_mean, home_std=model.home_std,
             away_mean=model.away_mean, away_std=model.away_std)
    meta = {"model": name, "train_seasons": list(train_seasons), "n_train": len(train_df),
            "training_fingerprint": fingerprint,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "home_adv": round(model.home_adv, 6), "theta": round(model.theta, 6)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved to {os.path.relpath(npz_path, ROOT)}")
    return model, meta


def save_live_season(played_rows, current_season):
    raw_path = os.path.join(RAW_DIR, f"pl{current_season}_live.csv")
    out = (played_rows
           .merge(build_odds_frame(raw_path), on=["Date", "HomeTeam", "AwayTeam"], how="left")
           .merge(build_bet365_frame(raw_path), on=["Date", "HomeTeam", "AwayTeam"], how="left"))
    out[DATASET_COLUMNS].to_csv(LIVE_SEASON_FILE, index=False)
    print(f"  Saved {len(out)} played {current_season} matches to {os.path.relpath(LIVE_SEASON_FILE, ROOT)}")


def scoreline_grid(lam1, lam2, lam3, max_goals=MAX_GOALS_GRID):
    xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
    x, y = xs.ravel().astype(float), ys.ravel().astype(float)
    grid = np.exp(_bivpois_logpmf(x, y, np.full_like(x, lam1), np.full_like(x, lam2), lam3))
    return grid.reshape(max_goals + 1, max_goals + 1)


def lambdas(model, df):
    X_home = df[HOME_COVARIATES].astype(float).fillna(pd.Series(model.home_mean, index=HOME_COVARIATES)).values
    X_away = df[AWAY_COVARIATES].astype(float).fillna(pd.Series(model.away_mean, index=AWAY_COVARIATES)).values
    lam1 = np.exp(model.home_adv + ((X_home - model.home_mean) / model.home_std) @ model.beta_home)
    lam2 = np.exp(((X_away - model.away_mean) / model.away_std) @ model.beta_away)
    return lam1, lam2, float(np.exp(model.theta))


def fair(p):
    return round(1 / p, 2) if p > 1e-12 else None


def write_status(status, message, **extra):
    os.makedirs(os.path.dirname(STATUS_JSON), exist_ok=True)
    with open(STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "status": status, "message": message, **extra}, f, indent=2)


def input_signature(played, xg):
    """Fingerprint of everything a prediction depends on that can change during
    the season: results + match stats, xG, and the manual Elo seeds."""
    h = hashlib.sha1()
    if played is not None:
        h.update(played.sort_values(["Date", "HomeTeam"]).to_csv(index=False).encode())
    h.update(xg.sort_values(["Team", "Date_str"]).to_csv(index=False).encode())
    with open(MANUAL_SEEDS_FILE, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def already_predicted(upcoming, signature):
    if not os.path.exists(LOG_CSV):
        return False
    log = pd.read_csv(LOG_CSV)
    if "data_signature" not in log.columns:
        return False
    done = set(zip(log.loc[log["data_signature"] == signature, "home_team"],
                   log.loc[log["data_signature"] == signature, "away_team"]))
    return all((h, a) in done for h, a in zip(upcoming["HomeTeam"], upcoming["AwayTeam"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=None, help="round to predict (default: next upcoming)")
    parser.add_argument("--force", action="store_true", help="predict even if a data source is behind")
    parser.add_argument("--skip-if-unchanged", action="store_true",
                        help="don't log new predictions if these fixtures were already predicted from the same inputs")
    args = parser.parse_args()

    history_seasons = completed_seasons()
    current_season = next_season(history_seasons[-1])
    train_seasons = history_seasons[-WINDOW:]
    now = now_uk()
    print(f"=== {current_season} live run, {now:%Y-%m-%d %H:%M} UK time ===")
    print(f"Training window: {train_seasons} + {current_season} played so far")

    print("Fetching data...")
    fixtures = fetch_fixtures(current_season)
    played = fetch_played_matches(current_season)
    xg = fetch_understat(current_season)
    history = load_history_matches(history_seasons)

    problems = data_problems(fixtures, played, xg, history)
    if problems:
        print("\nData not ready:" + "".join(f"\n  - {p}" for p in problems))
        if not args.force:
            write_status("not_ready", "Waiting for the data sources to catch up with the latest results.",
                         problems=problems)
            print("Stopping -- rerun once the sources catch up, or pass --force to predict anyway.")
            raise SystemExit(NOT_READY_EXIT)
        print("  --force given: predicting anyway.")

    round_no, upcoming = select_fixtures(fixtures, args.round, now)

    # Elo: one seed per team (its rating at its first match in our data),
    # everything after that computed from results. The processed datasets are
    # refreshed with the same ratings so training and prediction match.
    upcoming_fx = upcoming[["Date", "Time", "HomeTeam", "AwayTeam"]]
    elo_seeds = seed_ratings(
        pd.concat([history, played, upcoming_fx], ignore_index=True),
        pd.read_csv(ELO_FILE), fetch=update_elo_cache,
    )
    refresh_processed_elo(elo_seeds, history)

    played_rows, predict_df = build_live_frames(
        history_seasons, current_season, played, upcoming_fx, elo_seeds,
    )
    if played is not None:
        save_live_season(played_rows, current_season)

    if round_no is None:
        print("No upcoming fixtures -- season finished.")
        write_status("season_finished", "No upcoming fixtures -- the season is finished.")
        return
    if upcoming.empty:
        print(f"Round {round_no} has no unplayed fixtures -- nothing to predict.")
        write_status("nothing_to_predict", f"Round {round_no} has no unplayed fixtures.", round=round_no)
        return
    train_df = training_history(train_seasons, history_seasons, elo_seeds)
    model, model_meta = frozen_model(train_df, train_seasons)
    print(f"  home_adv={model.home_adv:.4f}  theta={model.theta:.4f}  "
          f"({len(played_rows)} {current_season} matches feed the features, not the fit)")

    # Same inputs AND same model -> same predictions, nothing new to log.
    signature = input_signature(played, xg) + "-" + model_meta["training_fingerprint"][:6]
    if args.skip_if_unchanged and already_predicted(upcoming, signature):
        print(f"Round {round_no} already predicted from these exact inputs -- nothing new to log.")
        write_status("unchanged", f"Round {round_no} already predicted from the latest data.", round=round_no)
        return
    extra = upcoming[upcoming["RoundNumber"] != round_no]
    if not extra.empty:
        print(f"Also predicting {len(extra)} fixture(s) from other rounds kicking off before round {round_no} ends: "
              + ", ".join(f"{r.HomeTeam} v {r.AwayTeam} (round {r.RoundNumber})" for r in extra.itertuples()))

    missing = predict_df[HOME_COVARIATES + AWAY_COVARIATES].isna().sum()
    if missing.any():
        print("  Missing covariates in prediction rows (mean-imputed):", missing[missing > 0].to_dict())

    proba = model.predict_proba(predict_df)
    lam1, lam2, lam3 = lambdas(model, predict_df)

    run_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fixture_info = upcoming.set_index(["HomeTeam", "AwayTeam"])
    rows = []
    for i, fx in enumerate(predict_df.itertuples(index=False)):
        p_h, p_d, p_a = proba[i]
        grid = scoreline_grid(lam1[i], lam2[i], lam3)
        rows.append({
            "run_timestamp": run_ts,
            "season": current_season,
            "round": int(fixture_info.loc[(fx.HomeTeam, fx.AwayTeam), "RoundNumber"]),
            "kickoff": fixture_info.loc[(fx.HomeTeam, fx.AwayTeam), "Kickoff"],
            "home_team": fx.HomeTeam,
            "away_team": fx.AwayTeam,
            "p_home": round(p_h, 4), "p_draw": round(p_d, 4), "p_away": round(p_a, 4),
            "fair_odds_home": fair(p_h), "fair_odds_draw": fair(p_d), "fair_odds_away": fair(p_a),
            "lambda_home": round(lam1[i], 4), "lambda_away": round(lam2[i], 4), "lambda_shared": round(lam3, 4),
            "home_elo": fx.Home_Elo, "away_elo": fx.Away_Elo,
            "n_train": len(train_df),
            "model": model_meta["model"],
            "model_trained_at": model_meta["trained_at"],
            "n_current_season_played": len(played_rows),
            "data_signature": signature,
            "scoreline_fair_odds_json": json.dumps({
                f"{h}-{a}": fair(grid[h, a]) for h in range(grid.shape[0]) for a in range(grid.shape[1])
            }),
        })

    log_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    if os.path.exists(LOG_CSV) and pd.read_csv(LOG_CSV, nrows=0).columns.tolist() != log_df.columns.tolist():
        # Log schema changed -- rewrite instead of appending misaligned columns.
        pd.concat([pd.read_csv(LOG_CSV), log_df], ignore_index=True)[log_df.columns].to_csv(LOG_CSV, index=False)
    else:
        log_df.to_csv(LOG_CSV, mode="a", header=not os.path.exists(LOG_CSV), index=False)
    print(f"\nAppended {len(log_df)} predictions to {os.path.relpath(LOG_CSV, ROOT)}")
    write_status("predicted", f"Predicted {len(log_df)} fixture(s) for round {round_no}.",
                 round=round_no, forced=bool(problems), problems=problems)

    print(f"\n=== Round {round_no} ===")
    for r in log_df.itertuples():
        tag = "" if r.round == round_no else f"  (round {r.round})"
        print(f"{r.home_team:>16} vs {r.away_team:<16}  "
              f"H {r.p_home:6.1%} ({r.fair_odds_home:>5})   "
              f"D {r.p_draw:6.1%} ({r.fair_odds_draw:>5})   "
              f"A {r.p_away:6.1%} ({r.fair_odds_away:>5}){tag}")


if __name__ == "__main__":
    main()
