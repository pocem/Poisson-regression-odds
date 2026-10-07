"""
Round-by-round odds maker, per league (league settings: src/leagues.py). For
each league, every run:

  1. pulls fresh fixtures, played-match stats and Understat xG (sources.py),
  2. recomputes Elo for every match from each team's seed rating
     (src/pipeline/compute_elo.py -- ClubElo is only contacted to seed a team
     we have no rating for) and rebuilds features for the live season (features.py),
  3. predicts the next round with the league's FROZEN bivariate Poisson model --
     trained once on its 3 most recent completed seasons (WINDOW = 3) and saved
     as data/models/<league>.npz; it is not refit during the season. Only the
     features (Elo, xG, form, PPG) move from round to round. The saved model is
     refit only when its training data changes (new season, or an edited
     manual Elo seed),
  4. appends the predictions to data/leagues/<league>/predictions/predictions_log.csv.

Usage, from anywhere:
    python src/live/predict_round.py                          # every league, next upcoming round
    python src/live/predict_round.py --league bundesliga      # one league
    python src/live/predict_round.py --league premier_league --round 7   # a specific round
    python src/live/predict_round.py --force                  # predict even if a source is behind
    python src/live/predict_round.py --skip-if-unchanged      # no new rows if the inputs haven't changed

Before predicting, every played match in the fixture feed must already be in
football-data.co.uk (stats) and Understat (xG), and team names must line up
across sources -- otherwise that league stops rather than predict from stale
features (exit code 3 when running a single league). football-data usually
lags results by a day or two.

Season rollover: once a season is finished, save its final football-data.co.uk
file as data/leagues/<league>/raw/<season>.csv (e.g. 26-27.csv). The live season
is always the one after the newest completed raw file, and training uses the 3 newest.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "football_odds"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from leagues import leagues_from_arg  # noqa: E402
from Bivariate_Poisson import (  # noqa: E402
    PoissonRegressionGoalsMeanImpute, _bivpois_logpmf, HOME_COVARIATES, AWAY_COVARIATES, DATASET_COLUMNS,
)
from sources import (  # noqa: E402
    TBC_WINDOW_DAYS, align_xg_dates, now_uk, fetch_fixtures, fetch_played_matches, fetch_understat,
    update_elo_cache,
)
from features import build_live_frames  # noqa: E402
from process_season_data import load_data  # noqa: E402
from add_bet365_odds import build_bet365_frame  # noqa: E402
from compute_elo import completed_seasons, load_history_matches, seed_ratings, refresh_processed_elo  # noqa: E402

WINDOW = 3
MAX_GOALS_GRID = 6
NOT_READY_EXIT = 3  # distinct from a crash (1), so automation can treat "wait for data" as normal
REARRANGED_DAYS = 5  # a round's games span Fri-Mon; further out = rearranged


class NotReady(Exception):
    """A data source hasn't caught up with the latest results yet."""


def next_season(season):
    a = int(season.split("-")[0]) + 1
    return f"{a:02d}-{a + 1:02d}"


def rel(path):
    return os.path.relpath(path, ROOT)


def select_fixtures(fixtures, requested, now):
    """(target round, fixtures to predict).

    A fixture is "rearranged" if it kicks off more than REARRANGED_DAYS from
    its round's median kickoff (postponed and re-dated, or brought forward).
    Target = the round of the earliest upcoming fixture that isn't rearranged
    -- so mid-round it's the rest of the current round, never the next one.
    Rearranged fixtures kicking off before the target round ends are predicted
    with it. Postponed games with no new date (kickoff in the past, unplayed)
    are skipped. A fixture whose kickoff time isn't announced yet (KickoffTBC)
    stays upcoming until the end of its weekend."""
    unplayed = fixtures[~fixtures["Played"]]
    if requested is not None:
        return requested, unplayed[unplayed["RoundNumber"] == requested]
    tbc = unplayed["KickoffTBC"] if "KickoffTBC" in unplayed else False
    still_upcoming = unplayed["Kickoff"] + pd.to_timedelta(tbc * TBC_WINDOW_DAYS, unit="D") >= now
    upcoming = unplayed[still_upcoming].sort_values("Kickoff")
    if upcoming.empty:
        return None, upcoming
    round_median = fixtures.groupby("RoundNumber")["Kickoff"].median()
    rearranged = (upcoming["Kickoff"] - upcoming["RoundNumber"].map(round_median)).abs() > pd.Timedelta(days=REARRANGED_DAYS)
    regular = upcoming[~rearranged]
    target = int((regular if not regular.empty else upcoming).iloc[0]["RoundNumber"])
    last_kickoff = upcoming.loc[upcoming["RoundNumber"] == target, "Kickoff"].max()
    return target, upcoming[(upcoming["RoundNumber"] == target) | (rearranged & (upcoming["Kickoff"] <= last_kickoff))]


def data_problems(league, fixtures, played, xg, history):
    """Reasons the inputs aren't ready to predict from. Empty list = ready."""
    problems = []
    feed_teams = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    fd_teams = set() if played is None else set(played["HomeTeam"]) | set(played["AwayTeam"])

    # Team names: every source must use the fixture feed's (canonical) names.
    for t in sorted(fd_teams - feed_teams):
        problems.append(f"football-data.co.uk team '{t}' isn't in the fixture feed "
                        f"-- add the feed's spelling to {league.id}'s fixturedownload_names in src/leagues.py")
    for t in sorted(set(xg["Team"]) - feed_teams):
        problems.append(f"Understat team '{t}' isn't in the fixture feed "
                        f"-- add it to {league.id}'s understat_names in src/leagues.py")
    known = (set(history["HomeTeam"]) | set(history["AwayTeam"]) | fd_teams
             | set(pd.read_csv(league.elo_file)["team"]) | set(pd.read_csv(league.manual_seeds_file)["team"]))
    for t in sorted(feed_teams - known):
        problems.append(f"fixture-feed team '{t}' isn't known to football-data.co.uk, ClubElo or the manual seeds "
                        f"-- a spelling mismatch (fix {league.id}'s fixturedownload_names in src/leagues.py) "
                        f"or a new team (add its starting Elo to {rel(league.manual_seeds_file)})")

    # Freshness: every result the fixture feed has must be in football-data
    # (stats) and Understat (xG), or the features are a round out of date.
    feed_played = fixtures[fixtures["Played"]]
    fd_pairs = set() if played is None else set(zip(played["HomeTeam"], played["AwayTeam"]))
    behind = [f"{r.HomeTeam} v {r.AwayTeam} (round {r.RoundNumber})"
              for r in feed_played.itertuples() if (r.HomeTeam, r.AwayTeam) not in fd_pairs]
    if behind:
        problems.append(f"football-data.co.uk is missing {len(behind)} played match(es): {', '.join(behind)}")

    if played is not None:
        dates = played["Date"].dt.strftime("%Y-%m-%d")
        xg = align_xg_dates(xg, list(zip(played["HomeTeam"], dates)) + list(zip(played["AwayTeam"], dates)))
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


def training_history(league, train_seasons, history_seasons, elo_seeds):
    """Completed-season training rows from the league's processed dataset.
    Seasons not in that file yet (e.g. after a rollover) are rebuilt from raw."""
    processed = pd.read_csv(league.processed_file, parse_dates=["Date"])
    hist = processed[processed["Season"].isin(train_seasons)]
    missing = [s for s in train_seasons if s not in set(hist["Season"])]
    for season in missing:
        played = load_data(league.raw_file(season))
        prior = [h for h in history_seasons if h < season]
        rebuilt, _ = build_live_frames(league, prior, season, played, played.iloc[0:0], elo_seeds)
        hist = pd.concat([hist, rebuilt], ignore_index=True)
    return hist


def frozen_model(league, train_df, train_seasons):
    """The league's frozen model: loaded from data/models/<league>.npz if it was
    trained on exactly this data, otherwise trained now and saved. Returns (model, meta)."""
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"] + list(dict.fromkeys(HOME_COVARIATES + AWAY_COVARIATES))
    fingerprint = hashlib.sha1(_stable_csv(train_df[cols])).hexdigest()[:12]

    if os.path.exists(league.model_json) and os.path.exists(league.model_npz):
        with open(league.model_json, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("training_fingerprint") == fingerprint:
            npz = np.load(league.model_npz)
            model = PoissonRegressionGoalsMeanImpute()
            for k in ["beta_home", "beta_away", "home_mean", "home_std", "away_mean", "away_std"]:
                setattr(model, k, npz[k])
            model.home_adv, model.theta = float(npz["home_adv"]), float(npz["theta"])
            print(f"Using frozen model {rel(league.model_npz)} (trained {meta['trained_at']} on {meta['n_train']} matches)")
            return model, meta
        print(f"Training data for {rel(league.model_npz)} changed since it was saved -- retraining.")

    print(f"Training frozen model on {len(train_df)} matches ({train_seasons[0]}..{train_seasons[-1]})...")
    model = PoissonRegressionGoalsMeanImpute().fit(train_df)
    os.makedirs(os.path.dirname(league.model_npz), exist_ok=True)
    np.savez(league.model_npz, beta_home=model.beta_home, beta_away=model.beta_away,
             home_adv=model.home_adv, theta=model.theta,
             home_mean=model.home_mean, home_std=model.home_std,
             away_mean=model.away_mean, away_std=model.away_std)
    meta = {"model": league.id, "league": league.name, "train_seasons": list(train_seasons),
            "n_train": len(train_df), "training_fingerprint": fingerprint,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "home_adv": round(model.home_adv, 6), "theta": round(model.theta, 6)}
    with open(league.model_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved to {rel(league.model_npz)}")
    return model, meta


def save_live_season(league, played_rows, current_season):
    """Played live-season matches with features + Bet365 odds, rewritten every
    run -- Bivariate_Poisson.py appends it to the history to evaluate round by round."""
    out = played_rows.merge(build_bet365_frame(league.live_raw_file(current_season)),
                            on=["Date", "HomeTeam", "AwayTeam"], how="left")
    out[DATASET_COLUMNS].to_csv(league.live_season_file, index=False)
    print(f"  Saved {len(out)} played {current_season} matches to {rel(league.live_season_file)}")


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


def write_status(league, status, message, **extra):
    """Outcome of the latest run (predicted / not ready / unchanged ...), shown on the site."""
    os.makedirs(os.path.dirname(league.status_json), exist_ok=True)
    with open(league.status_json, "w", encoding="utf-8") as f:
        json.dump({"run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "status": status, "message": message, **extra}, f, indent=2)


def _stable_csv(df):
    """CSV bytes that are identical on every machine: fixed line endings
    (pandas writes CRLF on Windows, LF on Linux) and rounded floats."""
    floats = df.select_dtypes("float").columns
    return df.assign(**{c: df[c].round(6) for c in floats}).to_csv(index=False, lineterminator="\n").encode()


def input_signature(league, played, xg):
    """Fingerprint of everything a prediction depends on that can change during
    the season: results + match stats, xG, and the manual Elo seeds."""
    h = hashlib.sha1()
    if played is not None:
        h.update(_stable_csv(played.sort_values(["Date", "HomeTeam"])))
    h.update(_stable_csv(xg.sort_values(["Team", "Date_str"])))
    with open(league.manual_seeds_file, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def already_predicted(league, upcoming, signature):
    if not os.path.exists(league.log_csv):
        return False
    log = pd.read_csv(league.log_csv)
    if "data_signature" not in log.columns:
        return False
    done = set(zip(log.loc[log["data_signature"] == signature, "home_team"],
                   log.loc[log["data_signature"] == signature, "away_team"]))
    return all((h, a) in done for h, a in zip(upcoming["HomeTeam"], upcoming["AwayTeam"]))


def run_league(league, args):
    history_seasons = completed_seasons(league)
    current_season = next_season(history_seasons[-1])
    train_seasons = history_seasons[-WINDOW:]
    now = now_uk()
    print(f"\n=== {league.name} {current_season} live run, {now:%Y-%m-%d %H:%M} UK time ===")
    print(f"Model trained on {train_seasons}; features updated with {current_season} played so far")

    print("Fetching data...")
    fixtures = fetch_fixtures(league, current_season)
    played = fetch_played_matches(league, current_season)
    xg = fetch_understat(league, current_season)
    history = load_history_matches(league, history_seasons)

    problems = data_problems(league, fixtures, played, xg, history)
    if problems:
        print("\nData not ready:" + "".join(f"\n  - {p}" for p in problems))
        if not args.force:
            write_status(league, "not_ready", "Waiting for the data sources to catch up with the latest results.",
                         problems=problems)
            raise NotReady
        print("  --force given: predicting anyway.")

    round_no, upcoming = select_fixtures(fixtures, args.round, now)

    # Elo: one seed per team spell, everything after that computed from results.
    # The processed dataset is refreshed with the same ratings so training and
    # prediction match.
    upcoming_fx = upcoming[["Date", "Time", "HomeTeam", "AwayTeam"]]
    elo_seeds = seed_ratings(league, pd.concat([history, played, upcoming_fx], ignore_index=True),
                             fetch=lambda dates: update_elo_cache(league, dates))
    refresh_processed_elo(league, elo_seeds, history)

    played_rows, predict_df = build_live_frames(league, history_seasons, current_season, played, upcoming_fx, elo_seeds)
    if played is not None:
        save_live_season(league, played_rows, current_season)

    if round_no is None:
        print("No upcoming fixtures -- season finished.")
        write_status(league, "season_finished", "No upcoming fixtures -- the season is finished.")
        return
    if upcoming.empty:
        print(f"Round {round_no} has no unplayed fixtures -- nothing to predict.")
        write_status(league, "nothing_to_predict", f"Round {round_no} has no unplayed fixtures.", round=round_no)
        return
    train_df = training_history(league, train_seasons, history_seasons, elo_seeds)
    model, model_meta = frozen_model(league, train_df, train_seasons)
    print(f"  home_adv={model.home_adv:.4f}  theta={model.theta:.4f}  "
          f"({len(played_rows)} {current_season} matches feed the features, not the fit)")

    # Same inputs AND same model -> same predictions, nothing new to log.
    signature = input_signature(league, played, xg) + "-" + model_meta["training_fingerprint"][:6]
    if args.skip_if_unchanged and already_predicted(league, upcoming, signature):
        print(f"Round {round_no} already predicted from these exact inputs -- nothing new to log.")
        write_status(league, "unchanged", f"Round {round_no} already predicted from the latest data.", round=round_no)
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
    log_csv = league.log_csv
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    if os.path.exists(log_csv) and pd.read_csv(log_csv, nrows=0).columns.tolist() != log_df.columns.tolist():
        # Log schema changed -- rewrite instead of appending misaligned columns.
        pd.concat([pd.read_csv(log_csv), log_df], ignore_index=True)[log_df.columns].to_csv(log_csv, index=False)
    else:
        log_df.to_csv(log_csv, mode="a", header=not os.path.exists(log_csv), index=False)
    print(f"\nAppended {len(log_df)} predictions to {rel(log_csv)}")
    write_status(league, "predicted", f"Predicted {len(log_df)} fixture(s) for round {round_no}.",
                 round=round_no, forced=bool(problems), problems=problems)

    print(f"\n=== {league.name} round {round_no} ===")
    for r in log_df.itertuples():
        tag = "" if r.round == round_no else f"  (round {r.round})"
        print(f"{r.home_team:>16} vs {r.away_team:<16}  "
              f"H {r.p_home:6.1%} ({r.fair_odds_home:>5})   "
              f"D {r.p_draw:6.1%} ({r.fair_odds_draw:>5})   "
              f"A {r.p_away:6.1%} ({r.fair_odds_away:>5}){tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="all", help="league id (see src/leagues.py), or 'all' (default)")
    parser.add_argument("--round", type=int, default=None, help="round to predict (default: next upcoming)")
    parser.add_argument("--force", action="store_true", help="predict even if a data source is behind")
    parser.add_argument("--skip-if-unchanged", action="store_true",
                        help="don't log new predictions if these fixtures were already predicted from the same inputs")
    args = parser.parse_args()
    leagues = leagues_from_arg(args.league)

    not_ready, crashed = [], []
    for league in leagues:
        try:
            run_league(league, args)
        except NotReady:
            not_ready.append(league.name)
            print(f"{league.name}: stopping -- rerun once the sources catch up, or pass --force to predict anyway.")
        except Exception:
            # Keep going so one league's failure doesn't block the others.
            crashed.append(league.name)
            traceback.print_exc()

    if crashed:
        raise SystemExit(f"Failed: {', '.join(crashed)}")
    if not_ready and len(leagues) == 1:
        raise SystemExit(NOT_READY_EXIT)


if __name__ == "__main__":
    main()
