"""
Builds the static website (GitHub Pages) from the pipeline's outputs, with one
section per league (src/leagues.py) and a league switcher:

  - next round   latest logged predictions (predictions/predictions_log.csv)
                 + current Bet365 odds from football-data.co.uk/fixtures.csv
  - season       every played match: the live forecast logged before kickoff,
                 or the frozen-model backtest if there was none, vs Bet365
                 (processed/live_season.csv)
  - Elo table    current self-computed ratings
  - status       outcome of the last pipeline run (predictions/last_run.json)

Fills the template next to this script (page_template.html -- the page's
source: layout, styling, charts) with the data, and writes the finished,
self-contained page to generated_site/index.html. That folder is build output:
safe to delete, never committed, rebuilt on every run. Open the file straight
from disk to preview:

    python src/site/build_site.py
"""

import io
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "live"))
from leagues import LEAGUES  # noqa: E402
from sources import HEADERS, fetch_fixtures, now_uk  # noqa: E402
from predict_round import WINDOW, next_season  # noqa: E402
from backtest import season_backtest  # noqa: E402
from compute_elo import completed_seasons, load_history_matches, seed_ratings, compute_elo  # noqa: E402
from process_season_data import load_data  # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page_template.html")
# Site icon: football (soccer ball) by flomar, CC0 / public domain, via Wikimedia Commons.
FAVICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.svg")
OUT_DIR = os.path.join(ROOT, "generated_site")
UPCOMING_ODDS_URL = "https://www.football-data.co.uk/fixtures.csv"
OUTCOMES = ["H", "D", "A"]


def fair_probs(odds):
    """Bookmaker odds -> margin-free probabilities (proportional de-vig)."""
    odds = np.asarray(odds, dtype=float)
    if np.isnan(odds).any():
        return None
    inv = 1 / odds
    return (inv / inv.sum()).tolist()


def num(x, digits=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), digits)


def download_upcoming_odds():
    """football-data.co.uk's fixtures.csv: upcoming matches of every league with
    their odds. Filled on Fridays (weekend rounds) and Tuesdays (midweek)."""
    try:
        resp = requests.get(UPCOMING_ODDS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
    except (requests.RequestException, ValueError) as e:
        print(f"  WARNING: couldn't fetch {UPCOMING_ODDS_URL} ({e}); using each league's cache if present")
        return None


def upcoming_bet365_odds(league, all_odds):
    """{(home, away): [h, d, a] Bet365 odds} for the league's upcoming fixtures;
    the last non-empty download is cached in the league's cache/ folder."""
    df = pd.DataFrame()
    if all_odds is not None and "Div" in all_odds:
        df = all_odds[all_odds["Div"] == league.football_data_code]
        if not df.empty:
            df.to_csv(league.upcoming_odds_cache, index=False)
    if df.empty and os.path.exists(league.upcoming_odds_cache):
        df = pd.read_csv(league.upcoming_odds_cache)
    return {(r["HomeTeam"], r["AwayTeam"]): [num(r.get("B365" + o), 2) for o in OUTCOMES]
            for _, r in df.iterrows()}


def top_scorelines(json_str, n=6):
    odds = json.loads(json_str)
    probs = sorted(((1 / o, s) for s, o in odds.items() if o), reverse=True)[:n]
    return [{"score": s, "p": round(p, 4)} for p, s in probs]


def next_round(log, fixtures, b365_odds):
    """The latest run's predictions for fixtures that haven't been played yet."""
    if log.empty:
        return None
    latest = log[log["run_timestamp"] == log["run_timestamp"].max()]
    # Fixtures are keyed by (home, away, round): the same pairing can appear twice in a feed.
    played = set(zip(fixtures.loc[fixtures["Played"], "HomeTeam"], fixtures.loc[fixtures["Played"], "AwayTeam"],
                     fixtures.loc[fixtures["Played"], "RoundNumber"]))
    latest = latest[[(h, a, rd) not in played
                     for h, a, rd in zip(latest["home_team"], latest["away_team"], latest["round"])]]
    if latest.empty:
        return None

    tbc = dict(zip(zip(fixtures["HomeTeam"], fixtures["AwayTeam"], fixtures["RoundNumber"]), fixtures["KickoffTBC"]))
    games = []
    for r in latest.sort_values("kickoff").itertuples():
        p = [r.p_home, r.p_draw, r.p_away]
        b365 = b365_odds.get((r.home_team, r.away_team))
        b365_fair = fair_probs(b365) if b365 else None
        games.append({
            "round": int(r.round), "kickoff": str(r.kickoff), "home": r.home_team, "away": r.away_team,
            "kickoff_tbc": bool(tbc.get((r.home_team, r.away_team, r.round), False)),
            "p": p, "odds": [r.fair_odds_home, r.fair_odds_draw, r.fair_odds_away],
            "xg": [num(r.lambda_home + r.lambda_shared, 2), num(r.lambda_away + r.lambda_shared, 2)],
            "elo": [num(r.home_elo, 0), num(r.away_elo, 0)],
            "scores": top_scorelines(r.scoreline_fair_odds_json),
            "b365": b365, "b365_fair": [round(1 / x, 2) for x in b365_fair] if b365_fair else None,
            "edge": [round(p[i] - b365_fair[i], 4) for i in range(3)] if b365_fair else None,
        })
    rounds = latest["round"].value_counts()
    return {"round": int(rounds.idxmax()), "predicted_at": latest["run_timestamp"].iloc[0], "games": games}


def season_matches(league, fixtures, log):
    """Every played match with the model's pre-match probabilities. Uses the
    latest live prediction logged before kickoff when there is one (a real
    forecast), otherwise the frozen-model backtest."""
    bt = season_backtest(league, fixtures)
    live_fc = {}
    if not log.empty:
        # Compare against the feed's current kickoff (the real time once a match
        # is played), not the one logged -- that may have been a "time TBC" placeholder.
        actual = dict(zip(zip(fixtures["HomeTeam"], fixtures["AwayTeam"], fixtures["RoundNumber"]), fixtures["Kickoff"]))
        lg = log.copy()
        lg["kickoff"] = [actual.get((h, a, rd), k)
                         for h, a, rd, k in zip(lg["home_team"], lg["away_team"], lg["round"], lg["kickoff"])]
        kickoff_utc = pd.to_datetime(lg["kickoff"]).dt.tz_localize("Europe/London").dt.tz_convert("UTC")
        lg = lg[pd.to_datetime(lg["run_timestamp"], utc=True) < kickoff_utc].sort_values("run_timestamp")
        for r in lg.itertuples():
            live_fc[(r.home_team, r.away_team, r.round)] = [r.p_home, r.p_draw, r.p_away]

    rows = []
    for r in bt.itertuples():
        key = (r.HomeTeam, r.AwayTeam, int(r.RoundNumber))
        p = live_fc.get(key, [r.p_home, r.p_draw, r.p_away])
        y = OUTCOMES.index(r.FTR)
        b365 = [r.B365HomeOdds, r.B365DrawOdds, r.B365AwayOdds]
        b365_f = fair_probs(b365)
        rows.append({
            "round": int(r.RoundNumber), "kickoff": str(r.Kickoff), "home": r.HomeTeam, "away": r.AwayTeam,
            "score": [int(r.FTHG), int(r.FTAG)], "result": r.FTR,
            "source": "live" if key in live_fc else "backtest",
            "p": [round(x, 4) for x in p], "odds": [round(1 / x, 2) for x in p],
            "b365": [num(x, 2) for x in b365],
            "b365_fair": [round(1 / x, 2) for x in b365_f] if b365_f else [None] * 3,
            "pick": OUTCOMES[int(np.argmax(p))],
            "ll": {"model": -math.log(p[y]),
                   "b365": -math.log(b365_f[y]) if b365_f else None},
            "brier": {"model": sum((p[i] - (i == y)) ** 2 for i in range(3)),
                      "b365": sum((b365_f[i] - (i == y)) ** 2 for i in range(3)) if b365_f else None},
            "book_pick": OUTCOMES[int(np.argmax(b365_f))] if b365_f else None,
        })
    return rows


def round_stats(matches):
    df = pd.DataFrame([{"round": m["round"], "model": m["ll"]["model"], "b365": m["ll"]["b365"]} for m in matches])
    per = df.groupby("round")[["model", "b365"]].mean()
    n = df.groupby("round").size()
    cum = df.sort_values("round").groupby("round")[["model", "b365"]].sum().cumsum().div(n.cumsum(), axis=0)
    return [{"round": int(r), "n": int(n[r]),
             "per": {k: round(per.loc[r, k], 4) for k in per.columns},
             "cum": {k: round(cum.loc[r, k], 4) for k in cum.columns}} for r in per.index]


def summary(matches):
    def mean(key, sub):
        vals = [m[key][sub] for m in matches if m[key][sub] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "n": len(matches),
        "ll_model": mean("ll", "model"), "ll_b365": mean("ll", "b365"),
        "brier_model": mean("brier", "model"), "brier_b365": mean("brier", "b365"),
        "acc_model": round(sum(m["pick"] == m["result"] for m in matches) / len(matches), 4),
        "acc_b365": round(sum(m["book_pick"] == m["result"] for m in matches) / len(matches), 4),
    }


def elo_table(league, season, history_seasons):
    history = load_history_matches(league, history_seasons)
    live_path = league.live_raw_file(season)
    if not os.path.exists(live_path):
        return []
    played = load_data(live_path)
    seeds = seed_ratings(league, pd.concat([history, played], ignore_index=True))
    out = compute_elo(pd.concat([history, played], ignore_index=True), seeds)
    final = out.attrs["final_ratings"]
    live = out.iloc[len(history):]
    start = {}
    for r in live.sort_values(["Date", "Time"]).itertuples():
        start.setdefault(r.HomeTeam, r.Home_Elo)
        start.setdefault(r.AwayTeam, r.Away_Elo)
    rows = [{"team": t, "elo": round(final[t], 1), "change": round(final[t] - start[t], 1)} for t in start]
    return sorted(rows, key=lambda r: -r["elo"])


def league_data(league, all_odds):
    history_seasons = completed_seasons(league)
    season = next_season(history_seasons[-1])
    fixtures = fetch_fixtures(league, season)
    log = pd.read_csv(league.log_csv) if os.path.exists(league.log_csv) else pd.DataFrame()
    status = None
    if os.path.exists(league.status_json):
        with open(league.status_json, encoding="utf-8") as f:
            status = json.load(f)

    matches = season_matches(league, fixtures, log) if os.path.exists(league.live_season_file) else []
    return {
        "id": league.id,
        "name": league.name,
        "season": season,
        "rounds_total": league.rounds,
        "train_seasons": history_seasons[-WINDOW:],
        "elo_since": history_seasons[0],
        "status": status,
        "next": next_round(log, fixtures, upcoming_bet365_odds(league, all_odds)),
        "matches": matches,
        "rounds": round_stats(matches) if matches else [],
        "summary": summary(matches) if matches else None,
        "elo": elo_table(league, season, history_seasons),
    }


def main():
    all_odds = download_upcoming_odds()
    leagues = []
    for league in LEAGUES.values():
        d = league_data(league, all_odds)
        leagues.append(d)
        print(f"{league.name}: {len(d['matches'])} played matches, next round {d['next']['round'] if d['next'] else '-'}")

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_uk": f"{now_uk():%Y-%m-%d %H:%M}",
        "leagues": leagues,
    }
    html = open(TEMPLATE, encoding="utf-8").read()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html.replace("__DATA_JSON__", payload))
    shutil.copy(FAVICON, os.path.join(OUT_DIR, "favicon.svg"))
    print(f"Built {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
