## Live round-by-round predictions

Each league has its own frozen bivariate Poisson model. Leagues so far: **Premier League**
(`premier_league`) and **Bundesliga** (`bundesliga`), configured in `src/leagues.py`.

```
python src/live/predict_round.py                           # every league, next upcoming round
python src/live/predict_round.py --league bundesliga       # one league
python src/live/predict_round.py --league premier_league --round 7   # a specific round
python src/live/predict_round.py --force                   # predict even if a data source is behind
```

Each run pulls fresh fixtures (fixturedownload.com), match stats (football-data.co.uk) and
xG (Understat), recomputes Elo and rebuilds the features, then predicts with the league's frozen
model -- trained once on its last 3 completed seasons and saved as `data/models/<league>.npz`,
not refit during the season -- and appends the round's probabilities and fair odds to the
league's `predictions/predictions_log.csv`. A model is refit only when its training data changes
(a new season, or an edited manual Elo seed).

A league is skipped for that run (exit code 3 when running a single league) if
football-data.co.uk or Understat doesn't have every result the fixture feed already shows, or if
team names don't line up across sources -- rather than predict from features a round out of
date. football-data.co.uk usually lags results by a day or two.

Evaluate a league's model against Bet365: `python src/football_odds/Bivariate_Poisson.py --league bundesliga`

## Data layout

```
data/
  leagues/<league>/
    raw/           football-data.co.uk season files: 23-24.csv ... and <live season>_live.csv
    processed/     all_seasons.csv (training data), live_season.csv (played live-season matches)
    external/      elo_df.csv (ClubElo ratings by date), elo_seeds_manual.csv (hand-entered starting Elos)
    cache/         understat_<year>.json (xG), upcoming_odds.csv (Bet365 odds for upcoming fixtures)
    predictions/   predictions_log.csv, last_run.json, <season>_model_odds.csv
  models/          <league>.npz + <league>.json -- each league's frozen model
```

After a season ends, save its final football-data.co.uk file as
`data/leagues/<league>/raw/<season>.csv` (e.g. `26-27.csv`); the pipeline then moves on to the
next season automatically.

## Adding a league

1. Add an entry to `LEAGUES` in `src/leagues.py`: football-data.co.uk division code,
   fixturedownload.com feed name, Understat league name, ClubElo country, number of teams, and
   name maps from each source to football-data.co.uk's team names.
2. Download its history: `python src/pipeline/download_history.py --league <id> --seasons 23-24 24-25 25-26`
3. Put ClubElo ratings in `external/elo_df.csv` and/or starting Elos in `external/elo_seeds_manual.csv`.
4. Build the training data: `python src/pipeline/build_dataset.py --league <id>`
5. Run `python src/live/predict_round.py --league <id>` -- it trains the model and predicts.
   The website picks the league up automatically.

## Elo

Elo is computed in `src/pipeline/compute_elo.py`, not looked up: each team starts from its ClubElo
rating at its first match in the data (and again when it returns after relegation), and is then
updated after every league match with the standard Elo formula (K=20, no home advantage).
Hand-entered starting ratings in the league's `external/elo_seeds_manual.csv` (team, season, elo)
take priority; ClubElo is only contacted to seed a team with no rating yet.

## Website

`.github/workflows/update-site.yml` runs the pipeline for every league each day at 06:00 UTC (plus
Tuesday and Friday evenings, when bookmaker odds for the next round are posted), commits the
updated data, and publishes a static page to GitHub Pages with a league switcher and, per league:

- the next round's probabilities, fair odds, likeliest scorelines and expected goals,
  next to Bet365's odds (margin removed) once they're posted, with the model's edge;
- season performance against Bet365 (log loss, Brier score, favourite-won rate, log loss by round);
- every played match with model and Bet365 odds (margin removed), the result and each forecaster's log loss;
- the current Elo table.

A run only logs new predictions when the inputs changed (`--skip-if-unchanged`), and a
"data not ready" stop is reported on the page rather than failing the workflow.

The page's source is `src/site/page_template.html` (layout, styling, charts); edit that to change
the site. `src/site/build_site.py` fills it with the data and writes the finished page to
`generated_site/index.html` -- build output that is never committed and safe to delete.

Preview locally: `python src/site/build_site.py`, then open `generated_site/index.html`
(add `#bundesliga` to the address to open a specific league).
One-time setup on GitHub: Settings → Pages → Source: **GitHub Actions**.
