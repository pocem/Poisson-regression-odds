## Live round-by-round predictions

```
python src/live/predict_round.py            # next upcoming round (+ rearranged games before it)
python src/live/predict_round.py --round 7  # a specific round
python src/live/predict_round.py --force    # predict even if a data source is behind
```

The run stops (exit code 3) if football-data.co.uk or Understat doesn't have every result
the fixture feed already shows, or if team names don't line up across sources -- rather than
predict from features a round out of date. football-data.co.uk usually lags results by a day or two.

Each run pulls fresh fixtures (fixturedownload.com), match stats (football-data.co.uk) and
xG (Understat), recomputes Elo, rebuilds features, retrains the bivariate Poisson model
on the last 3 completed seasons + every current-season match played so far, and appends
the round's probabilities and fair odds to `data/predictions/predictions_log.csv`.

After a season ends, save its final football-data.co.uk file as `data/raw/pl{season}.csv`
(e.g. `pl26-27.csv`); the pipeline then moves on to the next season automatically.

Elo is computed in `src/pipeline/compute_elo.py`, not looked up: each team starts from its ClubElo
rating at its first match in the data (and again when it returns after relegation), and is then
updated after every match with the standard Elo formula (K=20, no home advantage). Hand-entered
starting ratings in `data/external/elo_seeds_manual.csv` (team, season, elo) take priority;
ClubElo is only contacted to seed a team with no rating yet.

## Website

`.github/workflows/update-site.yml` runs the pipeline every day at 06:00 UTC (plus Tuesday and
Friday evenings, when bookmaker odds for the next round are posted), commits the updated data,
and publishes a static page to GitHub Pages with:

- the next round's probabilities, fair odds, likeliest scorelines and expected goals,
  next to Bet365 and the bookmaker average once they're posted, with the model's edge;
- season performance against Bet365 (log loss, Brier score, favourite-won rate, log loss by round);
- every played match with model and Bet365 odds, the result and each forecaster's log loss;
- the current Elo table.

A run only logs new predictions when the inputs changed (`--skip-if-unchanged`), and a
"data not ready" stop is reported on the page rather than failing the workflow.

Preview locally: `python src/site/build_site.py`, then open `_site/index.html`.
One-time setup on GitHub: Settings → Pages → Source: **GitHub Actions**.
