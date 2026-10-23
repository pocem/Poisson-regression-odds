## Live round-by-round predictions

Each league has its own frozen bivariate Poisson model. Leagues: **Premier League**
(`premier_league`), **Bundesliga** (`bundesliga`), **La Liga** (`la_liga`), **Serie A** (`serie_a`)
and **Ligue 1** (`ligue_1`), configured in `src/leagues.py`.

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
4. Build the training data: `python src/pipeline/build_dataset.py --league <id>` (an existing
   dataset is only replaced with `--overwrite` -- the Premier League's came from the ML project's
   longer history and can't be rebuilt identically)
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

- the next round's probabilities, odds, likeliest scorelines and expected goals,
  next to Bet365's odds once they're posted, with the difference between the two;
- season performance against Bet365 (log loss, Brier score, favourite-won rate, log loss by round);
- every played match with model and Bet365 odds, the result and each forecaster's log loss;
- the current Elo table.

Above the leagues, the page lists every upcoming outcome with an expected value of 10% or more
at Bet365's odds, and a short round summary written by Google's Gemini on the Gemini API's free
tier (`src/site/round_summary.py`) from the predictions, Bet365's odds and the latest results.
It uses Gemini 3.8 Flash, retrying and then falling back to other free models when that one is
overloaded. The summary is saved to `data/summary/round_summary.json` and only rewritten when that
data changes; it needs a `GEMINI_API_KEY` repository secret (an API key from
aistudio.google.com; Settings → Secrets and variables → Actions). Without it, the site builds
as usual and keeps the last saved summary.

A run only logs new predictions when the inputs changed (`--skip-if-unchanged`), and a
"data not ready" stop is reported on the page rather than failing the workflow.

The page's source is `src/site/page_template.html` (layout, styling, charts); edit that to change
the site. `src/site/build_site.py` fills it with the data and writes the finished page to
`generated_site/index.html` -- build output that is never committed and safe to delete.

## Accounts and bet log

Visitors can create an account on the page and log the bets they place from the next-round cards and
the value table. Bets are settled from the site's own results.

Accounts and the log live in Supabase (project `rstembhxuncyttzdmssl`); the site stays on GitHub
Pages and talks to Supabase from the browser with the project's publishable key, which is meant to
be public. Supabase Auth stores passwords as bcrypt hashes. One-time setup in the Supabase dashboard:

1. SQL Editor: run `supabase/schema.sql` (the `bets` table and its row level security: each user
   can only read, add and delete their own picks).
2. Authentication → Providers → Email: minimum password length 8, required characters
   "lowercase, uppercase letters, digits and symbols" (the page checks the same rules).
3. Authentication → URL Configuration: set the Site URL to the GitHub Pages address, so the
   confirmation email links back to the site.
4. SQL Editor: run `supabase/leaderboard.sql` (the `results` table and the `leaderboard()`
   function, which scores every logged bet and returns per-player totals only).
5. Project Settings → API Keys: create a secret key and add it as the `SUPABASE_SECRET_KEY`
   repository secret on GitHub. After each build, `src/site/sync_results.py` uses it to upload the
   season's results, which settle the bets on the leaderboard. Never put it in the page.

The leaderboard (signed-in users only) ranks players by score: the profit if every bet had been 1
unit, so stakes don't decide the ranking. Skipped picks don't count.

Logging a bet means it was placed at the Bet365 odds the site shows: the odds aren't typed in, and
`supabase/locked_odds.sql` makes the database overwrite every new bet's odds with the site's own
(uploaded to `match_odds` after each build by `src/site/sync_results.py`) and reject bets after
kickoff; a bet can be deleted only before its match starts. Run that SQL once, after the other two.

"My page" (`#me`) shows a signed-in user's own stats, all at the odds they bet at: score and rank,
return, profit at their stakes, wins against what the odds implied, how their bets from the value
list do against every value pick, the model's rating of their bets, score over time and a
breakdown by odds.

The page is split into four pages under a navigation bar: **This round** (`#round/<league>`, the
default: round summary, value table and the next round's predictions), **Season stats**
(`#season/<league>`: season performance, team charts, every match, Elo), **Leaderboard**
(`#leaderboard`) and **My page** (`#me`).

Preview locally: `python src/site/build_site.py`, then open `generated_site/index.html`
(add e.g. `#season/bundesliga` to the address to open a specific page and league).
One-time setup on GitHub: Settings → Pages → Source: **GitHub Actions**.
