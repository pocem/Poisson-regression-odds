"""
League registry -- everything that differs between leagues lives here: data
source codes, team-name maps, league size, and where each league's files are.

    data/leagues/<league id>/
        raw/            football-data.co.uk season files: 23-24.csv, ..., <live season>_live.csv
        processed/      all_seasons.csv (training data), live_season.csv (played live-season matches)
        external/       elo_df.csv (ClubElo ratings by date), elo_seeds_manual.csv (hand-entered starting Elos)
        cache/          understat_<year>.json (xG), upcoming_odds.csv (Bet365 odds for upcoming fixtures)
        predictions/    predictions_log.csv, last_run.json, <season>_model_odds.csv
    data/models/<league id>.npz / .json    the league's frozen model

Canonical team names are football-data.co.uk's; every other source is mapped
onto them with the dicts below. To add a league: add an entry to LEAGUES, put
its season files in raw/ and its ClubElo ratings in external/, and run
src/pipeline/build_dataset.py --league <id>.
"""

import os
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")


@dataclass(frozen=True)
class League:
    id: str
    name: str
    football_data_code: str         # football-data.co.uk division, e.g. "E0" -> .../mmz4281/2627/E0.csv
    fixtures_slug: str              # fixturedownload.com feed, "{year}" = season start year
    understat_name: str             # understat.com/getLeagueData/<name>/<year>
    clubelo_country: str            # ClubElo "Country" column
    teams: int
    fixturedownload_names: dict = field(default_factory=dict)
    understat_names: dict = field(default_factory=dict)
    clubelo_names: dict = field(default_factory=dict)

    @property
    def rounds(self):
        return 2 * (self.teams - 1)

    @property
    def matches_per_round(self):
        return self.teams // 2

    # ---- paths ----
    @property
    def dir(self):
        return os.path.join(DATA_DIR, "leagues", self.id)

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def raw_file(self, season):
        return self.path("raw", f"{season}.csv")

    def live_raw_file(self, season):
        return self.path("raw", f"{season}_live.csv")

    def understat_cache(self, year):
        return self.path("cache", f"understat_{year}.json")

    @property
    def processed_file(self):
        return self.path("processed", "all_seasons.csv")

    @property
    def live_season_file(self):
        return self.path("processed", "live_season.csv")

    @property
    def elo_file(self):
        return self.path("external", "elo_df.csv")

    @property
    def manual_seeds_file(self):
        return self.path("external", "elo_seeds_manual.csv")

    @property
    def upcoming_odds_cache(self):
        return self.path("cache", "upcoming_odds.csv")

    @property
    def log_csv(self):
        return self.path("predictions", "predictions_log.csv")

    @property
    def status_json(self):
        return self.path("predictions", "last_run.json")

    def model_odds_csv(self, season):
        return self.path("predictions", f"{season}_model_odds.csv")

    @property
    def model_npz(self):
        return os.path.join(DATA_DIR, "models", f"{self.id}.npz")

    @property
    def model_json(self):
        return os.path.join(DATA_DIR, "models", f"{self.id}.json")


PREMIER_LEAGUE = League(
    id="premier_league",
    name="Premier League",
    football_data_code="E0",
    fixtures_slug="epl-{year}",
    understat_name="EPL",
    clubelo_country="ENG",
    teams=20,
    fixturedownload_names={"Man Utd": "Man United", "Spurs": "Tottenham"},
    understat_names={
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Queens Park Rangers": "QPR",
        "West Bromwich Albion": "West Brom",
        "Wolverhampton Wanderers": "Wolves",
    },
    clubelo_names={"Forest": "Nott'm Forest"},
)

BUNDESLIGA = League(
    id="bundesliga",
    name="Bundesliga",
    football_data_code="D1",
    fixtures_slug="bundesliga-{year}",
    understat_name="Bundesliga",
    clubelo_country="GER",
    teams=18,
    fixturedownload_names={
        "1. FC Köln": "FC Koln",
        "1. FC Union Berlin": "Union Berlin",
        "1. FC Heidenheim 1846": "Heidenheim",
        "1. FSV Mainz 05": "Mainz",
        "Bayer 04 Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund",
        "Borussia Mönchengladbach": "M'gladbach",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Augsburg": "Augsburg",
        "FC Bayern München": "Bayern Munich",
        "FC Schalke 04": "Schalke 04",
        "FC St. Pauli": "St Pauli",
        "Hamburger SV": "Hamburg",
        "Holstein Kiel": "Holstein Kiel",
        "SC Paderborn 07": "Paderborn",
        "SV Elversberg": "Elversberg",
        "SV Werder Bremen": "Werder Bremen",
        "Sport-Club Freiburg": "Freiburg",
        "SC Freiburg": "Freiburg",
        "SV Darmstadt 98": "Darmstadt",
        "TSG Hoffenheim": "Hoffenheim",
        "VfB Stuttgart": "Stuttgart",
        "VfL Bochum 1848": "Bochum",
        "VfL Wolfsburg": "Wolfsburg",
    },
    understat_names={
        "Bayer Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund",
        "Borussia M.Gladbach": "M'gladbach",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Cologne": "FC Koln",
        "FC Heidenheim": "Heidenheim",
        "Hamburger SV": "Hamburg",
        "Hertha Berlin": "Hertha",
        "Mainz 05": "Mainz",
        "RasenBallsport Leipzig": "RB Leipzig",
        "St. Pauli": "St Pauli",
        "VfB Stuttgart": "Stuttgart",
    },
    clubelo_names={
        "Bayern": "Bayern Munich",
        "Frankfurt": "Ein Frankfurt",
        "Gladbach": "M'gladbach",
        "Hertha": "Hertha",
        "Holstein": "Holstein Kiel",
        "Koeln": "FC Koln",
        "Schalke": "Schalke 04",
        "Werder": "Werder Bremen",
    },
)

LEAGUES = {lg.id: lg for lg in [PREMIER_LEAGUE, BUNDESLIGA]}


def get_league(league_id):
    try:
        return LEAGUES[league_id]
    except KeyError:
        raise SystemExit(f"Unknown league '{league_id}'. Known: {', '.join(LEAGUES)}") from None


def leagues_from_arg(value):
    """'all' -> every league; otherwise one league id."""
    return list(LEAGUES.values()) if value == "all" else [get_league(value)]
