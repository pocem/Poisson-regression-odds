import glob
import os

import pandas as pd

MATCHES_FILE = "data/processed/all_seasons.csv"
OUTPUT_FILE = "data/processed/all_seasons_with_bookies.csv"

# Columns that are match-odds aggregates/derivatives, not individual bookmakers.
# (Football-data.co.uk's Betbrain max/avg, and the modern Max/Avg/Betfair-Exchange
# columns that replaced them.)
EXCLUDED_FAMILIES = {"Max", "Avg", "BbMx", "BbAv", "BFE"}


def find_bookie_prefixes(cols):
    """Identify genuine pre-match 1X2 bookmaker odds columns in a raw season file.

    A real bookmaker triplet has matching <prefix>H/<prefix>D/<prefix>A columns
    (this alone excludes Asian handicap and over/under columns, which never
    have a draw leg). Known aggregate families are dropped explicitly, and
    closing-odds columns (e.g. B365CH, the closing twin of B365H) are dropped
    by checking whether stripping a trailing "C" yields another real triplet.
    """
    cols_set = set(cols)
    candidates = set()
    for c in cols:
        if c.endswith("H") and len(c) > 1:
            prefix = c[:-1]
            if (prefix + "D") in cols_set and (prefix + "A") in cols_set:
                candidates.add(prefix)

    genuine = set()
    for p in candidates:
        if p in EXCLUDED_FAMILIES:
            continue
        if p.endswith("C") and p[:-1] in candidates:
            continue
        genuine.add(p)

    return sorted(genuine)


def build_odds_frame(path):
    season = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)

    prefixes = find_bookie_prefixes(df.columns.tolist())
    h_cols = [p + "H" for p in prefixes]
    d_cols = [p + "D" for p in prefixes]
    a_cols = [p + "A" for p in prefixes]

    odds = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], dayfirst=True),
        "HomeTeam": df["HomeTeam"],
        "AwayTeam": df["AwayTeam"],
        "AvgHomeOdds": df[h_cols].mean(axis=1, skipna=True),
        "AvgDrawOdds": df[d_cols].mean(axis=1, skipna=True),
        "AvgAwayOdds": df[a_cols].mean(axis=1, skipna=True),
        "NumBookies": df[h_cols].notna().sum(axis=1),
    })

    print(f"{season}: averaged over {len(prefixes)} bookies ({', '.join(prefixes)})")
    return odds


def main():
    season_files = sorted(glob.glob("pl*.csv"))
    odds_frames = [build_odds_frame(f) for f in season_files]
    odds_df = pd.concat(odds_frames, ignore_index=True)

    matches = pd.read_csv(MATCHES_FILE, parse_dates=["Date"])

    merged = matches.merge(
        odds_df,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="left"
    )

    missing = merged["AvgHomeOdds"].isna().sum()
    print(f"\nMerged shape: {merged.shape}")
    print(f"Matches with no odds found: {missing} / {len(merged)}")

    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
