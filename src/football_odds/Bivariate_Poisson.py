import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from sklearn.preprocessing import label_binarize
from scipy import stats

DATA_PATH = "data/processed/all_seasons_14window_ppg_full.csv"
# Played matches of the live season, written by src/live/predict_round.py on every run.
LIVE_SEASON_PATH = "data/processed/live_season.csv"
WINDOW = 3
N_CHUNKS = 38  # one chunk per round, retrained before each
MATCHES_PER_ROUND = 10

HOME_COVARIATES = [
    "Home_Elo", "Away_Elo", "Home_xG_Rolling5", "Away_xGA_Rolling5", "Home_PPG",
    "Home_ShotOnTargetDifference_RollingTeam7", "Away_ShotOnTargetDifference_RollingTeam7",
]
AWAY_COVARIATES = [
    "Away_Elo", "Home_Elo", "Away_xG_Rolling5", "Home_xGA_Rolling5", "Away_PPG",
    "Away_ShotOnTargetDifference_RollingTeam7", "Home_ShotOnTargetDifference_RollingTeam7",
]

# Everything the datasets keep: match identifiers, targets, the covariates
# above, and the bookmaker odds walk_forward_vs_market() compares against.
DATASET_COLUMNS = (
    ["Date", "Time", "HomeTeam", "AwayTeam", "Season", "FTHG", "FTAG", "FTR"]
    + list(dict.fromkeys(HOME_COVARIATES + AWAY_COVARIATES))
    + ["B365HomeOdds", "B365DrawOdds", "B365AwayOdds", "AvgHomeOdds", "AvgDrawOdds", "AvgAwayOdds"]
)


def _bivpois_logpmf(x, y, lam1, lam2, lam3):
    """Log-pmf of the bivariate Poisson (trivariate reduction) -- unchanged
    from Poisson_Covariates_Bivariate.py."""
    min_xy = np.minimum(x, y)
    max_k = int(min_xy.max())

    log_terms = np.full((max_k + 1, len(x)), -np.inf)
    log_ratio = np.log(lam3) - np.log(lam1) - np.log(lam2)
    for i in range(max_k + 1):
        mask = min_xy >= i
        xi, yi = x[mask], y[mask]
        logC_x = gammaln(xi + 1) - gammaln(i + 1) - gammaln(xi - i + 1)
        logC_y = gammaln(yi + 1) - gammaln(i + 1) - gammaln(yi - i + 1)
        ratio_term = i * (log_ratio if np.isscalar(log_ratio) else log_ratio[mask])
        log_terms[i, mask] = logC_x + logC_y + gammaln(i + 1) + ratio_term

    logS = logsumexp(log_terms, axis=0)
    return -(lam1 + lam2 + lam3) + x * np.log(lam1) - gammaln(x + 1) + y * np.log(lam2) - gammaln(y + 1) + logS


class PoissonRegressionGoalsMeanImpute:

    def __init__(self):
        self.k = len(HOME_COVARIATES)
        self.beta_home = None
        self.beta_away = None
        self.home_adv = None
        self.theta = None
        self.home_mean = self.home_std = self.away_mean = self.away_std = None

    def fit(self, matches_df):
        X_home_raw = matches_df[HOME_COVARIATES].astype(float)
        X_away_raw = matches_df[AWAY_COVARIATES].astype(float)

        # Mean/std computed skipping NaNs, THEN missing values filled with
        # that same mean -- an imputed value maps to exactly 0 post-standardization.
        self.home_mean = X_home_raw.mean(axis=0).values
        self.home_std = X_home_raw.std(axis=0).to_numpy(copy=True)
        self.away_mean = X_away_raw.mean(axis=0).values
        self.away_std = X_away_raw.std(axis=0).to_numpy(copy=True)
        self.home_std[self.home_std == 0] = 1
        self.away_std[self.away_std == 0] = 1

        X_home = X_home_raw.fillna(pd.Series(self.home_mean, index=HOME_COVARIATES)).values
        X_away = X_away_raw.fillna(pd.Series(self.away_mean, index=AWAY_COVARIATES)).values

        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        fthg = matches_df["FTHG"].values.astype(float)
        ftag = matches_df["FTAG"].values.astype(float)
        k = self.k

        def neg_log_likelihood(params):
            beta_home = params[:k]
            beta_away = params[k:2 * k]
            home_adv = params[2 * k]
            theta = params[2 * k + 1]

            lam1 = np.exp(home_adv + X_home_std @ beta_home)
            lam2 = np.exp(X_away_std @ beta_away)
            lam3 = np.exp(theta)

            ll = _bivpois_logpmf(fthg, ftag, lam1, lam2, lam3)
            return -ll.sum()

        x0 = np.zeros(2 * k + 2)
        x0[2 * k] = 0.2
        x0[2 * k + 1] = -3.0

        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

        self.beta_home = result.x[:k]
        self.beta_away = result.x[k:2 * k]
        self.home_adv = result.x[2 * k]
        self.theta = result.x[2 * k + 1]
        return self

    def predict_proba(self, matches_df, max_goals=10):
        """Returns an (n_matches, 3) array, columns ordered [H, D, A]."""
        X_home_raw = matches_df[HOME_COVARIATES].astype(float)
        X_away_raw = matches_df[AWAY_COVARIATES].astype(float)
        X_home = X_home_raw.fillna(pd.Series(self.home_mean, index=HOME_COVARIATES)).values
        X_away = X_away_raw.fillna(pd.Series(self.away_mean, index=AWAY_COVARIATES)).values
        X_home_std = (X_home - self.home_mean) / self.home_std
        X_away_std = (X_away - self.away_mean) / self.away_std

        lam1_all = np.exp(self.home_adv + X_home_std @ self.beta_home)
        lam2_all = np.exp(X_away_std @ self.beta_away)
        lam3 = np.exp(self.theta)

        xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
        x_flat, y_flat = xs.ravel().astype(float), ys.ravel().astype(float)

        probs = []
        for lam1, lam2 in zip(lam1_all, lam2_all):
            lam1_arr = np.full_like(x_flat, lam1)
            lam2_arr = np.full_like(x_flat, lam2)
            log_pmf = _bivpois_logpmf(x_flat, y_flat, lam1_arr, lam2_arr, lam3)
            grid = np.exp(log_pmf).reshape(max_goals + 1, max_goals + 1)
            grid = grid / grid.sum()

            p_home = np.tril(grid, -1).sum()
            p_draw = np.trace(grid)
            p_away = np.triu(grid, 1).sum()
            probs.append((p_home, p_draw, p_away))
        return np.array(probs)


def report_comparison(metric_name, model_arr, baseline_arr, baseline_name):
    """Paired t-test + bootstrap CI + win-rate for model vs. one baseline, on one metric."""
    t_stat, p_value = stats.ttest_rel(model_arr, baseline_arr)
    print(f"\nPaired t-test ({metric_name}, model vs {baseline_name}): t={t_stat:.3f}, p={p_value:.6f}")

    diff = model_arr - baseline_arr
    rng = np.random.default_rng(42)
    n = len(diff)
    boot_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI for mean {metric_name} diff (model - {baseline_name}): [{ci_low:.4f}, {ci_high:.4f}]")

    model_wins = (model_arr < baseline_arr).sum()
    baseline_wins = (baseline_arr < model_arr).sum()
    print(
        f"Model better on {model_wins}/{n} matches ({model_wins/n:.2%}), "
        f"{baseline_name} better on {baseline_wins}/{n} matches ({baseline_wins/n:.2%})"
    )


def walk_forward_vs_market(all_df):
    seasons = sorted(all_df["Season"].unique())

    chunk_rows = []
    all_model_ll, all_b365_ll, all_avg_ll = [], [], []
    all_model_brier, all_b365_brier, all_avg_brier = [], [], []
    all_model_correct = []
    per_chunk_idx_ll = {i: [] for i in range(N_CHUNKS + 1)}

    for i in range(WINDOW, len(seasons)):
        train_seasons = seasons[i - WINDOW:i]
        test_season = seasons[i]

        prior_df = all_df[all_df["Season"].isin(train_seasons)]
        season_df = all_df[all_df["Season"] == test_season].sort_values("Date").reset_index(drop=True)
        # Fixed 10-match chunks (= rounds), not array_split, so a live season
        # that's only partly played still gets one chunk per round.
        chunks = [season_df.iloc[r:r + MATCHES_PER_ROUND] for r in range(0, len(season_df), MATCHES_PER_ROUND)]

        for c_idx, test_chunk in enumerate(chunks):
            if test_chunk.empty:
                continue
            elapsed_chunks = pd.concat(chunks[:c_idx]) if c_idx > 0 else season_df.iloc[0:0]
            train_df = pd.concat([prior_df, elapsed_chunks])

            model = PoissonRegressionGoalsMeanImpute().fit(train_df)

            proba = model.predict_proba(test_chunk)
            classes_order = ["H", "D", "A"]

            b365_overround = 1 / test_chunk["B365HomeOdds"] + 1 / test_chunk["B365DrawOdds"] + 1 / test_chunk["B365AwayOdds"]
            fair_b365 = np.column_stack([
                ((1 / test_chunk["B365HomeOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365DrawOdds"]) / b365_overround).values,
                ((1 / test_chunk["B365AwayOdds"]) / b365_overround).values,
            ])

            avg_overround = 1 / test_chunk["AvgHomeOdds"] + 1 / test_chunk["AvgDrawOdds"] + 1 / test_chunk["AvgAwayOdds"]
            fair_avg = np.column_stack([
                ((1 / test_chunk["AvgHomeOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgDrawOdds"]) / avg_overround).values,
                ((1 / test_chunk["AvgAwayOdds"]) / avg_overround).values,
            ])

            y_true = test_chunk["FTR"].values
            y_onehot = label_binarize(y_true, classes=classes_order)

            eps = 1e-15
            model_ll = -np.log(np.clip((proba * y_onehot).sum(axis=1), eps, 1))
            b365_ll = -np.log(np.clip((fair_b365 * y_onehot).sum(axis=1), eps, 1))
            avg_ll = -np.log(np.clip((fair_avg * y_onehot).sum(axis=1), eps, 1))
            model_brier = ((proba - y_onehot) ** 2).sum(axis=1)
            b365_brier = ((fair_b365 - y_onehot) ** 2).sum(axis=1)
            avg_brier = ((fair_avg - y_onehot) ** 2).sum(axis=1)
            model_pred = np.array(classes_order)[proba.argmax(axis=1)]
            model_correct = (model_pred == y_true)

            all_model_ll.append(model_ll)
            all_b365_ll.append(b365_ll)
            all_avg_ll.append(avg_ll)
            all_model_brier.append(model_brier)
            all_b365_brier.append(b365_brier)
            all_avg_brier.append(avg_brier)
            all_model_correct.append(model_correct)
            per_chunk_idx_ll[c_idx].append(model_ll.mean())

            chunk_rows.append({
                "test_season": test_season,
                "chunk": c_idx,
                "n_train": len(train_df),
                "n_test": len(test_chunk),
                "model_ll": model_ll.mean(),
                "bet365_ll": b365_ll.mean(),
                "avg_ll": avg_ll.mean(),
            })

        print(f"{test_season}: done ({len(chunks)} chunks, train grew {len(prior_df)} -> {len(prior_df) + len(season_df)})")

    chunk_df = pd.DataFrame(chunk_rows)

    print(f"\n=== Within-season trend: mean model log loss by chunk index (0=start of season, {N_CHUNKS - 1}=end) ===")
    for c_idx in range(N_CHUNKS):
        vals = per_chunk_idx_ll[c_idx]
        if vals:
            print(f"chunk {c_idx}: mean_log_loss={np.mean(vals):.4f}  (n_seasons={len(vals)})")

    model_ll_all = np.concatenate(all_model_ll)
    b365_ll_all = np.concatenate(all_b365_ll)
    avg_ll_all = np.concatenate(all_avg_ll)
    model_brier_all = np.concatenate(all_model_brier)
    b365_brier_all = np.concatenate(all_b365_brier)
    avg_brier_all = np.concatenate(all_avg_brier)
    model_correct_all = np.concatenate(all_model_correct)

    print(f"\n=== Pooled across all {len(chunk_df)} chunks ({len(model_ll_all)} test matches) ===")
    print(f"Model log loss:      {model_ll_all.mean():.4f}")
    print(f"Bet365 log loss:     {b365_ll_all.mean():.4f}")
    print(f"Avg-bookie log loss: {avg_ll_all.mean():.4f}")
    print(f"Model Brier:         {model_brier_all.mean():.4f}")
    print(f"Bet365 Brier:        {b365_brier_all.mean():.4f}")
    print(f"Avg-bookie Brier:    {avg_brier_all.mean():.4f}")
    print(f"Model accuracy (1X2): {model_correct_all.mean():.4f}")

    print(
        "\n(Compare against Poisson_Covariates_Bivariate.py's reported result: "
        "0.9678 pooled log loss / 0.5746 Brier / 54.26% accuracy, "
        "and Poisson_Bivariate_Copula.py's: 0.9726 / 0.5780 / 53.74%.)"
    )

    print("\n--- Log loss ---")
    report_comparison("log loss", model_ll_all, b365_ll_all, "Bet365")
    report_comparison("log loss", model_ll_all, avg_ll_all, "avg-bookie")

    print("\n--- Brier score ---")
    report_comparison("Brier", model_brier_all, b365_brier_all, "Bet365")
    report_comparison("Brier", model_brier_all, avg_brier_all, "avg-bookie")


def main():
    all_df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    if os.path.exists(LIVE_SEASON_PATH):
        live_df = pd.read_csv(LIVE_SEASON_PATH, parse_dates=["Date"])
        all_df = pd.concat([all_df, live_df[[c for c in all_df.columns if c in live_df.columns]]], ignore_index=True)
    if all_df["Season"].nunique() <= WINDOW:
        raise SystemExit(f"Need more than WINDOW={WINDOW} seasons to have a test season -- "
                         f"run src/live/predict_round.py first to create {LIVE_SEASON_PATH}.")
    all_df = all_df.sort_values("Date").reset_index(drop=True)
    walk_forward_vs_market(all_df)


if __name__ == "__main__":
    main()
