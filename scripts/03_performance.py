"""
Predictive performance (Sections 2.5, 2.6 and 2.9; Table 1, Supplementary S1-S3, Table 5).

Per outer fold and model: MAE, wMAE, R2 and MSE on the validation subjects, on the held-out
fold and, to show under- or overfitting, on the training subjects. SBSA is scored on mu, Ridge
on its single fit, and each MLP variant as the mean of its members' scores; the MAE of the
population's mean prediction (SBSA candidates, MLP ensemble) is saved next to it.

SBSA is compared with every other model over the five folds, on the validation subjects and on
the held-out folds, with the corrected resampled t-test (Nadeau and Bengio, 2003). Its
correction is n_heldout / n_train of the folds, the overlap between their training sets; on
the validation subjects, which are the same for every fold, the test therefore reflects the
variability over training sets only. The uncorrected paired t-test is saved too, and so are
Benjamini-Hochberg adjusted p-values across feature sets and models.

Writes results/tables/performance_folds.csv, performance_summary.csv, performance_tests.csv
and training_cost.csv.
"""

import pandas as pd
from scipy import stats

from ebga_analyses.config import load_config
from ebga_analyses.results import iter_folds, load_fold, models_in, save_table
from ebga_analyses.results import scored_prediction
from ebga_analyses.stats import benjamini_hochberg, corrected_ttest, metrics

METRICS = ["mae", "wmae", "r2", "mse", "mae_population_mean"]
EVALUATION_SETS = {"eval": "validation", "heldout": "heldout_fold", "train": "training"}


def fold_scores(cfg):
    """One row per fold, model and evaluation set; and one row of fit costs per fold and model."""
    scores, costs = [], []
    for arch, feature_set, setting, fold in iter_folds(cfg):
        arrays, info = load_fold(cfg, arch, feature_set, setting, fold)
        key = {"arch": arch, "features": feature_set, "setting": setting, "fold": fold,
               "n_train": len(arrays["train_ids"]), "n_heldout": len(arrays["heldout_ids"])}
        for model in models_in(arrays):
            for which, subjects in EVALUATION_SETS.items():
                y = arrays[f"y_{which}"]
                member_scores = [metrics(p, y) for p in scored_prediction(arrays, model, which)]
                mean_scores = pd.DataFrame(member_scores).mean().to_dict()
                population_mean = arrays[f"{model}_pred_{which}"].mean(axis=0)
                scores.append({**key, "model": model, "subjects": subjects, **mean_scores,
                               "mae_population_mean": metrics(population_mean, y)["mae"]})
            if model in info:  # Ridge's costs are in the non-linear run only
                m = info[model]
                costs.append({**key, "model": model, "fit_seconds": m["fit_seconds"],
                              "total_seconds": m["total_seconds"], **m["best_params"]})
    return pd.DataFrame(scores), pd.DataFrame(costs)


def paired_tests(folds):
    """Every model against SBSA over the outer folds, per evaluation set (not the training one)."""
    rows = []
    evaluated = folds[folds["subjects"] != "training"]
    keys = ["arch", "features", "setting", "subjects"]
    for key, group in evaluated.groupby(keys):
        sbsa = group[group["model"] == "SBSA"].sort_values("fold")
        n_train, n_heldout = sbsa["n_train"].mean(), sbsa["n_heldout"].mean()
        for model in group["model"].unique():
            if model == "SBSA":
                continue
            other = group[group["model"] == model].sort_values("fold")
            for metric in ("mae", "wmae", "r2"):
                values, reference = other[metric].to_numpy(), sbsa[metric].to_numpy()
                diff = values - reference
                t, p = corrected_ttest(diff, n_train, n_heldout)
                t_plain, p_plain = stats.ttest_rel(values, reference)
                rows.append({**dict(zip(keys, key, strict=True)), "model": model,
                             "metric": metric, "mean_difference_vs_sbsa": diff.mean(),
                             "sd": diff.std(ddof=1), "t": t, "p": p,
                             "t_uncorrected": t_plain, "p_uncorrected": p_plain})
    tests = pd.DataFrame(rows)
    # False discovery rate across feature sets and models
    family = ["arch", "setting", "subjects", "metric"]
    tests["p_bh"] = tests.groupby(family)["p"].transform(benjamini_hochberg)
    return tests


def main():
    cfg = load_config()
    folds, costs = fold_scores(cfg)
    save_table(cfg, folds, "performance_folds")
    save_table(cfg, costs, "training_cost")

    keys = ["arch", "features", "setting", "subjects", "model"]
    summary = folds.groupby(keys)[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    save_table(cfg, summary.reset_index(), "performance_summary")

    save_table(cfg, paired_tests(folds), "performance_tests")


if __name__ == "__main__":
    main()
