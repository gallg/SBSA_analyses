"""
Per-subject uncertainty u(s) (Section 2.7, "Individual predictions").

For every model with a population (SBSA: P candidates of one run; each MLP variant: its P
members), per outer fold on the validation subjects:

    calibration     Spearman(u, |error|), should be positive
    direction       Spearman(u, signed error), should be about zero
    age extremes    Spearman(u, |age - mean training age|)
    large gaps      Spearman(u, |age-corrected gap|); mean |gap| in the top and bottom u
                    quartiles; share of the 10% largest |gap| in the top u quartile (chance 25%)
    agreement       Spearman of SBSA's u with the spread of each MLP ensemble

Each test is repeated on u adjusted for |prediction - mean training age| (rank residual).

Writes results/tables/subject_values.csv (one row per subject, model and fold),
subject_uncertainty_folds.csv and subject_uncertainty_summary.csv.
"""

import numpy as np
import pandas as pd

from sbsa_analyses.config import load_config
from sbsa_analyses.data import load_subjects
from sbsa_analyses.results import iter_folds, load_fold, save_table, subject_table
from sbsa_analyses.stats import age_corrected_gap, rank_residual, spearman

KEYS = ["arch", "features", "setting", "fold"]


def uncertainty_tests(u, d):
    """The tests of Section 2.7 for one uncertainty vector u on the subjects of table d."""
    gap = np.abs(d["corrected_gap"].to_numpy())
    top, bottom = u >= np.quantile(u, 0.75), u <= np.quantile(u, 0.25)
    largest = gap >= np.quantile(gap, 0.90)
    return {
        "calibration": spearman(u, d["abs_error"])[0],
        "direction": spearman(u, d["signed_error"])[0],
        "age_extremes": spearman(u, np.abs(d["age"] - d["mean_train_age"]))[0],
        "gap_rho": spearman(u, gap)[0],
        "gap_mean_top_quartile": gap[top].mean(),
        "gap_mean_bottom_quartile": gap[bottom].mean(),
        "largest_gaps_in_top_quartile": (top & largest).sum() / largest.sum(),
    }


def subject_values(cfg):
    """Per-subject table of every fold, with the age-corrected gap and the adjusted u."""
    subjects = load_subjects(cfg)
    tables = []
    for arch, feature_set, setting, fold in iter_folds(cfg):
        arrays, _ = load_fold(cfg, arch, feature_set, setting, fold)
        for _, d in subject_table(arrays, subjects).groupby("model", sort=False):
            d = d.assign(arch=arch, features=feature_set, setting=setting, fold=fold)
            d["corrected_gap"] = age_corrected_gap(d["signed_error"].to_numpy(),
                                                   d["age"].to_numpy())
            d["u_adjusted"] = rank_residual(d["u"], np.abs(d["prediction"] - d["mean_train_age"]))
            tables.append(d)
    return pd.concat(tables, ignore_index=True)


def main():
    cfg = load_config()
    values = subject_values(cfg)
    save_table(cfg, values, "subject_values")

    rows = []
    for key, d in values.groupby(KEYS + ["model"], sort=False):
        for kind in ("u", "u_adjusted"):
            rows.append({**dict(zip(KEYS + ["model"], key, strict=True)), "uncertainty": kind,
                         "median_u_years": d["u"].median(),
                         **uncertainty_tests(d[kind].to_numpy(), d)})

    # Agreement between SBSA's u(s) and the spread of each MLP ensemble, on the same subjects
    for key, group in values.groupby(KEYS):
        sbsa = group[group["model"] == "SBSA"].set_index("participant_id")
        for model, d in group[group["model"] != "SBSA"].groupby("model"):
            d = d.set_index("participant_id").loc[sbsa.index]
            for kind in ("u", "u_adjusted"):
                rows.append({**dict(zip(KEYS, key, strict=True)), "model": model,
                             "uncertainty": kind,
                             "agreement_with_sbsa": spearman(sbsa[kind], d[kind])[0]})

    folds = pd.DataFrame(rows)
    save_table(cfg, folds, "subject_uncertainty_folds")
    summary = (folds.drop(columns="fold")
               .groupby(["arch", "features", "setting", "model", "uncertainty"])
               .agg(["mean", "std"]))
    summary.columns = ["_".join(c) for c in summary.columns]
    save_table(cfg, summary.reset_index(), "subject_uncertainty_summary")


if __name__ == "__main__":
    main()
