"""
Confounder tests (Section 2.8).

Per outer fold on the validation subjects, for the reference and the no-ComBat models:

    bias of the predicted age   BAG ~ 1 + c + age + age^2; the coefficient of c in years (male vs
                                female, 3T vs 1.5T, per SD for continuous confounders), without
                                weights and weighted by 1/u(s)^2 (models with a population)
    bias of the uncertainty     u(s) ~ 1 + c + age + |error|, all variables standardized

The BAG is the signed error of the scored model (SBSA: f_mu - y; MLP: mean member error).
Effect sizes are saved without correction, together with Benjamini-Hochberg adjusted p-values
of every test across feature sets, models, confounders and folds.

Writes results/tables/confounders_folds.csv and confounders_summary.csv.
"""

import numpy as np
import pandas as pd

from sbsa_analyses.config import load_config
from sbsa_analyses.data import load_subjects
from sbsa_analyses.results import iter_folds, load_fold, models_in, save_table
from sbsa_analyses.results import scored_prediction
from sbsa_analyses.stats import benjamini_hochberg, regression


def zscore(x):
    return (x - x.mean()) / x.std()


def confounder_values(subjects, name):
    """Sex: male = 1; field strength: 3T = 1; continuous confounders standardized."""
    if name == "sex":
        return (subjects["sex"] == "male").to_numpy(dtype=float)
    if name == "magnetic_field_strength":
        return (subjects[name] == 3.0).to_numpy(dtype=float)
    return zscore(subjects[name].to_numpy(dtype=float))


def confounder_tests(bag, u, abs_error, age, c):
    """Shift of the predicted age (unweighted and 1/u^2-weighted) and bias of the uncertainty."""
    design = np.column_stack([c, age, age ** 2])
    b, se, p = regression(bag, design)
    row = {"shift_years": b[0], "shift_se": se[0], "shift_p": p[0]}
    if u is not None:
        b, se, p = regression(bag, design, weights=1 / u ** 2)
        row.update(weighted_shift_years=b[0], weighted_shift_se=se[0], weighted_shift_p=p[0])
        b, se, p = regression(zscore(u), np.column_stack([zscore(c), zscore(age),
                                                          zscore(abs_error)]))
        row.update(uncertainty_beta=b[0], uncertainty_se=se[0], uncertainty_p=p[0])
    return row


def main():
    cfg = load_config()
    subjects = load_subjects(cfg).set_index("participant_id")
    rows = []
    for arch, feature_set, setting, fold in iter_folds(cfg):
        names = list(cfg["confounders"]["all"])
        if feature_set == "thickness":
            names += cfg["confounders"]["thickness_only"]

        arrays, _ = load_fold(cfg, arch, feature_set, setting, fold)
        evaluated = subjects.loc[arrays["eval_ids"]]
        age = arrays["y_eval"]
        for model in models_in(arrays):
            scored = scored_prediction(arrays, model)
            bag = (scored - age).mean(axis=0)
            abs_error = np.abs(scored - age).mean(axis=0)
            u = None if model == "Ridge" else arrays[f"{model}_pred_eval"].std(axis=0)
            for name in names:
                c = confounder_values(evaluated, name)
                rows.append({"arch": arch, "features": feature_set, "setting": setting,
                             "fold": fold, "model": model, "confounder": name,
                             **confounder_tests(bag, u, abs_error, age, c)})

    folds = pd.DataFrame(rows)
    for test in ("shift", "weighted_shift", "uncertainty"):
        folds[f"{test}_p_bh"] = folds.groupby(["arch", "setting"])[f"{test}_p"].transform(
            benjamini_hochberg)
    save_table(cfg, folds, "confounders_folds")

    def n_significant(p):
        """Number of folds with p < 0.05; NaN when the test does not exist (Ridge has no u)."""
        return (p < 0.05).sum() if p.notna().any() else np.nan

    summary = folds.groupby(["arch", "features", "setting", "model", "confounder"]).agg(
        shift_years=("shift_years", "mean"),
        shift_folds_p05=("shift_p", n_significant),
        weighted_shift_years=("weighted_shift_years", "mean"),
        weighted_shift_folds_p05=("weighted_shift_p", n_significant),
        uncertainty_beta=("uncertainty_beta", "mean"),
        uncertainty_folds_p05=("uncertainty_p", n_significant),
        shift_folds_bh05=("shift_p_bh", n_significant),
        weighted_shift_folds_bh05=("weighted_shift_p_bh", n_significant),
        uncertainty_folds_bh05=("uncertainty_p_bh", n_significant))
    save_table(cfg, summary.reset_index(), "confounders_summary")


if __name__ == "__main__":
    main()
