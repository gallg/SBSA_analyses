"""
Importance and uncertainty of the brain regions (Sections 2.3 and 2.7, "Brain regions").

Per outer fold, on the validation subjects:

    maps    I_j (eq. 7) and U_j (eq. 8) over the P members of SBSA and of each MLP ensemble;
            U_j adjusted for I_j (rank residual) and the four classes; the same map read from
            SBSA's parameters (first-layer column norms of mu and sigma). Saved for the
            reference run and for the other partitions (column `setting`), so that the
            stability of every region can be examined later.

Reference run only:

    (i)     disagreement between every pair of models: % of regions above the median importance
            for one model and below it for the other
    (iii)   Spearman correlation, across regions, of SBSA's adjusted U_j with the SD of the
            region's importance rank across models (model type), its R^2 on all other regions
            (collinearity), its |correlation| with TIV and with sex (confounding), all on the
            fold's preprocessed training subjects, and with the spread of the MLP members

Writes results/tables/feature_maps.csv, importance_divergence.csv and uncertainty_correlates.csv.
"""

import itertools

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sbsa_analyses.config import ARCHITECTURES, FEATURE_SETS, load_config
from sbsa_analyses.data import load_features, load_split, load_subjects
from sbsa_analyses.populations import parameter_map, population_map
from sbsa_analyses.results import load_fold, models_in, save_table, training_dir
from sbsa_analyses.stats import classes, rank_residual, spearman


def region_properties(cfg, arch, feature_set, fold, arrays, X_raw, ids, subjects):
    """Collinearity and |correlation| with TIV and sex of every region, on the fold's
    training subjects after the fold's own preprocessing."""
    folder = training_dir(cfg, arch, feature_set, "reference")
    prep = joblib.load(folder / f"fold_{fold}_preprocessing.joblib")
    X = prep.transform(X_raw[pd.Index(ids).get_indexer(arrays["train_ids"])])

    collinearity = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        others = np.column_stack([np.ones(len(X)), np.delete(X, j, axis=1)])
        residual = X[:, j] - others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        collinearity[j] = 1 - residual.var() / X[:, j].var()

    train_subjects = subjects.loc[arrays["train_ids"]]
    male = (train_subjects["sex"] == "male").to_numpy(dtype=float)
    tiv = train_subjects["tiv"].to_numpy(dtype=float)
    return {
        "collinearity": collinearity,
        "tiv": np.abs([stats.pearsonr(x, tiv)[0] for x in X.T]),
        "sex": np.abs([stats.pearsonr(x, male)[0] for x in X.T]),
    }


def fold_maps(arrays, hidden_units):
    """Rows of the feature-map table for every model of one fold."""
    regions = arrays["feature_names"]
    maps = []
    for model in models_in(arrays):
        importance, uncertainty = population_map(arrays[f"{model}_importance"])
        entries = [(model, importance, uncertainty)]
        if model == "SBSA":
            entries.append(("SBSA (parameters)", *parameter_map(
                arrays["SBSA_mu"], arrays["SBSA_sigma"], hidden_units, len(regions))))
        for name, I, U in entries:
            single_fit = name == "Ridge"  # no population: no uncertainty
            adjusted = np.full(len(I), np.nan) if single_fit else rank_residual(U, I)
            maps.append(pd.DataFrame({
                "model": name, "region": regions, "importance": I, "uncertainty": U,
                "uncertainty_adjusted": adjusted,
                "region_class": "" if single_fit else classes(I, adjusted),
            }))
    return pd.concat(maps, ignore_index=True)


def divergence(importance):
    """(i): % of regions above the median for one model and below it for the other."""
    rows = []
    for a, b in itertools.combinations(importance, 2):
        above_a = importance[a] > np.median(importance[a])
        above_b = importance[b] > np.median(importance[b])
        rows.append({"model_a": a, "model_b": b,
                     "percent_switching": 100 * np.mean(above_a != above_b)})
    return rows


def main():
    cfg = load_config()
    split = load_split(cfg)
    subjects = load_subjects(cfg).set_index("participant_id")
    ids = split["participant_id"].to_numpy()

    partitions = [f"partition_{k}" for k in range(1, cfg["cv"]["n_partitions"])]
    maps, divergences, correlates = [], [], []
    for arch in ARCHITECTURES:
        hidden_units = cfg["architecture"][arch]["hidden_units"]
        for feature_set in FEATURE_SETS:
            X, _ = load_features(cfg, feature_set, ids)
            X_raw = np.column_stack([X, split["batch"].to_numpy()])
            for setting, fold in itertools.product(partitions, range(cfg["cv"]["n_outer"])):
                arrays, _ = load_fold(cfg, arch, feature_set, setting, fold)
                maps.append(fold_maps(arrays, hidden_units).assign(
                    arch=arch, features=feature_set, setting=setting, fold=fold))

            for fold in range(cfg["cv"]["n_outer"]):
                key = {"arch": arch, "features": feature_set, "fold": fold}
                arrays, _ = load_fold(cfg, arch, feature_set, "reference", fold)
                maps.append(fold_maps(arrays, hidden_units).assign(setting="reference", **key))

                importance = {m: population_map(arrays[f"{m}_importance"])[0]
                              for m in models_in(arrays)}
                divergences += [{**key, **row} for row in divergence(importance)]

                # (iii) what SBSA's adjusted uncertainty correlates with
                I_sbsa, U_sbsa = population_map(arrays["SBSA_importance"])
                adjusted_sbsa = rank_residual(U_sbsa, I_sbsa)
                ranks = np.array([stats.rankdata(v) for v in importance.values()])
                properties = region_properties(cfg, arch, feature_set, fold, arrays, X_raw,
                                               ids, subjects)
                properties["model_type"] = ranks.std(axis=0)
                for model in importance:
                    if model.startswith("MLP"):
                        I, U = population_map(arrays[f"{model}_importance"])
                        properties[f"{model} spread (adjusted)"] = rank_residual(U, I)
                        properties[f"{model} spread (raw)"] = U
                for name, values in properties.items():
                    rho_adjusted = spearman(adjusted_sbsa, values)[0]
                    rho_raw = spearman(U_sbsa, values)[0]
                    correlates.append({**key, "correlate": name,
                                       "spearman_with_sbsa_adjusted_u": rho_adjusted,
                                       "spearman_with_sbsa_raw_u": rho_raw})

    save_table(cfg, pd.concat(maps), "feature_maps")
    save_table(cfg, pd.DataFrame(divergences), "importance_divergence")
    save_table(cfg, pd.DataFrame(correlates), "uncertainty_correlates")


if __name__ == "__main__":
    main()
