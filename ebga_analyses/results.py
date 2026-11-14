"""
Access to the outputs of scripts/02_train_models.py.
"""

import json

import numpy as np
import pandas as pd

from .config import ARCHITECTURES, FEATURE_SETS, MLP_VARIANTS, results_path

# Settings read by the analyses of performance, per-subject uncertainty and confounders:
# the reference run of both families, and the no-ComBat run of the non-linear family.
ANALYSIS_SETTINGS = {"nonlinear": ("reference", "no_combat"), "linear": ("reference",)}


def training_dir(cfg, arch, feature_set, setting):
    return cfg["paths"]["results_dir"] / "training" / arch / feature_set / setting


def iter_folds(cfg, settings=ANALYSIS_SETTINGS):
    """Yield (arch, feature_set, setting, fold) for every saved fold of the given settings."""
    for arch in ARCHITECTURES:
        for feature_set in FEATURE_SETS:
            for setting in settings[arch]:
                for fold in range(cfg["cv"]["n_outer"]):
                    yield arch, feature_set, setting, fold


def load_fold(cfg, arch, feature_set, setting, fold):
    """
    Arrays (dict) and information (dict from the json file) of one outer fold.

    Linear-family reference folds also receive the arrays of Ridge, which is fitted once in
    the non-linear reference run (it has no hidden layer).
    """
    folder = training_dir(cfg, arch, feature_set, setting)
    with np.load(folder / f"fold_{fold}.npz") as z:
        arrays = dict(z)
    with open(folder / f"fold_{fold}.json") as f:
        info = json.load(f)

    if arch == "linear" and setting == "reference":
        baseline = training_dir(cfg, "nonlinear", feature_set, "reference")
        with np.load(baseline / f"fold_{fold}.npz") as z:
            if not np.array_equal(z["eval_ids"], arrays["eval_ids"]):
                raise ValueError("Ridge was fitted on different subjects")
            arrays.update({k: z[k] for k in z.files if k.startswith("Ridge_")})
    return arrays, info


def models_in(arrays):
    """Names of the models present in a fold, in the order used in the paper."""
    order = ["SBSA", *MLP_VARIANTS, "Ridge"]
    return [m for m in order if f"{m}_pred_eval" in arrays]


def population_models(arrays):
    """Models with a population of P members: SBSA (candidates) and the MLPs (members)."""
    return [m for m in models_in(arrays) if m != "Ridge"]


def scored_prediction(arrays, model, which="eval"):
    """
    Predictions whose errors are reported, shape (n_scored, n_subjects).

    SBSA is scored on its mean mu and Ridge on its single fit. For the MLPs every
    member is returned, because each member is scored and the scores are then averaged.
    """
    if model == "SBSA":
        return arrays[f"SBSA_pred_mu_{which}"][None, :]
    return arrays[f"{model}_pred_{which}"]


def subject_table(arrays, subjects):
    """
    Per-subject quantities of every population model on the evaluation subjects.

    Columns: model, participant_id, age, u, signed_error, abs_error, prediction,
    mean_train_age, and the covariates of `subjects`. The signed error is the brain-age gap of
    the scored model (SBSA: f_mu - y; MLP: mean of the members' errors).
    """
    y = arrays["y_eval"]
    tables = []
    for model in population_models(arrays):
        scored = scored_prediction(arrays, model)
        tables.append(pd.DataFrame({
            "model": model,
            "participant_id": arrays["eval_ids"],
            "age": y,
            "u": arrays[f"{model}_pred_eval"].std(axis=0),
            "signed_error": (scored - y).mean(axis=0),
            "abs_error": np.abs(scored - y).mean(axis=0),
            "prediction": scored.mean(axis=0),
            "mean_train_age": float(arrays["mean_train_age"]),
        }))
    table = pd.concat(tables, ignore_index=True)
    return table.merge(subjects.drop(columns="age"), on="participant_id", how="left")


def save_table(cfg, table, name):
    """Write a table to results/tables/<name>.csv and report it (the line ends up in the log)."""
    path = results_path(cfg, "tables", f"{name}.csv")
    table.to_csv(path, index=False)
    print(f"wrote tables/{name}.csv ({len(table)} rows)", flush=True)
