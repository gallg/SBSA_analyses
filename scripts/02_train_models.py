"""
Nested cross-validation: fit every model once per outer fold and save what the analyses need.

    python scripts/02_train_models.py --arch nonlinear --features volume --setting reference

Settings (Section 2.6):
    reference       partition 0, ComBat; all models (Ridge in the non-linear family only, as it
                    has no hidden layer); source of every headline result
    partition_<k>   partitions 1-4, ComBat; SBSA and MLPs; replicability only
    full_cohort     training and validation subjects together, ComBat; SBSA; in/out of sample
    no_combat       partition 0, batch one-hot encoded as input; SBSA and MLPs; non-linear only

Output, in results/training/<arch>/<features>/<setting>/:
    config.json                     the configuration that produced the folder
    fold_<f>.npz                    subject ids and ages; predictions of every candidate and
                                    member for the training, held-out and evaluation subjects;
                                    permutation importance; SBSA's mu and sigma
    fold_<f>.json                   selected hyperparameters, inner-CV scores of every grid
                                    point, fit times, seeds, package versions
    fold_<f>.joblib                 fitted models
    fold_<f>_preprocessing.joblib   fitted preprocessing

A fold whose .npz file exists is skipped, so an interrupted run can be resumed. Resuming with a
configuration that differs from the folder's config.json is refused.
"""

import argparse
import json
import platform
import time

import joblib
import numpy as np
import sklearn
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from sbsa_analyses.config import MLP_VARIANTS, fold_seeds, load_config
from sbsa_analyses.data import check_batches, inner_folds, load_features, load_split, load_subjects
from sbsa_analyses.models import MLPRegressor, resolve_device
from sbsa_analyses.models import predict_population, sample_population
from sbsa_analyses.populations import permutation_importance, shuffles
from sbsa_analyses.preprocessing import make_preprocessing
from sbsa_analyses.results import training_dir
from sbsa_analyses.tuning import mlp_params, sbsa, tune_mlp, tune_ridge, tune_sbsa


def setting_plan(arch, setting, n_partitions):
    """
    What a setting runs.

    Returns
    -------
    partition : int
        Index used for the inner-CV seeds (the full-cohort run uses n_partitions).
    harmonization : {'combat', 'one_hot'}
    models : list of str
    importance : bool
        Whether permutation importance is computed (feature maps are needed only from the
        reference and partition runs).
    """
    if setting == "reference":
        models = ["SBSA", *MLP_VARIANTS] + (["Ridge"] if arch == "nonlinear" else [])
        return 0, "combat", models, True
    if setting.startswith("partition_"):
        k = int(setting.split("_")[1])
        if not 1 <= k < n_partitions:
            raise ValueError(f"unknown setting: {setting}")
        return k, "combat", ["SBSA", *MLP_VARIANTS], True
    if setting == "full_cohort":
        return n_partitions, "combat", ["SBSA"], False
    if setting == "no_combat":
        if arch != "nonlinear":
            raise ValueError("the harmonization comparison runs for the non-linear models only")
        return 0, "one_hot", ["SBSA", *MLP_VARIANTS], False
    raise ValueError(f"unknown setting: {setting}")


def preprocessing(cfg, harmonization):
    h = cfg["harmonization"]
    return make_preprocessing(harmonization, age_model=h["age_model"], n_knots=h["n_knots"])


def fit_sbsa(cfg, arch, setting, seeds, inner, X_train, y_train, X_eval):
    """Tune and fit SBSA; draw its population of candidates; add the checkpoints (reference)."""
    best, table = tune_sbsa(cfg, arch, inner, seeds["training"])
    start = time.time()
    model = sbsa(cfg, arch, **best, random_state=seeds["training"]).fit(X_train, y_train)
    fit_seconds = time.time() - start

    P, seed = cfg["population"]["size"], seeds["candidates"]
    mu, sigma = model.optimizer_.mu, model.optimizer_.sigma
    thetas = sample_population(mu, sigma, P, seed)
    arrays = {"SBSA_mu": mu, "SBSA_sigma": sigma}

    # Stopping-point check (Section 2.6, point 2). With the same seed, a fit stopped at
    # iteration k is exactly iteration k of the full run, so each checkpoint is a shorter fit.
    if setting == "reference":
        for it in cfg["sbsa"]["checkpoints"]:
            if it == cfg["sbsa"]["iterations"]:
                short = model
            else:
                short = sbsa(cfg, arch, **best, random_state=seeds["training"], iterations=it)
                short.fit(X_train, y_train)
            cp_mu, cp_sigma = short.optimizer_.mu, short.optimizer_.sigma
            arrays[f"SBSA_checkpoint_{it}_mu"] = cp_mu
            arrays[f"SBSA_checkpoint_{it}_sigma"] = cp_sigma
            arrays[f"SBSA_checkpoint_{it}_pred_eval"] = predict_population(
                short, X_eval, sample_population(cp_mu, cp_sigma, P, seed))

    def predict_many(X):
        return predict_population(model, X, thetas)

    return model, predict_many, arrays, best, table, fit_seconds


def fit_mlp(cfg, arch, variant, seeds, inner, X_train, y_train):
    """Tune an MLP variant and train its P members in parallel, each with its own seed."""
    best, table = tune_mlp(cfg, arch, variant, inner, seeds["training"])
    params = {**mlp_params(cfg, arch, variant), **best}
    member_seeds = seeds["members"] + np.arange(cfg["population"]["size"])
    fits = Parallel(n_jobs=cfg["compute"]["n_jobs"])(
        delayed(_fit_member)(params, X_train, y_train, int(seed)) for seed in member_seeds)
    members = [member for member, _ in fits]

    def predict_many(X):
        return np.stack([member.predict(X) for member in members])

    return members, predict_many, {}, best, table, sum(seconds for _, seconds in fits)


def fit_ridge(cfg, inner, X_train, y_train):
    """Tune and fit Ridge (a single fit: no population)."""
    best, table = tune_ridge(cfg, inner)
    start = time.time()
    model = Ridge(alpha=best["alpha"]).fit(X_train, y_train)
    fit_seconds = time.time() - start

    def predict_many(X):
        return model.predict(X)[None, :]

    return model, predict_many, {"Ridge_coef": model.coef_}, best, table, fit_seconds


def _fit_member(params, X, y, seed):
    torch.set_num_threads(1)  # the members already train in parallel
    start = time.time()
    member = MLPRegressor(**params, random_state=seed).fit(X, y)
    return member, time.time() - start


def run_fold(cfg, arch, setting, fold, X_raw, y, batches, ids, train, heldout, evaluation):
    """Inner CV, preprocessing and final fit of every model of one outer fold."""
    partition, harmonization, models, with_importance = setting_plan(
        arch, setting, cfg["cv"]["n_partitions"])
    seeds = fold_seeds(cfg, partition, fold)
    check_batches(batches[train], batches[np.concatenate([heldout, evaluation])])

    # Preprocessing (ComBat, scaling) is fitted once on the outer-training subjects and applied
    # to the held-out fold and the validation subjects, which never enter any fit.
    prep = preprocessing(cfg, harmonization).fit(X_raw[train], y[train])
    X_train, X_heldout, X_eval = (prep.transform(X_raw[i]) for i in (train, heldout, evaluation))

    # Inner CV on the harmonized outer-training subjects: ComBat is not refitted on each inner
    # split, whose smaller batches would give noisier batch estimates (and less stable tuning).
    y_train = y[train]
    inner = [(X_train[tr], y_train[tr], X_train[te], y_train[te])
             for tr, te in inner_folds(cfg, batches[train], partition, fold)]

    arrays = {
        "train_ids": ids[train], "heldout_ids": ids[heldout], "eval_ids": ids[evaluation],
        "y_train": y_train, "y_heldout": y[heldout], "y_eval": y[evaluation],
        "mean_train_age": y_train.mean(),
    }
    if with_importance:
        perms = shuffles(len(evaluation), cfg["importance"]["n_repeats"], seeds["shuffles"])
        arrays["shuffles"] = perms
    info = {"seeds": seeds, "mlp_device": str(resolve_device(cfg["compute"]["device"])),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scikit-learn": sklearn.__version__, "torch": torch.__version__}}
    fitted = {}
    for name in models:
        start = time.time()
        if name == "SBSA":
            fit = fit_sbsa(cfg, arch, setting, seeds, inner, X_train, y_train, X_eval)
        elif name == "Ridge":
            fit = fit_ridge(cfg, inner, X_train, y_train)
        else:
            fit = fit_mlp(cfg, arch, name, seeds, inner, X_train, y_train)
        model, predict_many, extra, best, table, fit_seconds = fit

        # Predictions of every member; those of the training subjects are kept only to
        # compare the training error with the held-out and validation errors.
        arrays.update(extra)
        for which, X_part in (("train", X_train), ("heldout", X_heldout), ("eval", X_eval)):
            arrays[f"{name}_pred_{which}"] = predict_many(X_part)
            if name == "SBSA":  # SBSA is scored on its mean mu
                arrays[f"SBSA_pred_mu_{which}"] = model.predict(X_part)
        if with_importance:
            arrays[f"{name}_importance"] = permutation_importance(
                predict_many, X_eval, y[evaluation], perms).astype(np.float32)

        info[name] = {"best_params": best, "inner_cv": table, "fit_seconds": fit_seconds,
                      "total_seconds": time.time() - start}
        fitted[name] = model
        print(f"    {name}: {best}  ({time.time() - start:.0f} s)", flush=True)
    return arrays, info, prep, fitted


def check_config(cfg, path):
    """
    Save the configuration next to the results it produces, or, when the folder already has
    results, refuse to add folds obtained with a different configuration. Paths and compute
    settings are left out: they do not change the results.
    """
    current = {k: v for k, v in cfg.items() if k not in ("paths", "compute")}
    current = json.loads(json.dumps(current))  # same types as a configuration read back
    if path.exists():
        with open(path) as f:
            if json.load(f) != current:
                raise ValueError(f"{path.parent} holds results of a different configuration; "
                                 "delete the folder to run it again")
    else:
        with open(path, "w") as f:
            json.dump(current, f, indent=1)


def main(arch, feature_set, setting):
    cfg = load_config()
    split = load_split(cfg)
    ids = split["participant_id"].to_numpy()
    batches = split["batch"].to_numpy()
    y = load_subjects(cfg).set_index("participant_id").loc[ids, "age"].to_numpy(dtype=float)
    X, columns = load_features(cfg, feature_set, ids)
    X_raw = np.column_stack([X, batches])  # the batch is the last column, read by ComBat

    partition, *_ = setting_plan(arch, setting, cfg["cv"]["n_partitions"])
    if setting == "full_cohort":
        folds = split["fold_full_cohort"].to_numpy()
        pool = np.arange(len(split))
        evaluation = pool  # every subject is predicted by every fold model, seen or unseen
    else:
        folds = split[f"fold_partition_{partition}"].to_numpy()
        pool = np.flatnonzero(split["split"] == "train")
        evaluation = np.flatnonzero(split["split"] == "validation")

    out_dir = training_dir(cfg, arch, feature_set, setting)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_config(cfg, out_dir / "config.json")
    for fold in range(cfg["cv"]["n_outer"]):
        stem = out_dir / f"fold_{fold}"
        if stem.with_suffix(".npz").exists():
            print(f"[{arch} / {feature_set} / {setting}] outer fold {fold}: already saved, skipped")
            continue

        print(f"[{arch} / {feature_set} / {setting}] outer fold {fold}", flush=True)
        train, heldout = pool[folds[pool] != fold], pool[folds[pool] == fold]
        arrays, info, prep, fitted = run_fold(cfg, arch, setting, fold, X_raw, y, batches, ids,
                                              train, heldout, evaluation)
        arrays["feature_names"] = np.array(columns)

        joblib.dump(prep, out_dir / f"fold_{fold}_preprocessing.joblib")
        joblib.dump(fitted, stem.with_suffix(".joblib"))
        with open(stem.with_suffix(".json"), "w") as f:
            json.dump(info, f, indent=1)
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)  # written last: fold complete
        print(f"    fold {fold} saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", required=True, choices=["nonlinear", "linear"])
    parser.add_argument("--features", required=True, choices=["volume", "thickness", "pooled"])
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    main(args.arch, args.features, args.setting)
