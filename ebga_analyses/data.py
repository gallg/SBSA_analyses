"""
OpenBHB data, the train/validation split and the cross-validation folds (Sections 2.4, 2.6).

The split and every outer and inner fold are stratified by acquisition batch, so that every
batch is present in every training part and ComBat never meets an unseen batch.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import results_path

COVARIATES = ["age", "sex", "magnetic_field_strength", "tiv", "gmv", "wmv", "csfv"]


def load_subjects(cfg):
    """One row per participant: id, acquisition batch and covariates (no features)."""
    participants = pd.read_csv(cfg["paths"]["data_dir"] / "participants.tsv", sep="\t")
    participants = participants.rename(columns={cfg["data"]["batch_column"]: "batch"})
    return participants[["participant_id", "batch", *COVARIATES]].reset_index(drop=True)


def load_features(cfg, feature_set, participant_ids):
    """
    ROI features of one feature set.

    Returns
    -------
    X : ndarray of shape (n_subjects, n_features)
        Rows in the order of participant_ids.
    columns : list of str
        Feature names.
    """
    data_dir = cfg["paths"]["data_dir"]
    features = pd.concat([pd.read_csv(data_dir / "train" / "desikan_roi_features.csv"),
                          pd.read_csv(data_dir / "val" / "desikan_roi_features.csv")],
                         ignore_index=True)

    measures = cfg["data"]["feature_sets"][feature_set]
    columns = [c for m in measures for c in features.columns if c.endswith("_" + m)]
    features = features.set_index("participant_id").loc[participant_ids, columns]
    if features.isna().any().any():
        raise ValueError(f"missing feature values in {feature_set}")
    return features.to_numpy(dtype=float), columns


def make_split(cfg):
    """
    Make the train/validation split and the outer folds of every run, and save them.

    Writes results/split/split.csv with the columns participant_id, batch, split,
    fold_partition_<k> (outer fold of each training subject in partition k) and
    fold_full_cohort (outer folds of the full-cohort in/out-of-sample run).
    """
    # Sorted by id, so that the split does not depend on the row order of participants.tsv
    subjects = load_subjects(cfg).sort_values("participant_id", ignore_index=True)
    batches = subjects["batch"].to_numpy()
    index = np.arange(len(subjects))
    train, val = train_test_split(index, test_size=cfg["data"]["validation_size"],
                                  stratify=batches, random_state=cfg["seeds"]["split"])

    split = subjects[["participant_id", "batch"]].copy()
    split["split"] = "train"
    split.loc[val, "split"] = "validation"

    n_outer, seed = cfg["cv"]["n_outer"], cfg["seeds"]["outer_folds"]
    for k in range(cfg["cv"]["n_partitions"]):
        folds = StratifiedKFold(n_outer, shuffle=True, random_state=seed + k)
        split[f"fold_partition_{k}"] = -1
        for f, (_, test) in enumerate(folds.split(train, batches[train])):
            split.loc[train[test], f"fold_partition_{k}"] = f

    folds = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    split["fold_full_cohort"] = -1
    for f, (_, test) in enumerate(folds.split(index, batches)):
        split.loc[test, "fold_full_cohort"] = f

    split.to_csv(results_path(cfg, "split", "split.csv"), index=False)
    print(f"wrote split/split.csv ({len(split)} subjects)", flush=True)
    return split


def load_split(cfg):
    return pd.read_csv(cfg["paths"]["results_dir"] / "split" / "split.csv")


def inner_folds(cfg, batches, partition, outer_fold):
    """
    Inner 5-fold split of one outer-training part, stratified by batch.

    The seed depends on the partition and on the outer fold, so the inner splits are drawn
    independently of the outer ones.

    Returns
    -------
    list of (train, test) index arrays
    """
    seed = cfg["seeds"]["inner_folds"] + 10 * partition + outer_fold
    folds = StratifiedKFold(cfg["cv"]["n_inner"], shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(batches)), batches))


def check_batches(train_batches, test_batches):
    """Fail early if a batch of the test subjects is missing from the training part."""
    missing = set(np.unique(test_batches)) - set(np.unique(train_batches))
    if missing:
        raise ValueError(f"batches missing from the training part: {sorted(missing)}")
