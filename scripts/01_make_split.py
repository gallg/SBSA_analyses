"""
Train/validation split and outer folds, all stratified by acquisition batch (Sections 2.4, 2.6).

Writes results/split/split.csv and results/tables/sample_description.csv (the numbers of
Section 2.4 for each part of the split, and the association of field strength with age used in
Section 3.6). Fails if a batch is missing from any training part of the outer or inner CV.
"""

import numpy as np
import pandas as pd
from scipy import stats

from sbsa_analyses.config import load_config
from sbsa_analyses.data import inner_folds, load_subjects, make_split
from sbsa_analyses.results import save_table


def check_balance(cfg, split):
    """Smallest number of subjects of one batch in any outer and inner training part."""
    batches = split["batch"].to_numpy()
    n_batches = len(np.unique(batches))
    train = np.flatnonzero(split["split"] == "train")
    runs = [(k, f"fold_partition_{k}", train) for k in range(cfg["cv"]["n_partitions"])]
    runs.append((cfg["cv"]["n_partitions"], "fold_full_cohort", np.arange(len(split))))

    smallest = {"outer": np.inf, "inner": np.inf}
    for partition, column, pool in runs:
        folds = split[column].to_numpy()
        for f in range(cfg["cv"]["n_outer"]):
            outer_batches = batches[pool[folds[pool] != f]]
            if len(np.unique(outer_batches)) != n_batches:
                raise ValueError(f"{column}, fold {f}: a batch is missing from the training part")
            smallest["outer"] = min(smallest["outer"],
                                    pd.Series(outer_batches).value_counts().min())
            for inner_train, _ in inner_folds(cfg, outer_batches, partition, f):
                inner_batches = outer_batches[inner_train]
                if len(np.unique(inner_batches)) != n_batches:
                    raise ValueError(f"{column}, fold {f}: a batch is missing from an inner part")
                smallest["inner"] = min(smallest["inner"],
                                        pd.Series(inner_batches).value_counts().min())
    return smallest


def describe(cfg, split):
    """Sample description of every part of the split and of the reference outer folds."""
    subjects = load_subjects(cfg).set_index("participant_id").loc[split["participant_id"]]
    subjects["split"] = split["split"].to_numpy()
    subjects["fold"] = split["fold_partition_0"].to_numpy()

    parts = [("all", subjects), *subjects.groupby("split")]
    parts += [(f"reference_heldout_fold_{f}", subjects[subjects["fold"] == f])
              for f in range(cfg["cv"]["n_outer"])]
    rows = []
    for name, d in parts:
        field = d["magnetic_field_strength"]
        rows.append({
            "subjects": name,
            "n": len(d),
            "male": int((d["sex"] == "male").sum()),
            "female": int((d["sex"] == "female").sum()),
            "age_mean": d["age"].mean(),
            "age_sd": d["age"].std(),
            "age_min": d["age"].min(),
            "age_max": d["age"].max(),
            "n_3T": int((field == 3.0).sum()),
            "n_1_5T": int((field == 1.5).sum()),
            "batches": d["batch"].nunique(),
            "spearman_field_strength_age": stats.spearmanr(field, d["age"])[0],
        })
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    split = make_split(cfg)
    smallest = check_balance(cfg, split)
    print(f"Every batch is in every training part. Smallest number of subjects of one batch: "
          f"{smallest['outer']} (outer), {smallest['inner']} (inner).")

    table = describe(cfg, split)
    save_table(cfg, table, "sample_description")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
