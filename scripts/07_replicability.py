"""
Replicability of the importance and uncertainty estimates (Section 2.6, point 2; Table 4).

    per-subject u(s)    cross-fold: mean pairwise Spearman between the five reference fold
                        models; cross-partition: mean pairwise Spearman between the
                        fold-averaged u(s) of the five partitions; SBSA and every MLP ensemble
    feature map         the same two levels for I_j and the adjusted U_j, the % of regions in
                        the same class (chance 25%) and the split-half reliability of the map
                        averaged over all fold models; prediction and parameter space
    parameter space     per reference fold: Spearman of the parameter-space map with the
                        prediction-space map and % of regions in the same class
    stopping point      per reference fold: Spearman of SBSA's u(s) at each checkpoint with the
                        final u(s)
    in/out of sample    each subject's out-of-sample u(s) (the fold model that did not train
                        on it) against its in-sample u(s) (mean of the four that did): Spearman
                        and median paired difference (out - in, years), overall and per fold.
                        From the full-cohort run (SBSA, all subjects) and, without any extra
                        fit, from the reference run (SBSA and every MLP ensemble, training
                        subjects)

Writes results/tables/replicability.csv and in_out_of_sample.csv.
"""

import itertools

import numpy as np
import pandas as pd

from ebga_analyses.config import ARCHITECTURES, FEATURE_SETS, load_config
from ebga_analyses.populations import parameter_map, population_map
from ebga_analyses.results import load_fold, population_models, save_table
from ebga_analyses.stats import classes, pairwise_spearman, rank_residual, spearman
from ebga_analyses.stats import split_half_reliability


def same_class(maps):
    """Mean % of regions assigned to the same class by two maps; maps: list of (I, U_adjusted)."""
    labels = [classes(I, U) for I, U in maps]
    pairs = itertools.combinations(labels, 2)
    return float(np.mean([100 * np.mean(a == b) for a, b in pairs]))


def map_replicability(maps_per_partition):
    """
    Replicability rows of one model's feature map.

    maps_per_partition maps a partition index to the list of (I, U_adjusted) of its folds;
    partition 0 is the reference run.
    """
    reference = maps_per_partition[0]
    averaged = [tuple(np.mean(axis, axis=0) for axis in zip(*maps, strict=True))
                for maps in maps_per_partition.values()]
    every_fold = [m for maps in maps_per_partition.values() for m in maps]

    rows = []
    for i, quantity in enumerate(("importance", "uncertainty_adjusted")):
        rows.append({
            "quantity": quantity,
            "cross_fold": pairwise_spearman(np.column_stack([m[i] for m in reference])),
            "cross_partition": pairwise_spearman(np.column_stack([m[i] for m in averaged])),
            "reliability_of_average": split_half_reliability(
                np.column_stack([m[i] for m in every_fold])),
            "n_maps_averaged": len(every_fold),
        })
    rows.append({"quantity": "same_class_percent", "cross_fold": same_class(reference),
                 "cross_partition": same_class(averaged)})
    return rows


def in_out_of_sample(cfg, arch, feature_set, setting):
    """
    Out-of-sample vs in-sample u(s) of every model with a population in one run.

    Every subject of the run's folds is held out by one fold model and trained on by the
    others. The full-cohort run covers all subjects (SBSA only); the reference run covers the
    training subjects, for SBSA and the MLP ensembles.
    """
    folds = [load_fold(cfg, arch, feature_set, setting, f)[0]
             for f in range(cfg["cv"]["n_outer"])]
    ids = pd.Index(np.concatenate([arrays["heldout_ids"] for arrays in folds]))
    held_out_by = np.concatenate([np.full(len(a["heldout_ids"]), f) for f, a in enumerate(folds)])

    rows = []
    for model in population_models(folds[0]):
        out_sample = np.concatenate([a[f"{model}_pred_heldout"].std(axis=0) for a in folds])
        in_sum, in_count = np.zeros(len(ids)), np.zeros(len(ids))
        for arrays in folds:
            position = ids.get_indexer(arrays["train_ids"])
            in_sum[position] += arrays[f"{model}_pred_train"].std(axis=0)
            in_count[position] += 1
        in_sample = in_sum / in_count
        difference = out_sample - in_sample

        key = {"arch": arch, "features": feature_set, "setting": setting, "model": model}
        rows.append({**key, "fold": "all", "n_subjects": len(ids),
                     "spearman": spearman(out_sample, in_sample)[0],
                     "median_difference_out_minus_in": np.median(difference),
                     "median_u_out": np.median(out_sample), "median_u_in": np.median(in_sample)})
        for f in range(len(folds)):
            held = held_out_by == f
            rows.append({**key, "fold": f, "n_subjects": int(held.sum()),
                         "median_difference_out_minus_in": np.median(difference[held])})
    return rows


def same_fold_checks(cfg, arrays, maps, arch, feature_set, fold):
    """Parameter- vs prediction-space map and checkpoint vs final u(s), in one reference fold."""
    key = {"arch": arch, "features": feature_set, "fold": fold}
    (I_p, U_p), (I_q, U_q) = maps["SBSA (parameters)"][0][-1], maps["SBSA"][0][-1]
    checks = [("importance", spearman(I_p, I_q)[0]),
              ("uncertainty_adjusted", spearman(U_p, U_q)[0]),
              ("same_class_percent", same_class([(I_p, U_p), (I_q, U_q)]))]
    rows = [{**key, "model": "SBSA (parameters)", "quantity": f"{name}_vs_prediction_space",
             "same_fold": value} for name, value in checks]

    final = arrays["SBSA_pred_eval"].std(axis=0)
    for it in cfg["sbsa"]["checkpoints"]:
        checkpoint = arrays[f"SBSA_checkpoint_{it}_pred_eval"].std(axis=0)
        rows.append({**key, "model": "SBSA", "quantity": f"u_checkpoint_{it}_vs_final",
                     "same_fold": spearman(checkpoint, final)[0]})
    return rows


def main():
    cfg = load_config()
    settings = ["reference"] + [f"partition_{k}" for k in range(1, cfg["cv"]["n_partitions"])]
    rows, inout = [], []
    for arch in ARCHITECTURES:
        hidden_units = cfg["architecture"][arch]["hidden_units"]
        for feature_set in FEATURE_SETS:
            u = {}     # (model, partition) -> list of u(s) per fold
            maps = {}  # model -> {partition: list of (I, U_adjusted) per fold}
            for partition, setting in enumerate(settings):
                for fold in range(cfg["cv"]["n_outer"]):
                    arrays, _ = load_fold(cfg, arch, feature_set, setting, fold)
                    for model in population_models(arrays):
                        u.setdefault((model, partition), []).append(
                            arrays[f"{model}_pred_eval"].std(axis=0))
                        I, U = population_map(arrays[f"{model}_importance"])
                        maps.setdefault(model, {}).setdefault(partition, []).append(
                            (I, rank_residual(U, I)))
                    I, U = parameter_map(arrays["SBSA_mu"], arrays["SBSA_sigma"], hidden_units,
                                         len(arrays["feature_names"]))
                    maps.setdefault("SBSA (parameters)", {}).setdefault(partition, []).append(
                        (I, rank_residual(U, I)))
                    if setting == "reference":
                        rows += same_fold_checks(cfg, arrays, maps, arch, feature_set, fold)

            key = {"arch": arch, "features": feature_set}
            for model in sorted({m for m, _ in u}):
                per_partition = {k: np.column_stack(v) for (m, k), v in u.items() if m == model}
                fold_means = np.column_stack([v.mean(axis=1) for v in per_partition.values()])
                rows.append({**key, "model": model, "quantity": "u_subject",
                             "cross_fold": pairwise_spearman(per_partition[0]),
                             "cross_partition": pairwise_spearman(fold_means)})
            for model, per_partition in maps.items():
                rows += [{**key, "model": model, **row}
                         for row in map_replicability(per_partition)]
            for setting in ("full_cohort", "reference"):
                inout += in_out_of_sample(cfg, arch, feature_set, setting)

    save_table(cfg, pd.DataFrame(rows), "replicability")
    save_table(cfg, pd.DataFrame(inout), "in_out_of_sample")


if __name__ == "__main__":
    main()
