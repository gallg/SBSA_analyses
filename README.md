# EBGA analyses

Code for the analyses of the manuscript on the SBSA optimizer and its importance and
uncertainty estimates for brain-age prediction on OpenBHB. Section numbers below refer to the
manuscript's Methods.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .        # sbsa_analyses, to run single scripts (run_all.py does not need it)
```

The MLPs are trained on the device set in `config.toml` (`[compute] device`, default `"cuda"`);
without a GPU they fall back to the CPU. The device used is recorded in each fold's `.json`.
SBSA and Ridge always run on the CPU. The networks are small (at most 136 inputs and 32 hidden
units, full batch), so a GPU is not needed: with `device = "cpu"` the `n_jobs` parallel fits
each use one core.

`requirements.txt` pins every package, including the EBGA package (SBSA optimizer) at the
commit used for the paper.

## Data

The analyses use the OpenBHB dataset, hosted on
[Hugging Face](https://huggingface.co/datasets/benoit-dufumier/openBHB). The whole dataset is
about 372 GB (it includes the images); the analyses need only three small tables (11 MB in
total): the participants' information and the FreeSurfer Desikan-Killiany regional features of
the dataset's `train` and `val` folders. Download them from the repository root with:

```bash
BASE=https://huggingface.co/datasets/benoit-dufumier/openBHB/resolve/main
mkdir -p data/OpenBHB/train data/OpenBHB/val
curl -L -o data/OpenBHB/participants.tsv $BASE/participants.tsv
curl -L -o data/OpenBHB/train/desikan_roi_features.csv \
     $BASE/train/derivatives/freesurfer_roi/desikan_roi_features.csv
curl -L -o data/OpenBHB/val/desikan_roi_features.csv \
     $BASE/val/derivatives/freesurfer_roi/desikan_roi_features.csv
```

The `data/` folder must then look like this (in the dataset, the two feature tables are in
`<train|val>/derivatives/freesurfer_roi/`; here they sit directly in `train/` and `val/`):

```
data/
  OpenBHB/
    participants.tsv                   3984 participants: age, sex, site, acquisition setting
                                       (siteXacq), field strength, TIV, GMV, WMV, CSFV
    train/desikan_roi_features.csv     3227 participants x 7 measures x 68 regions
    val/desikan_roi_features.csv        757 participants x 7 measures x 68 regions
```

The dataset's own train/val folders are only read and merged: the split used in the paper is
made by `scripts/01_make_split.py`. `data/` is not tracked by git. OpenBHB is distributed under
its own terms (see the dataset page: CC BY-NC-SA, and the data usage agreements of the
cohorts it pools), which apply to these files.

## Running

```bash
python run_all.py --smoke     # check that everything runs (minutes; writes results_smoke/)
python run_all.py             # the full analysis
python run_all.py --steps performance subjects   # only some steps
```

Everything the steps print is also written, with the time of each line, to
`results/logs/run_<date>_<time>.log`. The log ends with a summary of the outcome and duration
of every step and says whether all training folds are present; the last line reads
`ALL STEPS FINISHED SUCCESSFULLY.` when the run is complete. A failed training run does not stop
the others, and the analysis steps run only once the training is complete.

Training can be interrupted and restarted: folds already saved are skipped. All settings
(paths, seeds, folds, grids, budgets, population size) are in `config.toml`; each results
folder keeps a copy of the configuration that produced it, and the training script refuses to
add folds to a folder made with a different configuration.

## What runs

**Models (Section 2.5).** SBSA (EBGA's `EBGARegressor`, fixed iterations, no early stopping)
and four MLPs trained by full-batch gradient descent (plain, +L2, +momentum and trust region,
both), all minimizing the MAE; and scikit-learn's Ridge regression (squared error with an L2
penalty). Every model is tuned and scored by the MAE. Non-linear networks: one hidden layer of
32 ReLU units.
Linear-restricted family (Section 2.6.3): the same networks with one linear hidden unit.

**Nested cross-validations (Section 2.6)**, each 5 outer x 5 inner folds, for every feature
set (volume, thickness, pooled):

| Setting | Models | Purpose |
|---|---|---|
| `reference` | all | partition 0; every headline result |
| `partition_1` ... `partition_4` | SBSA, MLPs | replicability (other fold partitions) |
| `full_cohort` | SBSA | in-sample vs out-of-sample uncertainty (train + validation subjects) |
| `no_combat` | SBSA, MLPs (non-linear only) | harmonization: batch as one-hot input instead of ComBat |

Non-linear: 7 settings, linear-restricted: 6, so 39 nested cross-validations in total.

**Inside one outer fold** (`scripts/02_train_models.py`):

1. Preprocessing fitted once on the outer-training subjects and applied to the held-out fold
   and the validation subjects: ComBat, then z-scoring (or z-scoring plus one-hot batch for
   `no_combat`). ComBat models age as a cubic spline (5 knots at training-age quantiles, as in
   ComBat-GAM), so that non-linear age trends are not removed from batches covering a narrow
   age range.
2. Inner 5-fold CV on the harmonized outer-training subjects: grid search of the
   hyperparameters. ComBat is not refitted on each inner split, whose smaller batches would
   give noisier batch estimates and a less stable hyperparameter search.
3. Final fit with the selected hyperparameters: SBSA one run, then P = 16 candidates from its
   final N(mu, sigma^2); each MLP variant 16 members, each with its own seed; Ridge once.
4. Saved: predictions of every candidate/member for the training subjects, the held-out fold
   and the validation subjects; permutation importance of every member (validation subjects);
   SBSA's mu and sigma (also at checkpoints); selected hyperparameters, inner-CV scores of every
   grid point, fit times, seeds and package versions; the fitted preprocessing and models.

**Analyses** (read the saved files only, no training):

| Script | Manuscript | Output (`results/tables/`) |
|---|---|---|
| `03_performance.py` | 2.5, 2.6, 2.9 | `performance_*.csv`, `training_cost.csv` |
| `04_subject_uncertainty.py` | 2.7 individual predictions | `subject_values.csv`, `subject_uncertainty_*.csv` |
| `05_confounders.py` | 2.8 | `confounders_*.csv` |
| `06_feature_importance.py` | 2.3, 2.7 brain regions | `feature_maps.csv`, `importance_divergence.csv`, `uncertainty_correlates.csv` |
| `07_replicability.py` | 2.6 point 2 | `replicability.csv`, `in_out_of_sample.csv` (full-cohort run for SBSA; reference run for SBSA and the MLP ensembles) |
| `08_figures.py` | 2.7 brain regions (2D map) | `results/figures/` |

## Results layout

```
results/
  split/split.csv                                   split and outer folds of every partition
  training/<nonlinear|linear>/<volume|thickness|pooled>/<setting>/
      config.json                                   configuration that produced the folder
      fold_<k>.{npz,json,joblib}                    arrays, information and fitted models
      fold_<k>_preprocessing.joblib                 fitted ComBat and scaler
  tables/                                           analysis outputs (csv)
  figures/
  logs/                                             one log per `run_all.py` call
```

Everything numerical is kept (all predictions and importance arrays), so new statistics can be
computed from `results/training/` without retraining. Figures are cheap to redraw and are kept
to the minimum.

## How leakage is prevented

- The validation subjects are never used to select anything; hyperparameters come from the
  inner CV only (including the MLPs' number of epochs, read from checkpoints of one run).
- ComBat and the scaler are fitted on the outer-training subjects of each fold and only
  applied to the held-out fold and the validation subjects, which never enter any fit. The
  inner CV tunes on these harmonized training subjects, so its test parts did contribute to the
  ComBat fit; this can affect only the choice of hyperparameters, never the reported scores.
- Age is never a model input. It is the training target, ComBat's fit covariate (training
  subjects only), and a regressor in the analyses after prediction. ComBat's transform removes
  only each batch's location shift, so it never uses the age of the subjects it transforms
  (`sbsa_analyses/preprocessing.py`).
- The split and every outer and inner fold are stratified by acquisition batch, so every batch
  is in every training part; `01_make_split.py` checks it and the training script fails if a
  batch is ever missing.
- Each model is fitted once; every analysis reads the same saved predictions.
- Uncertainty comes only from the population of a single fit (candidates or members);
  replicability only from different folds, partitions or the in/out comparison.

## Conventions

- u(s) and U_j are SDs over the P members with denominator P (eq. 8-9).
- Performance of an MLP variant = mean of its members' scores (not the ensemble mean).
- Mean +/- SD over folds use ddof = 1.
- Model comparisons over the five outer folds with the corrected resampled t-test (Nadeau and
  Bengio, 2003), whose correction is the ratio between the held-out and the training part of
  the folds; the uncorrected paired t-test is saved too. Spearman for associations; all tests
  two-sided. Benjamini-Hochberg adjusted p-values are saved next to the uncorrected ones
  for the model comparisons and the confounder tests.
- Reproducibility: every random choice (split, folds, training, candidate draws, MLP members,
  shuffles) has a fixed seed in `config.toml` (`[seeds]`), so rerunning any step gives identical
  results on the same machine and device; on a GPU the MLPs use deterministic algorithms. Every
  outer fold of every partition has its own seeds, so that no two fitted models share their
  initialization or their random draws.
- `neurocombat-sklearn` 0.1.3 is used through a small adapter (`sbsa_analyses/preprocessing.py`),
  which also maps the library's `OneHotEncoder(sparse=...)` call to current scikit-learn.

## License

GNU General Public License v3 (see `LICENSE`), as the EBGA package.
