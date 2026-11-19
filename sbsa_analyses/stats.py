"""
Metrics and statistics used by the analysis scripts (Sections 2.6-2.9).
"""

import itertools

import numpy as np
from scipy import stats


def metrics(prediction, y):
    """MAE, weighted MAE (MAE / age range of the evaluation set [52]), R2 and MSE."""
    error = prediction - y
    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    return {"mae": mae, "wmae": mae / (y.max() - y.min()), "r2": 1 - mse / np.var(y), "mse": mse}


def spearman(a, b):
    """Spearman correlation and its two-sided p-value."""
    rho, p = stats.spearmanr(a, b)
    return float(rho), float(p)


def corrected_ttest(differences, n_train, n_heldout):
    """
    Corrected resampled t-test (Nadeau and Bengio, 2003) on the paired differences of k folds.

    The folds' training sets overlap, so the differences are not independent: the variance of
    their mean is 1/k + n_heldout / n_train times their variance, instead of 1/k, where
    n_heldout and n_train are the sizes of the two parts of each resampling. Returns t and its
    two-sided p-value (k - 1 degrees of freedom).
    """
    d = np.asarray(differences, dtype=float)
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + n_heldout / n_train) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def benjamini_hochberg(p):
    """Benjamini-Hochberg adjusted p-values (false discovery rate); NaN entries are kept."""
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = ~np.isnan(p)
    m = valid.sum()
    if m:
        order = np.argsort(p[valid])
        ranked = p[valid][order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(ranked, 1.0)
        q[valid] = adjusted
    return q


def rank_residual(a, b):
    """Residual of rank(a) after a linear regression on rank(b): a adjusted for b (Section 2.7)."""
    rank_a, rank_b = stats.rankdata(a), stats.rankdata(b)
    slope, intercept = np.polyfit(rank_b, rank_a, 1)
    return rank_a - (slope * rank_b + intercept)


def regression(y, X, weights=None):
    """
    Least squares with an intercept, optionally weighted.

    Returns the coefficients, standard errors and two-sided p-values of the columns of X
    (the intercept is not returned).
    """
    X = np.column_stack([np.ones(len(y)), X])
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    residuals = (y - X @ coef) * sw
    dof = len(y) - X.shape[1]
    covariance = np.linalg.inv((X * w[:, None]).T @ X) * (residuals @ residuals) / dof
    se = np.sqrt(np.diag(covariance))
    p = 2 * stats.t.sf(np.abs(coef / se), dof)
    return coef[1:], se[1:], p[1:]


def age_corrected_gap(signed_error, age):
    """Residual of BAG ~ 1 + age + age^2 (Section 2.7, [53, 54])."""
    X = np.column_stack([np.ones(len(age)), age, age ** 2])
    coef = np.linalg.lstsq(X, signed_error, rcond=None)[0]
    return signed_error - X @ coef


def pairwise_spearman(runs):
    """Mean Spearman correlation over all pairs of columns of `runs` (one column per run)."""
    pairs = itertools.combinations(range(runs.shape[1]), 2)
    return float(np.mean([spearman(runs[:, i], runs[:, j])[0] for i, j in pairs]))


def split_half_reliability(maps, n_splits=100, seed=0):
    """
    Reliability of the average of all runs.

    Spearman correlation between the averages of two random halves of the runs, stepped up
    to the full number of runs with the Spearman-Brown formula. maps: (n_features, n_runs).
    """
    rng = np.random.RandomState(seed)
    n = maps.shape[1]
    half = n // 2
    rhos = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        rhos.append(spearman(maps[:, order[:half]].mean(1), maps[:, order[half:]].mean(1))[0])
    r, k = np.mean(rhos), n / half
    return float(k * r / (1 + (k - 1) * r))


def classes(importance, adjusted_uncertainty):
    """The four classes of the 2D map, from median splits of both axes (Section 2.7)."""
    important = importance > np.median(importance)
    uncertain = adjusted_uncertainty > np.median(adjusted_uncertainty)
    return np.where(important,
                    np.where(uncertain, "important-uncertain", "important-certain"),
                    np.where(uncertain, "unimportant-uncertain", "unimportant-certain"))
