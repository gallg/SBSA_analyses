"""
Permutation importance and the importance maps of Section 2.3 (eq. 7-8).

The per-subject uncertainty u(s) (eq. 9) is the SD over members of their predictions,
`predictions.std(axis=0)`, with denominator P.
"""

import numpy as np


def shuffles(n_subjects, n_repeats, seed):
    """Row permutations shared by every model of a fold, so all models see the same shuffles."""
    rng = np.random.RandomState(seed)
    return np.array([rng.permutation(n_subjects) for _ in range(n_repeats)])


def permutation_importance(predict_many, X, y, perms):
    """
    Increase in MAE when one feature is shuffled, for every member of a population.

    Parameters
    ----------
    predict_many : callable
        predict_many(X) returns the predictions of all members, shape (n_members, n_subjects).
    X : ndarray of shape (n_subjects, n_features)
    y : ndarray of shape (n_subjects,)
    perms : ndarray of shape (n_repeats, n_subjects)
        Row permutations, see shuffles().

    Returns
    -------
    ndarray of shape (n_members, n_repeats, n_features)
    """
    baseline = np.abs(predict_many(X) - y).mean(axis=1)
    importance = np.empty((len(baseline), len(perms), X.shape[1]))
    for j in range(X.shape[1]):
        X_shuffled = X.copy()
        for r, perm in enumerate(perms):
            X_shuffled[:, j] = X[perm, j]
            importance[:, r, j] = np.abs(predict_many(X_shuffled) - y).mean(axis=1) - baseline
    return importance


def population_map(importance):
    """
    Importance I_j (eq. 7) and uncertainty U_j (eq. 8) of every feature.

    The repeats of each member are averaged first; I_j and U_j are then the mean and the SD
    (denominator P) over members. importance has shape (n_members, n_repeats, n_features).
    """
    per_member = importance.mean(axis=1)
    return per_member.mean(axis=0), per_member.std(axis=0)


def parameter_map(mu, sigma, hidden_units, n_features):
    """
    Parameter-space reading of importance and uncertainty (Section 2.3).

    Column norms of the first-layer weights of mu and sigma. EBGA stores the first-layer
    weights first, as W.flatten() with W of shape (hidden_units, n_features).
    """
    n = hidden_units * n_features
    W = mu[:n].reshape(hidden_units, n_features)
    S = sigma[:n].reshape(hidden_units, n_features)
    return np.linalg.norm(W, axis=0), np.linalg.norm(S, axis=0)
