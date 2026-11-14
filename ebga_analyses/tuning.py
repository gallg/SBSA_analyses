"""
Hyperparameter selection in the inner cross-validation (Section 2.5).

Every tuning function receives `inner`, a list with one (X_train, y_train, X_test, y_test)
tuple per inner split, taken from the outer-training subjects after the preprocessing
fitted on them. The selected hyperparameters minimize the mean MAE over the inner test
parts. The score of every grid point is returned too, so that choices at the edge of a grid
can be checked after the run.
"""

import itertools

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from .models import MLPRegressor, make_sbsa


def sbsa(cfg, arch, tau, eta, random_state, iterations=None):
    """SBSA model with the configured settings; `iterations` defaults to the full budget."""
    s = cfg["sbsa"]
    return make_sbsa(**cfg["architecture"][arch], tau=tau, eta=eta,
                     iterations=iterations or s["iterations"],
                     population=cfg["population"]["size"], momentum=s["momentum"],
                     trust_radius=s["trust_radius"], sigma_min=s["sigma_min"],
                     sigma_max=s["sigma_max"], random_state=random_state)


def mlp_params(cfg, arch, variant):
    """Fixed settings of an MLP variant (Section 2.5); lr, alpha and epochs are tuned."""
    stabilized = "M+TR" in variant
    return {**cfg["architecture"][arch],
            "device": cfg["compute"]["device"],
            "momentum": cfg["mlp"]["momentum"] if stabilized else 0.0,
            "trust_radius": cfg["mlp"]["trust_radius"] if stabilized else 0.0}


def tune_sbsa(cfg, arch, inner, seed):
    """Grid search over the temperature tau and the sigma learning rate eta."""
    grid = cfg["sbsa"]["grid"]
    jobs = [(tau, eta, s) for tau, eta in itertools.product(grid["tau"], grid["eta"])
            for s in range(len(inner))]
    maes = Parallel(n_jobs=cfg["compute"]["n_jobs"])(
        delayed(_sbsa_mae)(cfg, arch, tau, eta, *inner[s], seed) for tau, eta, s in jobs)

    scores = {}
    for (tau, eta, _), mae in zip(jobs, maes, strict=True):
        scores.setdefault((("tau", tau), ("eta", eta)), []).append(mae)
    return _best(scores)


def tune_mlp(cfg, arch, variant, inner, seed):
    """Grid search over lr (and alpha for the L2 variants); the epochs come from one
    checkpointed run per grid point."""
    grid = cfg["mlp"]["grid"]
    alphas = grid["alpha"] if "L2" in variant else [0.0]
    jobs = [(lr, alpha, s) for lr, alpha in itertools.product(grid["lr"], alphas)
            for s in range(len(inner))]
    fixed = mlp_params(cfg, arch, variant)
    maes = Parallel(n_jobs=cfg["compute"]["n_jobs"])(
        delayed(_mlp_maes)({**fixed, "lr": lr, "alpha": alpha}, *inner[s], grid["epochs"], seed)
        for lr, alpha, s in jobs)

    scores = {}
    for (lr, alpha, _), mae_per_epochs in zip(jobs, maes, strict=True):
        for epochs, mae in mae_per_epochs.items():
            scores.setdefault((("lr", lr), ("alpha", alpha), ("epochs", epochs)), []).append(mae)
    return _best(scores)


def tune_ridge(cfg, inner):
    """Grid search over the L2 strength alpha of Ridge (scored, like every model, by the MAE)."""
    scores = {}
    for alpha in cfg["ridge"]["alpha"]:
        maes = []
        for X_train, y_train, X_test, y_test in inner:
            model = Ridge(alpha=alpha).fit(X_train, y_train)
            maes.append(float(np.mean(np.abs(model.predict(X_test) - y_test))))
        scores[(("alpha", alpha),)] = maes
    return _best(scores)


def _sbsa_mae(cfg, arch, tau, eta, X_train, y_train, X_test, y_test, seed):
    model = sbsa(cfg, arch, tau=tau, eta=eta, random_state=seed).fit(X_train, y_train)
    return float(np.mean(np.abs(model.predict(X_test) - y_test)))


def _mlp_maes(params, X_train, y_train, X_test, y_test, epochs, seed):
    torch.set_num_threads(1)  # one thread per worker: the grid points already run in parallel
    model = MLPRegressor(**params, random_state=seed)
    predictions = model.fit(X_train, y_train, checkpoints=epochs, X_eval=X_test)
    return {e: float(np.mean(np.abs(p - y_test))) for e, p in predictions.items()}


def _best(scores):
    """scores maps a grid point (tuple of (name, value)) to its MAE per inner split.
    Returns the best grid point as a dictionary, and the table of all grid points.
    A grid point whose training diverged (non-finite MAE) can never be selected."""
    table = [{**dict(point), "mae": float(np.mean(maes))} for point, maes in scores.items()]
    finite = [row for row in table if np.isfinite(row["mae"])]
    if not finite:
        raise ValueError("no grid point gave a finite inner-CV error")
    best = min(finite, key=lambda row: row["mae"])
    return {k: v for k, v in best.items() if k != "mae"}, table
