"""
Configuration: reads config.toml and resolves every path relative to the repository root.
"""

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FEATURE_SETS = ("volume", "thickness", "pooled")
ARCHITECTURES = ("nonlinear", "linear")
MLP_VARIANTS = ("MLP", "MLP+L2", "MLP+M+TR", "MLP+L2+M+TR")


def load_config(smoke=None):
    """
    Return the configuration as a dictionary.

    Parameters
    ----------
    smoke : bool, optional
        Use the reduced budgets of the [smoke] section and write to its results folder.
        If None, smoke mode is read from the EBGA_SMOKE environment variable, which
        `run_all.py --smoke` sets for every script it launches.
    """
    with open(ROOT / "config.toml", "rb") as f:
        cfg = tomllib.load(f)

    if smoke is None:
        smoke = os.environ.get("EBGA_SMOKE") == "1"
    if smoke:
        _apply_smoke_settings(cfg)
    cfg["smoke"] = smoke

    cfg["paths"]["data_dir"] = ROOT / cfg["paths"]["data_dir"]
    cfg["paths"]["results_dir"] = ROOT / cfg["paths"]["results_dir"]
    return cfg


def _apply_smoke_settings(cfg):
    """Replace the budgets with the [smoke] values (same code path, tiny runs)."""
    s = cfg["smoke"]
    cfg["paths"]["results_dir"] = s["results_dir"]
    cfg["cv"]["n_partitions"] = s["n_partitions"]
    cfg["population"]["size"] = s["population_size"]
    cfg["sbsa"]["iterations"] = s["sbsa_iterations"]
    cfg["sbsa"]["checkpoints"] = s["sbsa_checkpoints"]
    cfg["mlp"]["grid"]["epochs"] = s["mlp_epochs"]
    cfg["mlp"]["grid"]["lr"] = s["mlp_lr"]
    cfg["mlp"]["grid"]["alpha"] = s["alpha"]
    cfg["ridge"]["alpha"] = s["alpha"]
    cfg["importance"]["n_repeats"] = s["importance_repeats"]


def fold_seeds(cfg, partition, fold):
    """
    Seeds of one outer fold of one run (Section 2.6, point 2).

    They are the same in every rerun and different in every fold and partition, so that the
    agreement between folds and between partitions is never helped by a shared initialization
    or by shared random draws.
    """
    s = cfg["seeds"]
    offset = s["partition_stride"] * partition + s["fold_stride"] * fold
    return {name: s[name] + offset for name in ("training", "members", "candidates", "shuffles")}


def results_path(cfg, *parts):
    """Path inside the results folder; its parent folders are created."""
    path = cfg["paths"]["results_dir"].joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
