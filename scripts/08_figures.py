"""
Figure 1: the 2D map of the brain regions (Section 2.7), SBSA reference run.

Each region is placed at its importance I_j and importance-adjusted uncertainty, averaged over
the five outer folds, with the median splits that define the four classes. The script reads
only results/tables/feature_maps.csv, so the figure can be redrawn without rerunning anything.

Writes results/figures/feature_map_<arch>_<features>.png.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ebga_analyses.config import load_config, results_path  # noqa: E402


def main():
    cfg = load_config()
    maps = pd.read_csv(cfg["paths"]["results_dir"] / "tables" / "feature_maps.csv")
    sbsa = maps[(maps["model"] == "SBSA") & (maps["setting"] == "reference")]
    for (arch, feature_set), d in sbsa.groupby(["arch", "features"]):
        regions = d.groupby("region")[["importance", "uncertainty_adjusted"]].mean()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(regions["importance"], regions["uncertainty_adjusted"], s=14)
        ax.axvline(regions["importance"].median(), color="grey", lw=0.8)
        ax.axhline(regions["uncertainty_adjusted"].median(), color="grey", lw=0.8)
        ax.set_xlabel("Importance $I_j$ (increase in MAE, years)")
        ax.set_ylabel("Uncertainty adjusted for importance (rank residual)")
        ax.set_title(f"SBSA, {arch}, {feature_set}")
        fig.tight_layout()
        name = f"feature_map_{arch}_{feature_set}.png"
        fig.savefig(results_path(cfg, "figures", name), dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}", flush=True)


if __name__ == "__main__":
    main()
