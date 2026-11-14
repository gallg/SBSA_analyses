"""
Run the whole analysis, or some of its steps.

    python run_all.py                    every step, in order
    python run_all.py --steps train      one or more steps: split, train, performance,
                                         subjects, confounders, features, replicability, figures
    python run_all.py --smoke            the same code with tiny budgets, into results_smoke/

Training runs one nested cross-validation per (architecture, feature set, setting): 13 per
feature set, 39 in total. Folds already saved are skipped, so the command can be run again
after an interruption.

Everything the steps print is shown and written, with the time of each line, to
results/logs/run_<date>_<time>.log. The log ends with a summary: the outcome and duration of
every step, and whether all the training folds are present. A failed training run does not
stop the other training runs; the analysis steps run only when the training is complete. The
command exits with an error if any step failed.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ebga_analyses.config import FEATURE_SETS, load_config, results_path
from ebga_analyses.results import training_dir

ROOT = Path(__file__).resolve().parent
STEPS = {
    "split": "01_make_split.py",
    "train": "02_train_models.py",
    "performance": "03_performance.py",
    "subjects": "04_subject_uncertainty.py",
    "confounders": "05_confounders.py",
    "features": "06_feature_importance.py",
    "replicability": "07_replicability.py",
    "figures": "08_figures.py",
}
ANALYSIS_STEPS = ("performance", "subjects", "confounders", "features", "replicability", "figures")


def training_runs(n_partitions):
    """(arch, feature set, setting) of every nested cross-validation."""
    partitions = [f"partition_{k}" for k in range(1, n_partitions)]
    for feature_set in FEATURE_SETS:
        for setting in ["reference", *partitions, "full_cohort", "no_combat"]:
            yield "nonlinear", feature_set, setting
        for setting in ["reference", *partitions, "full_cohort"]:
            yield "linear", feature_set, setting


def missing_folds(cfg):
    """Training folds that are not on disk yet."""
    missing = []
    for arch, feature_set, setting in training_runs(cfg["cv"]["n_partitions"]):
        folder = training_dir(cfg, arch, feature_set, setting)
        missing += [f"{arch}/{feature_set}/{setting}/fold_{f}" for f in range(cfg["cv"]["n_outer"])
                    if not (folder / f"fold_{f}.npz").exists()]
    return missing


class Log:
    """Prints every line and writes it, with the time, to the log file."""

    def __init__(self, path):
        self.file = open(path, "w", buffering=1)  # noqa: SIM115 (kept open for the whole run)
        self.outcomes = []  # (step, outcome, seconds)

    def write(self, line):
        print(line, flush=True)
        self.file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")

    def run(self, name, script, *args):
        """Run one script, passing its output to the log; record its outcome and duration."""
        command = [sys.executable, "-u", str(ROOT / "scripts" / script), *args]
        self.write(f"--- {name}: started")
        start = time.time()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True)
        for line in process.stdout:
            self.write("    " + line.rstrip())
        ok = process.wait() == 0
        seconds = time.time() - start
        self.outcomes.append((name, "OK" if ok else "FAILED", seconds))
        self.write(f"--- {name}: {'finished' if ok else 'FAILED'} after {duration(seconds)}")
        return ok

    def skip(self, name, reason):
        self.outcomes.append((name, "SKIPPED", 0.0))
        self.write(f"--- {name}: skipped ({reason})")

    def summary(self, missing):
        self.write("")
        self.write("=== Summary")
        for name, outcome, seconds in self.outcomes:
            self.write(f"{outcome:8s} {duration(seconds):>10s}  {name}")
        if missing:
            self.write(f"Training folds missing: {len(missing)}, e.g. {', '.join(missing[:3])}")
        else:
            self.write("All training folds are present.")
        failed = [name for name, outcome, _ in self.outcomes if outcome != "OK"]
        self.write("ALL STEPS FINISHED SUCCESSFULLY." if not failed
                   else f"NOT FINISHED: {len(failed)} step(s) failed or were skipped.")
        return not failed


def duration(seconds):
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h {rest // 60:02d}m {rest % 60:02d}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        os.environ["EBGA_SMOKE"] = "1"  # read by load_config() in every script
    cfg = load_config()
    log = Log(results_path(cfg, "logs", f"run_{datetime.now():%Y%m%d_%H%M%S}.log"))
    log.write(f"Steps: {', '.join(s for s in STEPS if s in args.steps)}"
              f"{'  [smoke test]' if args.smoke else ''}")
    log.write(f"Results folder: {cfg['paths']['results_dir']}")

    for step in (s for s in STEPS if s in args.steps):
        if step == "train":
            for arch, feature_set, setting in training_runs(cfg["cv"]["n_partitions"]):
                log.run(f"train {arch} / {feature_set} / {setting}", STEPS[step],
                        "--arch", arch, "--features", feature_set, "--setting", setting)
        elif step in ANALYSIS_STEPS and missing_folds(cfg):
            log.skip(step, "the training is not complete")
        else:
            log.run(step, STEPS[step])

    sys.exit(0 if log.summary(missing_folds(cfg)) else 1)


if __name__ == "__main__":
    main()
