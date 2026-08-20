"""Tune ``PerJointTreeModel``/``PerJointForestModel`` hyperparameters with
file-grouped CV.

Standalone diagnostic script, not part of the main training CLI (see
``train_distillation_model.py``'s ``--model`` flag for actually training and
saving a model). Run this first to pick ``HGB_KWARGS``/``RF_KWARGS``, then
hardcode the winners as those classes' defaults in ``train_distillation_model.py``
and retrain normally.

Why file-grouped CV and not the row-level split train_distillation_model.py
otherwise uses: this project's real generalization target is unseen
*recordings* (``data/heldout/*.csv``), but a random row-level split pools
rows that are highly autocorrelated within a recording (the robot streams at
~128 Hz, so adjacent rows are ~8ms apart and near-duplicates once lag
features are involved) -- an easy, optimistic validation signal.
``HistGradientBoostingRegressor``'s own default early stopping validates
this way internally, which is one hypothesis for why the untuned tree lost
to the linear+gravity model on the honest file-level held-out set despite
winning the row-level one (see ``ModelReview.md`` §6b). This script's CV is
grouped by recording (``GroupKFold`` on the source-file index ``_design``
attaches to every row, see its docstring) so the score being optimized here
matches the metric that actually matters.

    python tune_tree_hyperparams.py --model tree_per_joint
    python tune_tree_hyperparams.py --model forest_per_joint --n-iter 40
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from preprocess import default_preprocess
from train_distillation_model import DEFAULT_CSVS, RESULTS_DIR, PerJointTreeModel, _RANDOM_STATE
from utils import JOINT_NAMES, N_JOINTS

# (estimator class, kwargs fixed during the search, distribution to sample
# from). Forest's own n_jobs is left at the default (1) here -- the search
# itself already parallelizes across CV folds/candidates with n_jobs=-1;
# also setting it on the estimator would oversubscribe cores. Set n_jobs=-1
# on RandomForestRegressor for the *production* RF_KWARGS instead, once
# tuning is done.
SEARCH_SPACES = {
    "tree_per_joint": (
        HistGradientBoostingRegressor,
        {"early_stopping": False},
        {
            "learning_rate":     [0.03, 0.05, 0.1, 0.2, 0.3],
            "max_iter":          [50, 100, 150, 200, 300],
            "max_leaf_nodes":    [7, 15, 31, 63],
            "min_samples_leaf":  [10, 20, 50, 100, 200],
            "l2_regularization": [0.0, 0.01, 0.1, 1.0],
        },
    ),
    "forest_per_joint": (
        RandomForestRegressor,
        # n_estimators fixed rather than searched: more trees is ~monotonically
        # as-good-or-better (pure variance reduction from averaging), so it
        # doesn't need CV to pick -- searching it just multiplies runtime for
        # no signal. max_depth=None (unbounded) dropped from the search space
        # for the same reason it's expensive here: on ~370k rows it grows
        # very deep, very slow trees for a marginal-at-best gain over a
        # bounded depth.
        {"n_estimators": 200},
        {
            "max_depth":        [6, 10, 16, 24],
            "min_samples_leaf": [1, 5, 20, 50],
            "max_features":     ["sqrt", 0.5, 1.0],
        },
    ),
}


def tune(model_name: str, n_iter: int, n_splits: int, csvs: list) -> dict:
    """Per-joint ``RandomizedSearchCV`` (file-grouped) for ``model_name``.

    Returns ``{joint_name: {"best_params": ..., "best_r2": ...}}``. Uses
    ``PerJointTreeModel()._design`` to build the design matrices -- shared
    feature set with ``PerJointForestModel`` (both inherit ``FEATURE_NAMES``
    from ``PerJointPositionModel`` unchanged), so either model's search reads
    the exact same ``X``/``y``/``groups`` a real training run would build.
    """
    from analysis import Recording

    pre = default_preprocess()
    recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                 for r in (Recording(p) for p in csvs)]
    design = PerJointTreeModel()._design(recordings)

    estimator_cls, fixed_kwargs, param_dist = SEARCH_SPACES[model_name]
    n_groups = len(np.unique(next(iter(design.values()))[2]))
    splits = min(n_splits, n_groups)

    best = {}
    for j in range(N_JOINTS):
        X, y, groups = design[j]
        search = RandomizedSearchCV(
            estimator_cls(random_state=_RANDOM_STATE, **fixed_kwargs),
            param_distributions=param_dist, n_iter=n_iter,
            cv=GroupKFold(n_splits=splits), scoring="r2",
            random_state=_RANDOM_STATE, n_jobs=-1)
        search.fit(X, y, groups=groups)
        best[JOINT_NAMES[j]] = {"best_params": search.best_params_,
                                "best_r2": float(search.best_score_)}
        print(f"[{model_name}] {JOINT_NAMES[j]:10s} best R2={search.best_score_:.4f}  "
             f"params={search.best_params_}")
    return best


def main():
    ap = argparse.ArgumentParser(
        description="Per-joint hyperparameter search (file-grouped CV) for "
                    "PerJointTreeModel/PerJointForestModel.")
    ap.add_argument("--model", choices=sorted(SEARCH_SPACES), default="tree_per_joint",
                    help="which model's hyperparameters to search (default: %(default)s)")
    ap.add_argument("--n-iter", type=int, default=25,
                    help="RandomizedSearchCV candidates per joint (default: %(default)s)")
    ap.add_argument("--cv-splits", type=int, default=5,
                    help="GroupKFold splits, capped at the number of recordings "
                        "(default: %(default)s)")
    ap.add_argument("--csvs", nargs="+", default=DEFAULT_CSVS,
                    help="recordings to search over (default: all of data/*.csv)")
    args = ap.parse_args()

    best = tune(args.model, args.n_iter, args.cv_splits, args.csvs)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(RESULTS_DIR, f"tuning_{args.model}_{dt_str}.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[tuning] -> {out_path}")


if __name__ == "__main__":
    main()
