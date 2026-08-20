"""Train and evaluate ``PooledLinearModel`` (one shared-slope linear fit
across all six joints, position target, no gravity/lag) for the presentation
comparison against ``PerJointPositionModelNoGravity``, ``PerJointPositionModel``,
and ``PerJointTreeModel`` -- see ``train_distillation_model.py``'s
``PooledLinearModel`` docstring for why this needs its own script rather
than the shared ``--model`` CLI: ``main()``'s row-level diagnostic assumes a
per-joint-keyed design, which this model's single stacked-across-joints fit
doesn't produce.

Mirrors ``train_distillation_model.py --out models/distill.pkl``'s
methodology as closely as possible (same ``DEFAULT_CSVS``/``DEFAULT_HELDOUT_CSVS``/
``DEFAULT_HOLDOUT``, same row-level-then-refit-on-100% flow, same
``log_run`` output shape) so this run's ``results/<timestamp>/`` folder is
directly comparable to the other three models':

    python train_pooled_linear.py
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from preprocess import default_preprocess
from train_distillation_model import (DEFAULT_CSVS, DEFAULT_HELDOUT_CSVS, DEFAULT_HOLDOUT,
                                      N_JOINTS, PooledLinearModel, _compute_metrics,
                                      _evaluate_model, log_run)


def _pooled_row_split_eval(model: PooledLinearModel, recordings, holdout: float) -> dict:
    """Row-level held-out diagnostic for ``PooledLinearModel``'s flat, all-
    joints-stacked design -- same deterministic every-Nth-row methodology as
    ``train_distillation_model._row_split_eval``, but operating on the flat
    ``(X, y, target_q, joint_ids, groups)`` design instead of a per-joint
    dict (this model has no per-joint design to key by, see its docstring).
    Reconstructs ``{"actual_q": {joint: {...}}}`` via ``joint_ids`` so the
    result plugs into the existing ``_compute_metrics``/``log_run`` unchanged.
    """
    X, y, tgt, jids, _groups = model._design(recordings)
    n = len(y)
    step = max(int(round(1 / holdout)), 2)
    is_test = np.arange(n) % step == 0

    coef, *_ = np.linalg.lstsq(X[~is_test], y[~is_test], rcond=None)
    pred_res = X[is_test] @ coef
    tgt_test = tgt[is_test]
    jid_test = jids[is_test]
    actual = tgt_test + y[is_test]
    pred = tgt_test + pred_res

    eval_data = {"actual_q": {}}
    for j in range(N_JOINTS):
        mask = jid_test == j
        eval_data["actual_q"][j] = {
            "pred": pred[mask], "actual": actual[mask], "target": tgt_test[mask],
            "residuals": pred[mask] - actual[mask],
        }
    return eval_data


def main():
    from analysis import Recording

    pre = default_preprocess()
    recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                 for r in (Recording(p) for p in DEFAULT_CSVS)]

    model = PooledLinearModel()
    holdout = DEFAULT_HOLDOUT

    held_out_eval = _pooled_row_split_eval(model, recordings, holdout)
    held_out_metrics = _compute_metrics(held_out_eval)
    n_rows = len(held_out_eval["actual_q"][0]["actual"])
    print(f"row-level held-out ({holdout:.0%} of rows, ~{n_rows} rows/joint) "
         f"from {len(recordings)} run(s)")
    ovr = held_out_metrics["actual_q"]["overall"]
    print(f"  actual_q: RMSE={ovr['rmse']:.4f}  R2={ovr['r2']:.4f}")
    for pj in held_out_metrics["actual_q"]["per_joint"]:
        print(f"    {pj['joint']:10s} RMSE={pj['rmse']:.4f}  R2={pj['r2']:.4f}")

    # Refit on 100% of the rows -- this is the model that ships.
    model.fit(recordings)
    in_sample_metrics = _compute_metrics(_evaluate_model(model, recordings))
    out_path = "models/distill_pooled_linear.pkl"
    model.save(out_path)
    print(f"refit on 100% of {len(recordings)} run(s), saved {out_path}")

    heldout_eval = None
    if DEFAULT_HELDOUT_CSVS:
        heldout_recordings = [Recording(r.path, df=pre.transform_distill(r.df))
                              for r in (Recording(p) for p in DEFAULT_HELDOUT_CSVS)]
        heldout_eval = _evaluate_model(model, heldout_recordings)
        heldout_metrics = _compute_metrics(heldout_eval)
        n_heldout_rows = len(heldout_eval["actual_q"][0]["actual"])
        print(f"file-level heldout ({n_heldout_rows} rows/joint) from "
             f"{len(heldout_recordings)} unseen file(s)")
        ovr = heldout_metrics["actual_q"]["overall"]
        print(f"  actual_q: RMSE={ovr['rmse']:.4f}  R2={ovr['r2']:.4f}")
        for pj in heldout_metrics["actual_q"]["per_joint"]:
            print(f"    {pj['joint']:10s} RMSE={pj['rmse']:.4f}  R2={pj['r2']:.4f}")
    else:
        print("no heldout csvs found -- skipping file-level heldout eval")

    dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_run(model, DEFAULT_CSVS, holdout, held_out_eval, in_sample_metrics, dt_str=dt_str,
           heldout_csvs=DEFAULT_HELDOUT_CSVS if DEFAULT_HELDOUT_CSVS else None,
           heldout_eval=heldout_eval)


if __name__ == "__main__":
    main()
