"""Evaluate a distilled model on recordings it never trained on.

``train_distillation_model.py``'s validation split (``loaders``' ``val_frac``) is
by move, but two moves from the *same recording* as a training move can still be
near-duplicates of each other (same script, same speed). ``data/ur5e/heldout/``
holds whole recordings (T09/T10) the model never saw at all -- this is the honest
"does it generalize to a trajectory it's never seen" number, not just the
random-split one ``fit()`` logs to ``runs/``.

    python evaluate.py --model models/distill-ur5e.pkl --data data/ur5e/heldout
"""
from __future__ import annotations

import argparse

import numpy as np

from common import segments
from train_distillation_model import DistillModel, load_recordings
from utils import get_block


def evaluate(model: DistillModel, recordings) -> dict:
    """{base: (err, std)} per predicted channel, one row per recorded sample.

    Each move (plus its settle window, ``common.segments``) is predicted in one
    call so the model sees the same causal context it was trained on, then rows
    across all moves and recordings are pooled per channel.
    """
    rows = {b: ([], []) for b in model.predicts()}
    for rec in recordings:
        for s in segments(rec):
            sub = rec.df.iloc[s.i0:s.i2]
            pred = model.predict(sub)
            for base in model.predicts():
                err = np.abs(pred["mean"][base] - get_block(sub, base))
                std = np.sqrt(np.clip(pred["var"][base], 0, None))
                rows[base][0].append(err)
                rows[base][1].append(std)
    return {b: (np.concatenate(e), np.concatenate(s)) for b, (e, s) in rows.items()}


def main():
    ap = argparse.ArgumentParser(
        description="Score a distilled model on recordings it never trained on.")
    ap.add_argument("--model", default="models/distill-ur5e.pkl", help="distilled model pickle")
    ap.add_argument("--data", required=True, help="folder of held-out recordings")
    args = ap.parse_args()

    model = DistillModel.load(args.model)
    recordings = load_recordings(args.data)
    print(f"{len(recordings)} recordings from {args.data}")

    for base, (err, std) in evaluate(model, recordings).items():
        coverage = float((err <= std).mean())
        print(f"{base}: mean {err.mean() * 1000:.4f} mrad   "
             f"median {np.median(err) * 1000:.4f} mrad   "
             f"p95 {np.percentile(err, 95) * 1000:.4f} mrad   "
             f"coverage {coverage:.3f} (ideal ~0.68)")


if __name__ == "__main__":
    main()
