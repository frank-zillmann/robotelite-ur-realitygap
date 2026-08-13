"""Distilled model that predicts ``actual_*`` channels for a commanded trajectory.

Trained on recorded real runs, the model predicts what an actual channel would be
for a given commanded trajectory. Applied to the sim's targets (whose
``actual_*`` columns read 0.0), it overwrites those columns with predictions so
the metric and the RL agent can run without hardware.

``DistillModel`` is the interface the pipeline depends on:

    fit(recordings)  -> learn from recorded real runs
    predicts()       -> which actual_* channels this model fills in
    predict(df)      -> for a commanded-trajectory DataFrame, (mean, std) of the
                        actual channels as {channel: (n, N_JOINTS)}; std is None
                        if the model has no notion of spread

The channels a model fills must be the ones the metric reads (see metrics.py).
``CNNModel``, the one implementation here, is a causal temporal CNN that predicts
the gap over a whole move.

    from train_distillation_model import CNNModel, augment
    from analysis import Recording
    m = CNNModel().fit([Recording("data/test-4.csv"), Recording("data/test-6.csv")])
    m.predicts()                          # ['actual_current']
    m.save("models/distill.pkl")
    augment(m, "sim_to_real.csv")         # overwrite actual_current with predictions

Run as a script to train on every run in ``data/``, print the error on the moves
held out by ``common.loaders``, and save.

    python train_distillation_model.py --out models/distill.pkl

train_rla.py and run.py depend only on the interface, so a custom subclass of
DistillModel can replace this one via its pickle.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import RESIDUAL, N_FEAT, blocks, features, loaders, moves, standardize
from preprocess import Identity, Preprocess, default_preprocess
from utils import JOINT_NAMES, N_JOINTS, get_block, set_block


class DistillModel(ABC):
    """Interface every distilled model must implement.

    Subclasses implement ``fit``, ``predicts``, and ``predict``, and optionally
    override ``bounds``.
    """

    @abstractmethod
    def fit(self, recordings) -> "DistillModel":
        """Train on a list of ``analysis.Recording`` (real robot runs)."""

    @abstractmethod
    def predicts(self) -> list[str]:
        """Per-joint channel bases this model predicts, e.g. ``["actual_current"]``.

        These are the ``actual_*`` columns ``predict`` returns and ``augment``
        overwrites in the recording.
        """

    @abstractmethod
    def predict(self, df) -> tuple[dict, dict | None]:
        """``(mean, std)`` for a commanded-trajectory DataFrame.

        ``df`` carries ``t``, ``target_q*``, ``target_qd*`` and the commanded
        ``vel``/``acc``. ``mean`` is ``{base: (n, N_JOINTS)}`` for every base in
        ``predicts()``; the caller overwrites those columns with it. ``std`` is
        the same shape, or ``None`` if the model has no notion of spread. A
        risk-averse objective can score ``mean + k * std``.
        """

    def bounds(self):
        """Optional ``((vel_lo, vel_hi), (acc_lo, acc_hi))`` the model trusts.

        Return the raw-number range the training data covered so the optimizer
        stays in-distribution, or ``None`` to let the caller pick its own.
        """
        return None

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "DistillModel":
        with open(path, "rb") as f:
            return pickle.load(f)


LOGVAR = (-9.0, 4.0)       # clamp keeping the Gaussian NLL well conditioned


class _Block(nn.Module):
    """Residual block reaching ``dilation * (kernel-1)`` rows further back."""

    def __init__(self, ch, dilation, kernel):
        super().__init__()
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.gain = nn.Parameter(torch.full((ch, 1), 0.1))   # start near identity

    def forward(self, x):
        # Left-pad only, so the output has the input's length and row t sees no
        # row after t. No norm layer: Batch/Group/LayerNorm pool over time, which
        # would make a prediction depend on the whole frame instead of a fixed
        # window, so training crops and inference frames would disagree.
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return x + self.gain * self.mix(nn.functional.gelu(y))


class CNNModel(DistillModel):
    """Causal dilated CNN over the commanded trajectory, six joints at once.

    The gap is dynamic, not per-row: after each stop the joint rings down for
    0.3-0.5 s, friction flips with ``sign(qd)``, and the bias drifts with pose.
    So row ``t`` is predicted from the last ``1 + (kernel-1)*sum(dilations)`` rows
    (127 by default, ~1.0 s at 128 Hz), which dilations cover in 6 layers instead
    of 63. Joints and quantities are all channels, so the net can mix across them.

    ``targets`` picks the channels, each learned as a residual on its commanded
    twin (``RESIDUAL``); switch ``metrics.py`` to match. ``members > 1`` makes it
    a deep ensemble; either way ``predict`` returns a spread.
    """

    def __init__(self, targets=("actual_current",), hidden: int = 48,
                 dilations=(1, 2, 4, 8, 16, 32), kernel: int = 3, members: int = 1,
                 epochs: int = 25, batch: int = 16, lr: float = 3e-3,
                 seed: int = 0, verbose: bool = True):
        self.targets = tuple(targets)
        self.epochs, self.batch = epochs, batch
        self.lr, self.seed, self.verbose = lr, seed, verbose
        self.pad = (kernel - 1) * sum(dilations)      # warm-up rows per sequence
        self.n_out = len(self.targets) * N_JOINTS
        self.train_dt = self.stats = None
        self.vel_range = self.acc_range = self.val_idx = None
        self.nets = []
        for m in range(members):
            torch.manual_seed(seed + m)
            # 1x1 embed -> dilated residual blocks -> 1x1 head of (mean, log_var).
            net = nn.Sequential(nn.Conv1d(N_FEAT, hidden, 1),
                                *[_Block(hidden, d, kernel) for d in dilations],
                                nn.Conv1d(hidden, 2 * self.n_out, 1))
            # A zeroed head starts every output at 0: mean = the average gap and
            # log_var = 0 (unit variance in standardised space). That is the best
            # constant predictor and keeps the first NLL steps well conditioned.
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

    def predicts(self) -> list[str]:
        return list(self.targets)

    def bounds(self):
        return None if self.vel_range is None else (self.vel_range, self.acc_range)

    # --- fit ------------------------------------------------------------------

    def _loss(self, net, xb, yb, mask, nll: bool):
        """Squared error, or Gaussian NLL once the mean is sane, over the real rows."""
        out = net(xb)[..., self.pad:]                 # drop the warm-up columns
        mu, lv = out[:, :self.n_out], out[:, self.n_out:].clamp(*LOGVAR)
        per = ((yb - mu) ** 2 if not nll
               else 0.5 * (lv + (yb - mu) ** 2 * torch.exp(-lv)))
        return (per * mask[:, None]).sum() / (mask.sum() * self.n_out)

    def fit(self, recordings) -> "CNNModel":
        """Train every member on the training moves of the recorded runs."""
        train, val = loaders(recordings, self.targets, self.pad, self.batch,
                             seed=self.seed)
        ds = train.dataset.dataset            # the MoveDataset behind the Subset
        self.stats, self.train_dt = ds.stats, ds.dt
        self.vel_range, self.acc_range = ds.vel_range, ds.acc_range
        self.val_idx = list(val.dataset.indices)   # into common.moves(recordings)
        if self.verbose:
            print(f"{len(train.dataset)} train / {len(val.dataset)} val moves, "
                  f"receptive field {self.pad + 1} rows "
                  f"({(self.pad + 1) * self.train_dt:.2f} s)")

        for m, net in enumerate(self.nets):
            # Members differ only by their random init and the order they draw
            # moves in; that is enough disagreement for a deep ensemble.
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, self.lr, self.epochs * len(train), pct_start=0.2)
            for ep in range(self.epochs):
                net.train()
                total = 0.0
                for xb, yb, mask in train:
                    loss = self._loss(net, xb, yb, mask, ep >= self.epochs // 3)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                    sched.step()
                    total += float(loss.detach())
                if self.verbose and ep % 5 == 4:
                    net.eval()
                    with torch.inference_mode():
                        v = np.mean([float(self._loss(net, *b, ep >= self.epochs // 3))
                                     for b in val])
                    print(f"  member {m + 1}  epoch {ep + 1:3d}/{self.epochs}  "
                          f"loss {total / len(train):+.4f}  val {v:+.4f}")
            net.eval()
        return self

    # --- predict --------------------------------------------------------------

    def predict(self, df):
        if self.stats is None:
            raise RuntimeError("not fitted: call fit() or load a pickle")
        _, _, my, sy = self.stats
        regrid = lambda z, src, dst: np.column_stack([np.interp(dst, src, c) for c in z.T])
        out = []
        for sl in blocks(df):                 # never filter across a script seam
            sub = df.iloc[sl]
            x, dt = features(sub, self.pad)
            # A learned temporal filter only holds at the rate it was trained on,
            # and path mode asks for 4-20 ms servoj steps against 7.8 ms
            # recordings, so resample onto the training grid and back.
            t, grid = np.arange(len(sub)) * dt, None
            if len(sub) > 1 and abs(dt - self.train_dt) > 0.01 * self.train_dt:
                grid = np.arange(0.0, t[-1] + self.train_dt / 2, self.train_dt)
                x = np.vstack([x[:self.pad], regrid(x[self.pad:], t, grid)])
            with torch.inference_mode():
                xb = torch.from_numpy(standardize(x, self.stats).T[None]).float()
                o = np.stack([n(xb)[0, :, self.pad:].T.numpy() for n in self.nets])
            mu, var = o[..., :self.n_out], np.exp(o[..., self.n_out:].clip(*LOGVAR))
            # Ensemble mixture: mean of variances (aleatoric) + variance of means.
            mean, std = mu.mean(0), np.sqrt(var.mean(0) + mu.var(0))
            if grid is not None:
                mean, std = regrid(mean, grid, t), regrid(std, grid, t)
            out.append((mean * sy + my, std * sy))
        gap, spread = (np.vstack(v) for v in zip(*out))
        cols = lambda k: slice(k * N_JOINTS, (k + 1) * N_JOINTS)
        return ({b: gap[:, cols(k)] + get_block(df, RESIDUAL[b])
                 for k, b in enumerate(self.targets)},
                {b: spread[:, cols(k)] for k, b in enumerate(self.targets)})


def augment(model: DistillModel, csv: str, pre: Preprocess = None):
    """Overwrite a recording's actual_* columns with the model's predictions.

    Reads ``csv``, asks ``model.predict`` for the channels in
    ``model.predicts()``, writes them back into the same columns, and saves in
    place. The schema is unchanged; the actual_* columns become predictions,
    ready for metrics.py.

    ``pre`` (a preprocess.Preprocess) is applied around the model as in training:
    ``transform_distill`` before predict, ``revert_distill`` after, so the saved
    columns land back in real units. Defaults to a no-op ``Identity``.
    """
    pre = pre or Identity()
    df = pre.transform_distill(pd.read_csv(csv))
    preds, _ = model.predict(df)
    for base in model.predicts():
        set_block(df, base, preds[base])
    df = pre.revert_distill(df)
    df.to_csv(csv, index=False)
    print(f"overwrote {model.predicts()} with predictions -> {csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Train the distillation model.")
    ap.add_argument("--out", default="models/distill.pkl", help="pickle path")
    args = ap.parse_args()

    # Import under the real module name (not "__main__") so the saved pickle
    # loads cleanly in train_rla.py and run.py.
    from train_distillation_model import CNNModel
    from analysis import Recording

    # Every run, preprocessed the way the model will see it later.
    pre = default_preprocess()
    recordings = [Recording(p, df=pre.transform_distill(pd.read_csv(p)))
                  for p in sorted(glob.glob("data/test-*.csv"))]
    model = CNNModel().fit(recordings)

    # Score the held-out moves, predicting each exactly as the RL env would. The
    # gap actual-target is what is scored: an R2 against raw actual_q would read
    # 0.9999 for a model that only echoes target_q and says nothing.
    got = {b: [] for b in model.predicts()}
    want = {b: [] for b in model.predicts()}
    all_moves = moves(recordings)
    for rec, seg in (all_moves[i] for i in model.val_idx):
        sub = rec.df.iloc[seg.i0:seg.i2]
        pred, _ = model.predict(sub)
        for base in got:
            ref = get_block(sub, RESIDUAL[base])
            got[base].append(np.asarray(pred[base], dtype=float) - ref)
            want[base].append(get_block(sub, base) - ref)
    for base in got:
        y = np.vstack(want[base])
        mse, var = ((np.vstack(got[base]) - y) ** 2).mean(0), np.maximum(y.var(0), 1e-12)
        print(f"  held-out {base} gap ({len(y)} rows)")
        for j, name in enumerate(JOINT_NAMES):
            print(f"    {name:10s} RMSE {np.sqrt(mse[j]):9.4f}   R2 {1 - mse[j]/var[j]:7.4f}")
        print(f"    {'mean':10s} RMSE {np.sqrt(mse).mean():9.4f}   "
              f"R2 {(1 - mse / var).mean():7.4f}")

    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
