"""Distilled model that predicts ``actual_*`` channels for a commanded trajectory.

Trained on recorded real runs, it predicts what an actual channel would be for a
given commanded trajectory. Applied to the sim's targets (whose ``actual_*``
columns read 0.0), it overwrites those columns so metrics.py and the RL agent can
run without hardware.

``DistillModel`` is the interface the pipeline depends on:

    fit(recordings)  -> learn from recorded real runs
    predicts()       -> which actual_* channels this model fills in
    predict(df)      -> (mean, std) of those channels, {channel: (n, N_JOINTS)}

The channels must be the ones the metric reads (metrics.py). ``CNNModel``, the one
implementation here, is a causal temporal CNN over a whole move.

    m = CNNModel().fit([Recording("data/test-4.csv"), Recording("data/test-6.csv")])
    m.save("models/distill.pkl")
    augment(m, "sim_to_real.csv")         # overwrite actual_current with predictions

As a script: train on every run in ``data/``, log to ``runs/``, save the pickle.

    python train_distillation_model.py --out models/distill.pkl
"""
from __future__ import annotations

import argparse
import glob
import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from common import RESIDUAL, N_FEAT, blocks, features, loaders, standardize
from preprocess import Identity, Preprocess, default_preprocess
from utils import N_JOINTS, get_block, set_block


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
        """``(mean, std)`` as ``{base: (n, N_JOINTS)}`` for a commanded-trajectory
        frame (``t``, ``target_q*``, ``target_qd*``) sampled at the training rate.
        ``std`` is None without a notion of spread; a risk-averse objective can
        score ``mean + k * std``."""

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


LOGVAR = (-9.0, 4.0) # clamp keeping the Gaussian NLL well conditioned


class _Block(nn.Module):
    """Residual block reaching ``dilation * (kernel-1)`` rows further back."""

    def __init__(self, ch, dilation, kernel):
        super().__init__()
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.mix = nn.Conv1d(ch, ch, 1)
        self.gain = nn.Parameter(torch.full((ch, 1), 0.1))   # start near identity

    def forward(self, x):
        # Left-pad only: the output keeps the input's length and row t sees no later
        # row. No norm layer -- Batch/Group/LayerNorm pool over time, which would make
        # a prediction depend on the whole frame instead of a fixed window.
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return x + self.gain * self.mix(nn.functional.gelu(y))


class CNNModel(DistillModel):
    """Causal dilated CNN over the commanded trajectory, all joints at once.

    Joints and quantities are channels, so the net mixes across them; each target is
    learned as a residual on its commanded twin (``RESIDUAL``). ``members > 1`` makes
    it a deep ensemble, trained jointly; ``predict`` returns the ensemble mean and
    its std, aleatoric (predicted log-variance) plus epistemic (disagreement).
    """

    def __init__(self, targets=("actual_current",), hidden: int = 48,
                 dilations=(1, 2, 4, 8, 16, 32), kernel: int = 3, members: int = 3,
                 epochs: int = 50, batch: int = 16, lr: float = 3e-3,
                 val_frac: float = 0.2, seed: int = 0):
        self.targets = tuple(targets)
        self.epochs, self.batch, self.val_frac = epochs, batch, val_frac
        self.lr, self.seed = lr, seed
        self.pad = (kernel - 1) * sum(dilations) # warm-up rows per sequence
        self.n_out = len(self.targets) * N_JOINTS
        self.train_dt = self.stats = None
        self.nets = []
        for m in range(members):
            torch.manual_seed(seed + m) # the members' only difference
            # 1x1 embed -> dilated residual blocks -> 1x1 head of (mean, log_var).
            net = nn.Sequential(nn.Conv1d(N_FEAT, hidden, 1),
                                *[_Block(hidden, d, kernel) for d in dilations],
                                nn.Conv1d(hidden, 2 * self.n_out, 1))
            # Zeroed head: every output starts at 0, so mean = the average gap and
            # log_var = 0 (unit variance), the best constant predictor.
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

    def predicts(self) -> list[str]:
        return list(self.targets)

    # --- fit ------------------------------------------------------------------

    def parameters(self):
        return [p for net in self.nets for p in net.parameters()]

    def forward(self, x):
        """Standardised ``(B, N_FEAT, pad + T)`` -> ``(mu, log_var)``, both
        ``(members, B, n_out, T)`` with the warm-up columns dropped."""
        out = torch.stack([net(x)[..., self.pad:] for net in self.nets])
        return out[:, :, :self.n_out], out[:, :, self.n_out:].clamp(*LOGVAR)

    def loss(self, xb, yb, mask, warmup: bool):
        """``(objective, info)`` for one batch, averaged over the real rows.

        ``info`` is what gets logged; the ``err/*`` entries are de-standardised, so
        they are in the target's own unit (A for currents, rad for angles):

            loss/objective    what this step minimizes: mse warming up, nll after
            loss/sq           precision-weighted squared error, mean over members
            loss/logvar       0.5*log var, the penalty on being wide
            loss/nll          sq + logvar, the Gaussian NLL
            loss/mse          plain squared error
            err/mean          |ensemble mean - measured|
            err/std           |predicted std - that error|
            err/coverage      share of rows inside +-1 std (0.68 if calibrated)
            err/disagreement  std of the member means
        """
        mu, lv = self.forward(xb)
        w = mask[:, None].expand_as(yb)             # padded rows count for nothing
        mean = lambda z: (z * w).sum() / w.sum()
        sq = mean((0.5 * (yb - mu) ** 2 * torch.exp(-lv)).mean(0))
        logvar = mean(0.5 * lv.mean(0))
        mse = mean(((yb - mu) ** 2).mean(0))
        objective = mse if warmup else sq + logvar
        # Ensemble mixture: mean of variances (aleatoric) + variance of means. The
        # label std turns a standardised difference back into the real unit.
        sy = torch.as_tensor(self.stats[3])[:, None]
        err = (yb - mu.mean(0)).abs() * sy
        std = torch.sqrt(torch.exp(lv).mean(0) + mu.var(0, unbiased=False)) * sy
        return objective, {
            "loss/objective": objective, "loss/sq": sq, "loss/logvar": logvar,
            "loss/nll": sq + logvar, "loss/mse": mse,
            "err/mean": mean(err), "err/std": mean((std - err).abs()),
            "err/coverage": mean((err <= std).float()),
            "err/disagreement": mean(mu.std(0, unbiased=False) * sy)}

    def _epoch(self, loader, warmup, opt=None, sched=None):
        """One pass over ``loader``, optimizing if ``opt`` is given. Mean of ``info``."""
        logs = []
        for xb, yb, mask in loader:
            with torch.set_grad_enabled(opt is not None):
                objective, info = self.loss(xb, yb, mask, warmup)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                objective.backward()
                # The NLL weights the error by exp(-log_var), so one outlier row under
                # a confident prediction can otherwise blow up the step.
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()
                sched.step()
            logs.append({k: float(v.detach()) for k, v in info.items()})
        return {k: float(np.mean([g[k] for g in logs])) for k in logs[0]}

    def fit(self, recordings) -> "CNNModel":
        """Train the ensemble, logging epochs 0 (before any step) to ``epochs``."""
        train, val = loaders(recordings, self.targets, self.pad, self.batch,
                             self.val_frac, self.seed)
        ds = train.dataset.dataset # the MoveDataset behind the Subset
        self.stats, self.train_dt = ds.stats, ds.dt
        run = f"runs/{time.strftime('%Y%m%d-%H%M%S')}"
        print(f"{len(train.dataset)} train / {len(val.dataset)} val moves, receptive "
              f"field {self.pad + 1} rows ({(self.pad + 1) * self.train_dt:.2f} s)\n"
              f"logging to {run}, watch with: tensorboard --logdir runs")
        train_log = SummaryWriter(f"{run}/train")
        val_log = SummaryWriter(f"{run}/val") if len(val.dataset) else None

        # One optimizer over all members: their parameters are disjoint, so this is
        # the same training as one loop each, with the data loaded once.
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, self.lr, self.epochs * len(train), pct_start=0.2)
        for ep in range(self.epochs + 1):
            warmup = ep <= self.epochs // 3   # squared error first, then the NLL
            # Epoch 0 only measures, so every curve starts before the first step.
            info = self._epoch(train, warmup, *((opt, sched) if ep else ()))
            info["lr"] = sched.get_last_lr()[0]
            for k, v in info.items():
                train_log.add_scalar(k, v, ep)
            if val_log:
                for k, v in self._epoch(val, warmup).items():
                    val_log.add_scalar(k, v, ep)
        train_log.close()
        if val_log:
            val_log.close()
        return self

    # --- predict --------------------------------------------------------------

    def predict(self, df):
        if self.stats is None:
            raise RuntimeError("not fitted: call fit() or load a pickle")
        _, _, my, sy = self.stats
        out = []
        for sl in blocks(df): # never filter across a script seam
            # ``df`` must be sampled at ``train_dt``: a learned temporal filter only
            # holds at its training rate (train_rla.py builds its frames that way).
            x = standardize(features(df.iloc[sl], self.pad), self.stats)
            with torch.inference_mode():
                mu, lv = self.forward(torch.from_numpy(x.T[None]).float())
                # Ensemble mixture: mean of variances + variance of means.
                std = torch.sqrt(torch.exp(lv).mean(0) + mu.var(0, unbiased=False))
                out.append((mu.mean(0)[0].T.numpy() * sy + my, std[0].T.numpy() * sy))
        gap, spread = (np.vstack(v) for v in zip(*out))
        cols = lambda k: slice(k * N_JOINTS, (k + 1) * N_JOINTS)
        return ({b: gap[:, cols(k)] + get_block(df, RESIDUAL[b])
                 for k, b in enumerate(self.targets)},
                {b: spread[:, cols(k)] for k, b in enumerate(self.targets)})


def augment(model: DistillModel, csv: str, pre: Preprocess = None):
    """Overwrite ``csv``'s actual_* columns with the model's predictions, in place.

    ``pre`` wraps the model as in training (``transform_distill`` before predict,
    ``revert_distill`` after), so the saved columns land back in real units.
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

    # Import under the real module name (not "__main__") so the pickle loads
    # cleanly in train_rla.py and run.py.
    from train_distillation_model import CNNModel
    from analysis import Recording

    pre = default_preprocess() # every run, as the model will see it later
    recordings = [Recording(p, df=pre.transform_distill(pd.read_csv(p)))
                  for p in sorted(glob.glob("data/test-*.csv"))]
    CNNModel().fit(recordings).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
