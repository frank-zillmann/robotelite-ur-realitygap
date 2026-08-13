"""Split a recording into coordinated waypoint-to-waypoint segments and turn them into model input.

One shared definition of "a move", used by the distillation and RL stages so they
all cut the recording the same way. A segment is the full recorded path from one
waypoint to the next: all joints moving together, dynamic length.

    from analysis import Recording
    from common import segments
    for seg in segments(Recording("sim_to_real.csv")):
        print(seg.joint, seg.i0, seg.i1, seg.i2, seg.dist)

The second half is the data preparation the distilled models share, so a new
architecture only has to bring its own network and training loop:

    from common import loaders
    train, val = loaders(recordings, targets=("actual_current",), pad=127)
    for xb, yb, mask in train:      # (batch, N_FEAT, pad+n), (batch, n_out, n), (batch, n)
        ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              random_split)

from utils import (ACC_COL, N_JOINTS, SCRIPT_COL, VEL_COL, frame_dt, get_block)

# Seconds of settle kept after a move ends; the rest of the pause is dropped.
SETTLE_S = 1.0

# Each channel is learned as ``actual - target``, the gap itself: more accurate
# than predicting the channel outright, and CurrentGapMetric is |actual - target|
# so the score reduces to |predicted gap|, independent of dynamics.py.
RESIDUAL = {"actual_current": "target_current", "actual_q": "target_q",
            "actual_qd": "target_qd"}

N_FEAT = 6 * N_JOINTS + 2  # sin q, cos q, qd, qdd, qddd, tanh(qd/eps) | vel, acc
REST = slice(12, 36)       # feature columns that are zero when the robot stands still
X_CLIP = 6.0               # standardised features are clipped to this many sigma


@dataclass
class Segment:
    """One coordinated waypoint-to-waypoint move, as indices into a Recording.

    All joints move together over ``[i0, i1]`` (the movej) and settle over
    ``[i1, i2]`` (at most ``SETTLE_S``). ``joint`` is the widest-travel joint, which represents the
    segment in the observation; ``start``/``dest``/``dist`` are that joint's angles
    (rad). ``vel``/``acc`` are the commanded movej numbers, or None if not logged.
    """

    joint: int
    i0: int
    i1: int
    i2: int
    start: float
    dest: float
    dist: float
    vel: float = None
    acc: float = None


def segments(rec) -> list[Segment]:
    """Split a recording into waypoint-to-waypoint segments, one per movej.

    A ``movej`` moves the robot from wherever it is to a waypoint, so one movej is
    one segment. The controller reports which movej line is running in
    ``script_control_line`` (scl): it holds that line's number through the move,
    reads 0 on the sleeps and overhead in between, and changes to the next movej's
    line when the next move starts. So a segment is the run of rows belonging to one
    movej line, from where that line first appears to where the next movej line
    appears. For example the scl sequence

        3 0 3 3 0 3 0 | 4 4 0 4 0 4 4 0 | 5 5

    is one segment on line 3, the next on line 4, then the start of line 5 (dropped,
    see below). Zeros belong to the current segment; only a change to a different
    nonzero line ends it.

    Segmentation runs within each source script separately (rows carry their script
    name, see Recording.script), so a segment never spans two pooled recordings.
    An interior movej is bounded by the next movej. The last movej of a script is
    bounded by the recording's end, and is dropped if its line was still running
    there, which leaves it partial. So N completed movejs give N segments.

    ``i0`` is where the movej begins and ``i1`` one row past its last, so ``[i1, i2]``
    is the settle window where the ring lives. A blend (two movejs with no stop
    between) gives ``i1 == i2``: that segment simply has no settle window. ``i2`` is
    where the next movej begins (or the recording ends), but at most ``SETTLE_S``
    past ``i1``: the rest of a long pause is a robot standing still, and the last
    movej of a run would otherwise carry the whole idle remainder of the recording.
    """
    settle = max(1, int(round(SETTLE_S / rec.dt)))
    scl, script = rec.scl, rec.script
    n = len(scl)
    # Split the run into contiguous same-script blocks; segment each on its own.
    starts = [0] + [i for i in range(1, n) if script[i] != script[i - 1]] + [n]
    out = []
    for a, b in zip(starts, starts[1:]):
        # Onset of each distinct movej line in this block (0s are transparent).
        bounds, last = [], 0
        for i in range(a, b):
            if scl[i] != 0 and scl[i] != last:
                bounds.append(i)
                last = scl[i]
        for k, i0 in enumerate(bounds):
            i2 = bounds[k + 1] if k + 1 < len(bounds) else b   # next movej, or end
            same = np.nonzero(scl[i0:i2] == scl[i0])[0]        # rows on this movej line
            i1 = i0 + int(same[-1]) + 1                        # one past the movej's end
            if k + 1 == len(bounds) and i1 >= i2:
                continue                            # last movej never ended: partial, drop
            if i1 - i0 < 3:
                continue                            # too few motion samples: not a real move
            i2 = min(i2, i1 + settle)
            j = int(np.argmax(np.abs(rec.target_q[i1] - rec.target_q[i0])))
            start, dest = float(rec.target_q[i0, j]), float(rec.target_q[i1, j])
            vel = float(rec.vel_cmd[i0]) if rec.vel_cmd is not None else None
            acc = float(rec.acc_cmd[i0]) if rec.acc_cmd is not None else None
            out.append(Segment(j, i0, i1, i2, start, dest, abs(dest - start), vel, acc))
    return out


# --- data preparation shared by the distilled models -------------------------

def moves(recordings):
    """Every usable ``(recording, segment)``, in a fixed order."""
    return [(rec, s) for rec in recordings for s in segments(rec) if s.i2 - s.i0 >= 4]


def features(df, pad: int = 0):
    """Per-row model inputs ``(n + pad, N_FEAT)`` and the frame's sample period.

    Pose as sin/cos (wrap-safe, and what gravity and inertia vary with), the
    commanded motion, jerk because that is what excites the ring, and a smooth
    ``sign(qd)`` for Coulomb friction. ``REST`` spans the blocks that vanish at
    standstill. Deliberately not ``target_current``: in a recording that column
    is the controller's, but in a candidate frame it is only dynamics.py's
    estimate of it.
    """
    dt = frame_dt(df)
    dt = dt if np.isfinite(dt) and dt > 0 else 1.0
    d = lambda x: np.gradient(x, dt, axis=0) if len(x) > 1 else np.zeros_like(x)
    col = lambda c: (df[c].to_numpy(float) if c in df else np.zeros(len(df)))[:, None]
    q, qd = get_block(df, "target_q"), get_block(df, "target_qd")
    qdd = d(qd)
    x = np.column_stack([np.sin(q), np.cos(q), qd, qdd, d(qdd), np.tanh(qd / 0.05),
                         col(VEL_COL), col(ACC_COL)]).astype(np.float32)
    if pad:
        # Warm-up rows: the start pose held still, which is what the robot really
        # does before a movej, so a short candidate frame is primed exactly the
        # way training pads it and row 0 is meaningful.
        rest = x[:1].copy()
        rest[:, REST] = 0.0
        x = np.vstack([np.repeat(rest, pad, axis=0), x])
    return x, dt


def standardize(x, stats):
    """Standardise features, clipped to +-``X_CLIP`` sigma.

    The clip matters: asked about a speed regime it never saw, an unclipped
    model extrapolates to thousands of amps.
    """
    return np.clip((x - stats[0]) / stats[1], -X_CLIP, X_CLIP).astype(np.float32)


class MoveDataset(Dataset):
    """The recorded moves, one per item, each at its own natural length.

    Item ``i`` is ``(x, y)`` for ``self.moves[i] = (recording, segment)``: inputs
    ``(N_FEAT, pad + n)`` and the measured gap ``(len(targets) * N_JOINTS, n)``,
    standardised with ``self.stats`` (which the model reuses at predict time).
    The leading ``pad`` rows are the warm-up the network needs before its first
    real output.
    """

    def __init__(self, recordings, targets=("actual_current",), pad: int = 0):
        self.moves = moves(recordings)
        if not self.moves:
            raise ValueError("no moves found in the recordings")
        seqs = []
        for rec, s in self.moves:
            sub = rec.df.iloc[s.i0:s.i2]      # motion plus the capped settle window
            gap = np.column_stack([get_block(sub, b) - get_block(sub, RESIDUAL[b])
                                   for b in targets]).astype(np.float32)
            seqs.append((features(sub, pad)[0], gap))

        X = np.concatenate([x for x, _ in seqs])
        Y = np.concatenate([y for _, y in seqs])
        self.stats = (X.mean(0), np.maximum(X.std(0), 1e-6),
                      Y.mean(0), np.maximum(Y.std(0), 1e-6))
        # The last two feature columns are vel and acc: their spans are the
        # raw-number range the model was trained on.
        self.vel_range = (float(X[:, -2].min()), float(X[:, -2].max()))
        self.acc_range = (float(X[:, -1].min()), float(X[:, -1].max()))
        self.dt = float(np.median([r.dt for r in recordings]))
        my, sy = self.stats[2], self.stats[3]
        self.data = [(standardize(x, self.stats), (y - my) / sy) for x, y in seqs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return torch.from_numpy(x.T).float(), torch.from_numpy(y.T).float()


def collate(batch):
    """Stack moves of different lengths: right-pad to the longest, plus a mask.

    The network is causal, so zeros appended on the right cannot reach any real
    output; ``mask`` (1 per real row) keeps the padding out of the loss.
    """
    n = max(y.shape[1] for _, y in batch)
    fill = lambda z, k: torch.nn.functional.pad(z, (0, n - k))
    return (torch.stack([fill(x, y.shape[1]) for x, y in batch]),
            torch.stack([fill(y, y.shape[1]) for _, y in batch]),
            torch.stack([fill(torch.ones(y.shape[1]), y.shape[1]) for _, y in batch]))


def loaders(recordings, targets=("actual_current",), pad: int = 0, batch: int = 16,
            val_frac: float = 0.2, seed: int = 0):
    """``(train, val)`` DataLoader over whole moves of ``recordings``.

    ``random_split`` holds out ``val_frac`` of the moves, never rows: at 128 Hz a
    held-out row's neighbours would be in the training set and nearly identical
    to it. Training moves are drawn with probability proportional to their length,
    so every recorded row is equally likely to be learned from. ``stats`` covers
    all moves, not only the training ones: a mean and a std over ~1e6 rows barely
    move, so the leak is negligible. The dataset behind
    either loader is a ``Subset`` of one ``MoveDataset``: ``train.dataset.dataset``
    carries ``stats`` and the move list, ``.indices`` says which moves it kept.
    """
    ds = MoveDataset(recordings, targets, pad)
    g = torch.Generator().manual_seed(seed)
    n_val = int(val_frac * len(ds))
    train, val = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    w = [len(ds.data[i][1]) for i in train.indices]
    return (DataLoader(train, batch, collate_fn=collate,
                       sampler=WeightedRandomSampler(w, len(w), generator=g)),
            DataLoader(val, batch, collate_fn=collate))


def blocks(df):
    """Row slices of ``df`` that are one continuous trajectory each.

    ``sim_to_real.csv`` pools several scripts; a sequence model must not run
    across the seam, where the joints jump from one script's end pose to the
    next's start.
    """
    if SCRIPT_COL not in df or df[SCRIPT_COL].nunique() < 2:
        return [slice(0, len(df))]
    tag = df[SCRIPT_COL].to_numpy()
    cut = [0] + [i for i in range(1, len(tag)) if tag[i] != tag[i - 1]] + [len(tag)]
    return [slice(a, b) for a, b in zip(cut, cut[1:])]
