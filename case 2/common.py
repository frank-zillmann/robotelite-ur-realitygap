"""Cut a recording into moves and turn them into model input.

``segments`` is the one shared definition of "a move", used by the distillation and
the RL stages so both cut a recording the same way:

    for seg in segments(Recording("sim_to_real.csv")):
        print(seg.joint, seg.i0, seg.i1, seg.i2, seg.dist)

The rest is the data preparation the distilled models share, so a new architecture
only has to bring its own network and training loop:

    train, val = loaders(recordings, targets=("actual_q",), pad=127)
    for xb, yb, mask in train:   # (b, N_FEAT, pad+n), (b, n_out, n), (b, n)
        ...
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, WeightedRandomSampler,
                              random_split)

from utils import N_JOINTS, SCRIPT_COL, frame_dt, get_block

SETTLE_S = 1.0             # settle kept after a move ends; the rest of a pause is idle

# Each channel is learned as ``actual - target``, the gap itself: more accurate than
# predicting the channel outright, and GapMetric is |actual - target|, so the
# score reduces to |predicted gap|, independent of dynamics.py.
RESIDUAL = {"actual_current": "target_current", "actual_q": "target_q",
            "actual_qd": "target_qd"}

N_FEAT = 5 * N_JOINTS      # sin q, cos q, qd, qdd, qddd (see ``features``)
X_CLIP = 6.0               # standardised features are clipped to this many sigma


@dataclass
class Segment:
    """One coordinated waypoint-to-waypoint move, as indices into a Recording.

    All joints move together over ``[i0, i1]`` and settle over ``[i1, i2]``.
    ``joint`` is the widest-travel joint, the one representing the segment in the
    observation; ``start``/``dest``/``dist`` are its angles (rad). ``vel``/``acc``
    are the commanded movej numbers, or None if not logged.
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

    ``script_control_line`` (scl) names the running movej line, reading 0 on the
    sleeps and overhead in between, so a segment is the run of rows of one line::

        3 0 3 3 0 3 0 | 4 4 0 4 0 4 4 0 | 5 5

    Zeros belong to the current segment; only a change to another nonzero line ends
    it. Segmenting runs per source script (rows carry their script name), so no
    segment spans two pooled recordings.

    - ``i0``  the movej's first row.
    - ``i1``  one past its last, where the settle window (the ring) begins. A blend
      gives ``i1 == i2``, i.e. no settle window.
    - ``i2``  the next movej's start or the recording's end, at most ``SETTLE_S``
      past ``i1`` -- the rest of a pause is a robot standing still, and the last
      movej would otherwise carry the whole idle tail of the recording.

    The last movej is dropped if its line was still running at the end (partial),
    so N completed movejs give N segments.
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

def features(df, pad: int = 0):
    """Per-row model inputs ``(n + pad, N_FEAT)``, from the commanded trajectory only.

    - ``sin q``, ``cos q``  pose, wrap-safe, and what gravity and inertia vary with.
    - ``qd``, ``qdd``       the commanded motion.
    - ``qddd``              jerk, what excites the ring (worth ~5% of the error).

    Not ``target_current``: in a recording that column is the controller's, but in
    a candidate frame it is only dynamics.py's estimate, and the metric already
    subtracts it. Not the movej ``vel``/``acc`` registers either: they were worth
    nothing measurable and are absent from a servoj path.
    """
    d = lambda z: np.gradient(z, frame_dt(df), axis=0)
    q, qd = get_block(df, "target_q"), get_block(df, "target_qd")
    qdd = d(qd)
    x = np.column_stack([np.sin(q), np.cos(q), qd, qdd, d(qdd)]).astype(np.float32)
    if pad:
        # Padding: the start pose held still
        rest = x[:1].copy()
        rest[:, 12:N_FEAT] = 0.0 # feature columns that are zero when the robot stands still
        x = np.vstack([np.repeat(rest, pad, axis=0), x])
    return x


def standardize(x, stats):
    """Standardise features, clipped to +-``X_CLIP`` sigma.

    Unclipped, a model asked about a speed regime it never saw extrapolates to
    thousands of amps.
    """
    return np.clip((x - stats[0]) / stats[1], -X_CLIP, X_CLIP).astype(np.float32)


class MoveDataset(Dataset):
    """The recorded moves, one per item, each at its own natural length.

    Item ``i`` is ``(x, y)`` for ``self.moves[i] = (recording, segment)``: inputs
    ``(N_FEAT, pad + n)`` and the measured gap ``(len(targets) * N_JOINTS, n)``,
    standardised with ``self.stats``, which the model reuses at predict time. The
    leading ``pad`` rows are the network's warm-up.
    """

    def __init__(self, recordings, targets=("actual_q",), pad: int = 0):
        self.moves = [(rec, s) for rec in recordings for s in segments(rec)]
        seqs = []
        for rec, s in self.moves:
            sub = rec.df.iloc[s.i0:s.i2]      # motion plus the capped settle window
            gap = np.column_stack([get_block(sub, b) - get_block(sub, RESIDUAL[b])
                                   for b in targets]).astype(np.float32)
            seqs.append((features(sub, pad), gap))

        X = np.concatenate([x for x, _ in seqs])
        Y = np.concatenate([y for _, y in seqs])
        self.stats = (X.mean(0), X.std(0), Y.mean(0), Y.std(0))
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

    The network is causal, so zeros appended on the right cannot reach a real
    output; ``mask`` (1 per real row) keeps the padding out of the loss.
    """
    n = max(y.shape[1] for _, y in batch)
    fill = lambda z, k: torch.nn.functional.pad(z, (0, n - k))
    return (torch.stack([fill(x, y.shape[1]) for x, y in batch]),
            torch.stack([fill(y, y.shape[1]) for _, y in batch]),
            torch.stack([fill(torch.ones(y.shape[1]), y.shape[1]) for _, y in batch]))


def loaders(recordings, targets=("actual_q",), pad: int = 0, batch: int = 16,
            val_frac: float = 0.2, seed: int = 0):
    """``(train, val)`` DataLoader over whole moves of ``recordings``.

    - Split by whole move, never by row: at 128 Hz a held-out row's neighbours
      would sit in the training set and be nearly identical to it.
    - Training moves are drawn with probability proportional to their length, so
      every recorded row is equally likely to be learned from.
    - ``stats`` covers all moves, not only the training ones; a mean and a std over
      ~1e6 rows barely move, so the leak is negligible.
    - Either loader wraps a ``Subset`` of one ``MoveDataset``:
      ``train.dataset.dataset`` carries ``stats`` and ``moves``, ``.indices`` says
      which moves it kept.
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

    ``sim_to_real.csv`` pools several scripts; a sequence model must not run across
    the seam, where the joints jump from one script's end pose to the next's start.
    """
    if SCRIPT_COL not in df or df[SCRIPT_COL].nunique() < 2:
        return [slice(0, len(df))]
    tag = df[SCRIPT_COL].to_numpy()
    cut = [0] + [i for i in range(1, len(tag)) if tag[i] != tag[i - 1]] + [len(tag)]
    return [slice(a, b) for a, b in zip(cut, cut[1:])]
