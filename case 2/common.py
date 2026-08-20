"""Cut a recording into moves and turn them into model input.

``segments`` is the one shared definition of "a move", used by the distillation and
the RL stages so both cut a recording the same way:

    for seg in segments(Recording("data/ur5e/T01_fast_r1.csv")):
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

from utils import DT, N_JOINTS, SCRIPT_COL, get_block

SETTLE_S = 1.0             # settle kept after a move ends; the rest of a pause is idle

# Each channel is learned as ``actual - target``, the gap itself: more accurate than
# predicting the channel outright, and the optimizer scores |actual - target|,
# so its objective is simply |predicted gap|.
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

def _diff(x, dt: float):
    """``np.gradient``'s rule in torch: central inside, one-sided at both ends."""
    return torch.cat([(x[1:2] - x[:1]) / dt,
                      (x[2:] - x[:-2]) / (2 * dt),
                      (x[-1:] - x[-2:-1]) / dt])


def features(q, dt: float = DT, pad: int = 0):
    """Per-row model inputs ``(n + pad, N_FEAT)`` from the commanded angles alone.

    - ``sin q``, ``cos q``  pose, wrap-safe, and what gravity and inertia vary with.
    - ``qd``, ``qdd``       the commanded motion.
    - ``qddd``              jerk, what excites the ring (worth ~5% of the error).

    Everything is differentiated from ``q`` rather than read from ``target_qd``, so
    a trajectory the optimizer invents is turned into inputs exactly the way a
    recording is. Torch throughout, so the optimizer can differentiate through it.
    Not ``target_current``: nothing models torque any more.
    """
    q = q if torch.is_tensor(q) else torch.as_tensor(np.asarray(q, np.float32))
    qd = _diff(q, dt)
    qdd = _diff(qd, dt)
    x = torch.cat([torch.sin(q), torch.cos(q), qd, qdd, _diff(qdd, dt)], dim=1)
    if pad:
        # Warm-up: the start pose held still, which is what the robot really does
        # before a move, so row 0 of a short frame is already meaningful.
        still = torch.zeros(1, N_FEAT)
        still[:, :2 * N_JOINTS] = 1.0        # keep the pose, drop the motion
        x = torch.cat([(x[:1] * still).expand(pad, -1), x])
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
        for rec in recordings:                # everything downstream assumes DT
            if abs(rec.dt - DT) > 0.05 * DT:
                raise ValueError(f"{rec.path} runs at {rec.dt * 1000:.2f} ms, not "
                                 f"{DT * 1000:.2f}; re-record it with --hz {1 / DT:.0f}")
        self.moves = [(rec, s) for rec in recordings for s in segments(rec)]
        seqs = []
        for rec, s in self.moves:
            sub = rec.df.iloc[s.i0:s.i2]      # motion plus the capped settle window
            gap = np.column_stack([get_block(sub, b) - get_block(sub, RESIDUAL[b])
                                   for b in targets]).astype(np.float32)
            seqs.append((features(get_block(sub, "target_q"), DT, pad).numpy(), gap))

        X = np.concatenate([x for x, _ in seqs])
        Y = np.concatenate([y for _, y in seqs])
        self.stats = (X.mean(0), X.std(0), Y.mean(0), Y.std(0))
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

    A pooled recording holds several scripts; a sequence model must not run across
    the seam, where the joints jump from one script's end pose to the next's start.
    """
    if SCRIPT_COL not in df or df[SCRIPT_COL].nunique() < 2:
        return [slice(0, len(df))]
    tag = df[SCRIPT_COL].to_numpy()
    cut = [0] + [i for i in range(1, len(tag)) if tag[i] != tag[i - 1]] + [len(tag)]
    return [slice(a, b) for a, b in zip(cut, cut[1:])]
