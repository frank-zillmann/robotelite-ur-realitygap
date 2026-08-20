"""Rebuild the commanded path of every recording, offline, as the RL training bank.

convert.py gets a path by running a script on the controller and recording what it
commanded. That is the right thing for a *new* script, but every recording in
``data/`` already contains exactly the same signal -- ``target_q`` at ``utils.DT``,
with ``script_control_line`` marking the moves. So the whole training set can be
rebuilt without a robot:

    python paths.py --data data/ur5e --model models/distill-ur5e.pkl --robot UR5e \\
        --out models/bank-ur5e.pkl

``cycle`` is the offline twin of ``convert.convert``, with one difference that
matters. convert runs the script itself with ``loop=2`` and can therefore assume
two passes; collect_data.py records with ``--cycles 3``, so that assumption gives
one and a half cycles and a path that does not close -- by whole radians, not a
rounding error. Counting ``movej`` lines in the script does not rescue it either: a
zero-length closing move (a triangle's ``C3 -> C1 -> C1`` across the loop seam) is
dropped by ``common.segments``, so the effective period is not the written one.
``cycle`` measures the period from the recording instead, and asserts closure.

Each item carries everything scoring it needs -- the block boundaries, the per-joint
travel, the recorded path's own error and limit overshoot -- so the RL environment
never recomputes them and every worker sees identical baselines.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

import optimize as O
from analysis import Recording
from common import segments
from utils import DT, N_JOINTS, Robot

CYCLE_TOL = 1e-3        # rad: two segment starts this close are the same waypoint
MIN_ROWS = 150          # a sub-path shorter than the model's warm-up is not worth scoring
MIN_BLOCKS = 2
# Pooled copies are several unrelated runs concatenated. ``cycle`` will happily find
# a "period" across that seam and return a path that is really the tail of one script
# and the head of another -- a trajectory no robot ever executed. Skipped by name,
# for the same reason train_distillation_model.load_recordings skips it.
SKIP = ("pooled.csv",)


def cycle(rec, tol: float = CYCLE_TOL) -> tuple[np.ndarray, int]:
    """One closed steady-state cycle of a recording, as ``(q, period)``.

    The period is the smallest ``P`` for which some segment's start pose repeats ``P``
    segments later; the cycle returned is the *second* full pass from there, which is
    the steady state -- it begins where the previous pass ended, so it closes on
    itself and does not carry the approach move in from wherever the robot was
    parked. Same intent as ``convert.convert``, without assuming how many times the
    script was looped.
    """
    segs = segments(rec)
    if len(segs) < 2:
        raise ValueError(f"{rec.path}: {len(segs)} moves, need at least 2")
    start = np.array([rec.target_q[s.i0] for s in segs])
    for period in range(1, len(segs) // 2 + 1):
        for a in range(len(segs) - 2 * period + 1):
            if np.abs(start[a] - start[a + period]).max() < tol:
                cyc = segs[a + period: a + 2 * period]
                i0, i2 = cyc[0].i0, cyc[-1].i2
                # A pooled recording holds several runs end to end, and two of them
                # can meet at a pose similar enough to look like a period. The result
                # would be the tail of one script joined to the head of another: a
                # trajectory nothing ever executed, and the seam is a step the model
                # would be asked to explain.
                if len(np.unique(rec.script[i0:i2])) > 1:
                    raise ValueError(f"{rec.path}: the cycle spans two source runs")
                q = rec.target_q[i0:i2]
                gap = np.abs(q[0] - q[-1]).max()
                if gap > tol:
                    raise ValueError(f"{rec.path}: cycle does not close ({gap * 1e3:.2f} mrad)")
                return q, period
    raise ValueError(f"{rec.path}: no repeating waypoint in {len(segs)} moves")


@dataclass
class Item:
    """One path the agent can be asked to optimize, and its baseline.

    ``before`` and ``pen0`` are the recorded path scored by the distilled model and
    by ``optimize.penalty``: the numbers every reward is expressed relative to. They
    are computed once here so that the environment -- six worker processes of it --
    cannot disagree about what the baseline was.
    """

    name: str
    q: np.ndarray            # (n, N_JOINTS) commanded angles at DT
    bnd: np.ndarray          # (n_blocks + 1,) movable block boundary times
    travel: np.ndarray       # (N_JOINTS,) total distance each joint covers
    T0: float
    gap_mean: float          # mean |predicted gap| over the recorded path
    sd_mean: float           # mean of the model's own uncertainty over it
    pen0: float              # its limit overshoot; must be 0 for a recorded path
    gap_rms: np.ndarray      # (N_JOINTS,) per-joint rms gap, i.e. which joints ring
    peak0: float             # worst gap while the recorded path stands still
    settle0: float           # how long it rings there
    v_tcp_max: float
    obs: np.ndarray = None

    @property
    def n_blocks(self) -> int:
        return len(self.bnd) - 1

    def baseline(self, k: float = None) -> float:
        """The recorded path's own error, at ``k`` standard deviations of caution.

        Every reward is a ratio to this, so it has to be available at whatever ``k``
        the caller scores candidates with -- otherwise the numerator and denominator
        are differently risk-averse and the recorded path stops scoring 1. It is
        exact rather than a refit because the mean is linear in ``k``:
        ``mean(|gap| + k*sd) == mean|gap| + k*mean(sd)``.
        """
        return self.gap_mean + (O.K if k is None else k) * self.sd_mean

    @property
    def before(self) -> float:
        return self.baseline()


def observe(item: Item, robot: Robot) -> np.ndarray:
    """The path descriptor the policy reads: shape, timing and where it hurts.

    Everything is a ratio or an angle, so no component carries a unit that would
    dominate the rest before normalization. The per-block entries are padded out to
    ``optimize.B_MAX`` with zeros, matching the dead tail of the action.
    """
    dur = np.diff(item.bnd)
    # A block is a pause if the path barely moves through it. Reading it off travel
    # rather than off movable again keeps this independent of the threshold there.
    q, bnd = item.q, item.bnd
    rows = np.round(bnd / DT).astype(int).clip(0, len(q) - 1)
    span = np.array([np.abs(q[b] - q[a]).max() for a, b in zip(rows[:-1], rows[1:])])
    pad = lambda v: np.concatenate([v, np.zeros(O.B_MAX - len(v))])[:O.B_MAX]
    return np.concatenate([
        [item.n_blocks / O.B_MAX, np.log(item.T0), np.log(max(item.before, 1e-9))],
        pad(dur / item.T0),                       # where the time goes
        pad((span < 1e-3).astype(float)),         # which blocks are pauses
        pad(span / max(span.max(), 1e-9)),        # how far each block travels
        np.sin(q[0]), np.cos(q[0]),               # the pose it starts from
        item.travel / max(item.travel.max(), 1e-9),
        item.gap_rms / max(item.before, 1e-9),    # which joints the model says ring
        [item.v_tcp_max / robot.v_tcp,            # how much speed headroom is left
         item.peak0 / max(item.before, 1e-9),     # how much of the error is the ring
         item.settle0 / item.T0,                  # how much of the cycle is ringing
         item.sd_mean / max(item.before, 1e-9)],  # how much of it the model is guessing
    ]).astype(np.float32)


def measure(name: str, q: np.ndarray, model, robot: Robot) -> Item:
    """Score one recorded path with the model and wrap it as an ``Item``."""
    ref = torch.as_tensor(np.asarray(q, np.float32))
    with torch.inference_mode():
        gap, sd = O.predict_gap(model, ref)
    gap, sd = gap.numpy(), sd.numpy()
    still = O.movable(q).numpy() < 0.5
    peak0, _ = O.ring(gap, still, DT, np.inf)          # the peak sets the band
    _, settle0 = O.ring(gap, still, DT, max(O.BAND_FRAC * peak0, O.BAND_FLOOR))
    return Item(name=name, q=np.asarray(q, float), bnd=O.bounds(q),
                travel=np.abs(np.diff(q, axis=0)).sum(0),
                T0=(len(q) - 1) * DT,
                gap_mean=float(np.abs(gap).mean()), sd_mean=float(sd.mean()),
                pen0=float(O.penalty(ref, robot)),
                gap_rms=np.sqrt((gap ** 2).mean(0)),
                peak0=peak0, settle0=settle0,
                v_tcp_max=float(robot.tcp_speed(q).max()))


REPEAT = 2          # also train on each cycle run twice, for the block counts it adds
REPEAT_MIN_BLOCKS = 6


def repeat(q: np.ndarray, n: int = REPEAT) -> np.ndarray:
    """The closed cycle run ``n`` times end to end.

    Physically the same thing as ``send.py --loop n``, so it is a real trajectory and
    not a synthetic one -- and it is the cheapest way to cover block counts the
    recordings do not reach on their own. It matters because the action is a fixed
    vector whose ``k``-th coordinate drives the ``k``-th block: a policy trained only
    on 2-to-7-block paths has never exercised the coordinates an 8-block path uses,
    and produces an untrained, infeasible answer when it meets one.

    Sub-paths cannot fill that hole -- every one has fewer blocks than its parent, so
    they pile up at the low end and make the imbalance worse.

    The last row is dropped from every pass but the last: a closed cycle ends where it
    began, so keeping both would repeat a row and read as a zero-length step.
    """
    return np.vstack([q[:-1]] * (n - 1) + [q])


def sub_paths(q: np.ndarray, bnd: np.ndarray):
    """Every contiguous run of whole blocks, as ``(name suffix, rows)``.

    Blocks begin and end at a standstill -- a move block is one ``movej``, a pause
    block is the sleep after it -- so any run of them is itself a legal trajectory a
    ``servoj`` stream could execute from rest. That is what makes this augmentation
    safe where a random row crop is not: a crop starting mid-move would begin at
    speed, which no path can.
    """
    rows = np.round(bnd / DT).astype(int).clip(0, len(q) - 1)
    n = len(bnd) - 1
    for a in range(n):
        for b in range(a + MIN_BLOCKS, n + 1):
            if (a, b) == (0, n):
                continue                                   # that is the whole path
            piece = q[rows[a]: rows[b] + 1]
            if len(piece) >= MIN_ROWS:
                yield f"[{a}:{b}]", piece


class PathBank:
    """The items the agent trains on, plus the statistics used to normalize them.

    The observation normalizer lives here rather than in a ``VecNormalize`` wrapper
    on purpose: the observation distribution is a fixed finite set, so a running
    estimate buys nothing, and keeping it in the bank means a saved policy cannot be
    loaded without the statistics it was trained under.
    """

    def __init__(self, items: list[Item], robot: str):
        if not items:
            raise ValueError("empty bank")
        self.items, self.robot = items, robot
        obs = np.stack([it.obs for it in items])
        self.mean = obs.mean(0)
        self.std = np.maximum(obs.std(0), 1e-6)       # a constant component stays 0

    CONST = 1e-6      # a component with this std never varied in the bank
    CLIP = 20.0       # above anything the training set itself reaches (19.85)

    def normalize(self, obs) -> np.ndarray:
        """Standardise an observation, safely for data the bank never saw.

        Two guards, both for evaluating a policy on paths outside its bank:

        A component that is *constant* here carries no information, and a model
        trained on this bank only ever saw it as exactly 0. Dividing an unseen value
        by its 1e-6 std instead sends it a million sigma out -- which is what made a
        held-out evaluation look like a total failure to generalize. Constant
        components are therefore held at 0 rather than scaled, which is exactly what
        training saw.

        The rest are clipped, at a bound deliberately above the training set's own
        maximum, so the guard cannot alter the inputs the policy was fitted on while
        still keeping an outlier from dominating them.
        """
        z = np.where(self.std <= self.CONST, 0.0, (obs - self.mean) / self.std)
        return np.clip(z, -self.CLIP, self.CLIP).astype(np.float32)

    def stats(self) -> dict:
        """The normalizer, to be saved with a policy trained under it."""
        return {"mean": self.mean, "std": self.std}

    def adopt(self, stats: dict) -> "PathBank":
        """Use another bank's normalizer -- the one a policy was trained with.

        A policy's inputs are defined by the statistics it saw in training, so
        evaluating it on a different bank must not renormalize with that bank's own
        statistics. Doing so silently feeds the policy a different input space and
        looks exactly like a failure to generalize.
        """
        self.mean, self.std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        return self

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PathBank":
        with open(path, "rb") as f:
            return pickle.load(f)


def build(folder: str, model, robot: Robot, sub: bool = True,
          exclude=("heldout",)) -> PathBank:
    """Every recording in ``folder`` as a closed path, plus its sub-paths.

    ``exclude`` names folders whose contents can never enter a bank, checked on the
    path of every file rather than left to the shape of the glob: a held-out
    trajectory that leaks in makes every number downstream meaningless, and it fails
    silently. Pointing ``--data`` straight at one is an error rather than an empty
    bank, since that is a mistake and not a request.
    """
    items, seen, held = [], set(), 0
    for p in sorted(glob.glob(f"{folder}/**/*.csv", recursive=True)):
        parts = os.path.normpath(p).split(os.sep)
        if any(x in parts for x in exclude):
            held += 1
            continue
        if os.path.basename(p) in SKIP:
            print(f"  skipping {p}: unrelated runs concatenated, not one trajectory")
            continue
        stem = os.path.basename(p)[:-4]
        try:
            q, period = cycle(Recording(p))
        except Exception as e:
            print(f"  skipping {stem}: {type(e).__name__} {e}")
            continue
        it = measure(stem, q, model, robot)
        if it.pen0 > 1e-6:
            print(f"  skipping {stem}: the recorded path already breaks its own "
                  f"limits by {it.pen0 * 100:.2f}%, so it is not a usable baseline")
            continue
        items.append(it)
        print(f"  {stem:24s} period {period}  {len(q):5d} rows  {it.T0:6.2f} s  "
              f"{it.n_blocks} blocks  before {it.before * 1000:6.3f} mrad")
        seen.add(stem)
        if sub:
            items += [measure(stem + s, piece, model, robot)
                      for s, piece in sub_paths(q, it.bnd)]
        # The doubled cycle, plus only those of its sub-paths that are richer in
        # blocks than the recordings already provide -- the rest would just deepen
        # the pile at the low end.
        if not sub:
            continue                       # --no-sub-paths means no augmentation at
                                           # all, which is what an evaluation set wants
        q2 = repeat(q)
        it2 = measure(f"{stem}x{REPEAT}", q2, model, robot)
        if it2.n_blocks > O.B_MAX:
            print(f"  {stem}: doubled to {it2.n_blocks} blocks, over B_MAX={O.B_MAX}; skipped")
        else:
            items.append(it2)
            if sub:
                items += [measure(f"{stem}x{REPEAT}{s}", piece, model, robot)
                          for s, piece in sub_paths(q2, it2.bnd)
                          if len(O.bounds(piece)) - 1 >= REPEAT_MIN_BLOCKS]
    if held:
        print(f"  held out {held} recording(s) under {'/, '.join(exclude)}/")
    if not items:
        raise SystemExit(f"no usable paths in {folder}"
                         + (f" -- all {held} of them are held out; a held-out set is "
                            "for evaluating, not for building a bank from" if held else ""))
    for it in items:
        it.obs = observe(it, robot)
    print(f"\n{len(seen)} recordings -> {len(items)} items "
          f"({len(items) - len(seen)} sub-paths)")
    return PathBank(items, robot.model)


def main():
    ap = argparse.ArgumentParser(description="Build the RL path bank from recordings.")
    ap.add_argument("--data", required=True, help="folder of recordings, e.g. data/ur5e")
    ap.add_argument("--model", required=True, help="distilled model, for the baselines")
    ap.add_argument("--robot", required=True, choices=list(Robot.MODELS))
    ap.add_argument("--out", required=True, help="bank pickle to write")
    ap.add_argument("--no-sub-paths", action="store_true",
                    help="no augmentation: the recorded cycles only, no sub-paths and "
                         "no repeats. What an evaluation set should be.")
    ap.add_argument("--held-out", action="store_true",
                    help="build from the held-out runs instead of excluding them. For "
                         "evaluating a trained policy, never for training one -- the "
                         "bank it writes must not be passed to `rl_optimize train`.")
    args = ap.parse_args()

    # Import under the real module name (not "__main__") so the pickled Items carry
    # `paths.Item` and load anywhere, the same reason train_distillation_model does it.
    from paths import build as _build
    from train_distillation_model import DistillModel

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)          # the bank only ever scores, never fits
    bank = _build(args.data, model, Robot(args.robot), sub=not args.no_sub_paths,
                  exclude=() if args.held_out else ("heldout",))
    if args.held_out:
        print("  NOTE: this bank is the held-out set. Evaluate with it; never train on it.")
    bank.save(args.out)
    print(f"wrote {args.out}: {len(bank.items)} items, obs dim {len(bank.mean)}")


if __name__ == "__main__":
    main()
