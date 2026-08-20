"""A PPO policy over the same objective optimize.py differentiates.

optimize.py solves one path at a time: 600 Adam steps, and the next path starts from
nothing. It is also confined to terms a gradient can flow through, and it starts at
the recorded path and descends -- so it finds the nearest local optimum, which on
these recordings is not the good one. The pauses a script sleeps through are 25-43%
of the cycle on half the trajectories, and ``phase``'s uniform slices cannot cut one
without stretching the move beside it.

This lane changes three things and keeps everything else identical:

- **the basis** -- one time scale per ``movable`` block (4 to 9 of them, each wholly
  a move or wholly a pause) instead of 24 slices that straddle both, plus a coarser
  offset spline. See ``optimize.bounds`` / ``optimize.phase_blocks``.
- **the search** -- a policy trained across many paths, so an unseen one costs a
  forward pass instead of a solve. That is the point: not a better optimum per path,
  an optimum that is already known.
- **the objective** -- optionally widened with the ring terms (``--w-peak``,
  ``--w-settle``) that no gradient can reach.

The reward is ``optimize.loss`` negated and shifted so the recorded path scores
exactly 0, i.e. the reward *is* the improvement over what the controller does today.
With the ring weights at 0 it is the gradient lane's objective term for term.

    python rl_optimize.py preflight --bank models/bank-ur5e.pkl --model ... --robot UR5e
    python rl_optimize.py train     --bank models/bank-ur5e.pkl --model ... --robot UR5e
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

import optimize as O
from paths import PathBank
from utils import DT, N_JOINTS, Robot

# The ring terms, off by default: with both at 0 the reward is optimize.loss exactly,
# which is what makes --compare a comparison rather than two different questions.
W_PEAK, W_SETTLE = 0.0, 0.0

REWARD_SCALE = 2.0


def squash(reward: float) -> float:
    """Compress the *training* reward's negative tail, keeping its order.

    A good action scores 0.2 to 1.0, but one that breaks a joint limit scores -55,
    because ``penalty`` is unbounded while the time term caps at 2.93. That 70:1 ratio
    makes the value network span 56 units of reward to resolve differences of 0.1, and
    leaves every advantage dominated by disasters.

    Clipping at a floor fixes the scale and breaks something worse: it tells the policy
    that a slight violation and a catastrophic one are equally bad, so it stops
    avoiding the catastrophic one. Measured -- a policy trained against a hard floor
    went on to emit a path violating the limits by 3.6x on the one path where the gain
    had to come from speeding up rather than from cutting a pause.

    So squash rather than clip: identity to first order at 0, asymptotic to
    ``-REWARD_SCALE``, and increasing, so worse stays worse. The ordering is what
    matters and it survives where it counts -- -0.5, -1, -2 and -5 map to -0.44,
    -0.79, -1.26 and -1.84, a gradient the policy can act on. Past about -60 it
    saturates into floating point and stops telling two disasters apart, which is
    exactly the region where the difference means nothing anyway.

    Reporting (eval, compare, CEM) uses the true reward; this shapes learning only.
    """
    if reward >= 0.0:
        return reward
    return -REWARD_SCALE * (1.0 - np.exp(reward / REWARD_SCALE))


def action_space_bounds(n_points: int):
    """``(low, high)`` for the action box: symmetric, every coordinate in [-1, 1].

    An earlier version capped the time coordinates at +0.25, to stop a policy that
    wandered upward from spending rollouts scoring 30 000-row trajectories. That put a
    hard wall a quarter of a unit from the neutral action, i.e. right where the policy
    lives, and SB3 clips at the box without correcting the log-prob -- so the mean
    drifted into the clipped region and 19% of a trained policy's time coordinates
    came back pinned to it. The compute it saved was not worth the pathology it
    created: a stretched path is punished by ``ALPHA * T/T0`` anyway, which is a
    gradient the policy can follow, unlike a wall.
    """
    n = O.B_MAX + n_points * N_JOINTS
    return -np.ones(n, np.float32), np.ones(n, np.float32)


def score(model, robot: Robot, item, q, T, k: float = None,
          w_peak: float = W_PEAK, w_settle: float = W_SETTLE) -> dict:
    """Every term of a candidate path's score, named, plus the total.

    ``loss`` is ``optimize.loss`` -- ``err/before + ALPHA*T/T0 + LIMIT*penalty`` --
    with the ring terms added when they are weighted. All four are ratios to the
    recorded path, so it scores ``1 + ALPHA + w_peak + w_settle`` whatever its scale,
    and ``reward`` below turns that into a 0.
    """
    gap, sd = O.predict_gap(model, q)
    before = item.baseline(k)
    err = float((gap.abs() + (O.K if k is None else k) * sd).mean())
    pen = float(O.penalty(q, robot))
    out = {"err": err, "T": float(T), "pen": pen,
           "loss": err / before + O.ALPHA * float(T) / item.T0 + O.LIMIT * pen}
    if w_peak or w_settle:
        qn = q.detach().numpy() if torch.is_tensor(q) else np.asarray(q)
        still = O.movable(qn).numpy() < 0.5
        band = max(O.BAND_FRAC * item.peak0, O.BAND_FLOOR)
        peak, settle = O.ring(gap.detach().numpy(), still, DT, band)
        out["peak"], out["settle"] = peak, settle
        out["loss"] += (w_peak * peak / max(item.peak0, 1e-12)
                        + w_settle * settle / max(item.settle0, DT))
    return out


def baseline_loss(w_peak: float = W_PEAK, w_settle: float = W_SETTLE) -> float:
    """What the recorded path scores, by construction: the zero of the reward."""
    return 1.0 + O.ALPHA + w_peak + w_settle


def evaluate(model, robot: Robot, item, action, k=None, w_peak=W_PEAK,
             w_settle=W_SETTLE) -> tuple[float, dict, np.ndarray, float]:
    """``(reward, terms, q, T)`` for one action on one item."""
    ref = torch.as_tensor(np.asarray(item.q, np.float32))
    free = O.movable(item.q)[:, None]
    with torch.inference_mode():
        q, T = O.decode(action, ref, item.bnd, item.travel, free)
        terms = score(model, robot, item, q, T, k, w_peak, w_settle)
    return baseline_loss(w_peak, w_settle) - terms["loss"], terms, q.numpy(), float(T)


# --- the environment ---------------------------------------------------------

class PathEnv(gymnasium.Env):
    """One action -> one candidate path -> the score optimize.py minimizes.

    A one-step episode: there is no state to carry, because the action already
    describes a whole trajectory and the reward is a deterministic function of it.
    ``action = 0`` replays the recorded path exactly, so a fresh policy starts
    feasible, at the baseline it has to beat, rather than behind a penalty cliff --
    which is what ``optimize.bounds``' pinned last boundary buys.
    """

    metadata = {"render_modes": []}

    def __init__(self, bank_path: str, model_path: str, robot: str,
                 n_points: int = O.RL_POINTS, k: float = None,
                 w_peak: float = W_PEAK, w_settle: float = W_SETTLE, seed: int = 0):
        from train_distillation_model import DistillModel

        self.bank = PathBank.load(bank_path)
        self.model = DistillModel.load(model_path)
        for p in self.model.parameters():
            p.requires_grad_(False)          # this lane only ever scores
        self.robot = Robot(robot)
        self.k, self.w_peak, self.w_settle = k, w_peak, w_settle
        low, high = action_space_bounds(n_points)
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (len(self.bank.mean),), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.item = self.bank.items[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.item = self.bank.items[self.rng.integers(len(self.bank.items))]
        return self.bank.normalize(self.item.obs), {}

    def step(self, action):
        reward, terms, _, _ = evaluate(self.model, self.robot, self.item, action,
                                       self.k, self.w_peak, self.w_settle)
        info = {"path": self.item.name, "loss": terms["loss"], "cycle": terms["T"],
                "err": terms["err"], "pen": terms["pen"], "reward_true": reward,
                "shorter": 1.0 - terms["T"] / self.item.T0}
        return (self.bank.normalize(self.item.obs), squash(reward),
                True, False, info)

    def close(self):
        pass


def make_env(bank_path, model_path, robot, n_points, k, w_peak, w_settle, rank):
    """Factory for ``SubprocVecEnv``: picklable, and it loads its own model.

    Paths rather than objects, because ``spawn`` pickles this closure -- a loaded
    model in it would be serialized once per worker. The thread cap has to be set
    inside the child too: six workers each helping themselves to every core is
    slower than one worker, and the parent's setting does not survive the spawn.
    """
    def _init():
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        return PathEnv(bank_path, model_path, robot, n_points, k, w_peak, w_settle,
                       seed=rank)
    return _init


# --- preflight: is there anything here worth training for? -------------------

def ceiling(item) -> float:
    """The score reached by deleting every pause and touching nothing else.

    Pure geometry, no model: the pauses are dead time, so removing them is a gain
    that costs no tracking error at all. It is a *reference point*, not an upper
    bound -- a search can and does go past it by compressing the moves too, which
    buys more time at some error. Read it as "how much is free here": a path whose
    ceiling is near zero has no dead time to reclaim, and anything it gains has to
    be paid for.
    """
    still = O.movable(item.q).numpy() < 0.5
    moving_frac = 1.0 - still.mean()
    return 1.0 + O.ALPHA * moving_frac


def sweep(model, robot, item, k=None, w_peak=W_PEAK, w_settle=W_SETTLE,
          grid=(-1.0, -0.75, -0.5, -0.25, 0.0, 0.25)):
    """Reward along each block's time coordinate on its own, others neutral.

    Two things to read off it: whether the reward is monotone in the pause blocks
    (it should be -- if not, the model is extrapolating), and whether the best value
    sits *inside* the box. SB3's PPO clips actions at the box without correcting the
    log-prob, so an optimum on the wall makes the mean drift out and ``log_std``
    collapse; this is the direct test that ``T_GAIN``'s tanh scaling avoided that.
    """
    n = O.B_MAX + O.RL_POINTS * N_JOINTS
    out = np.zeros((item.n_blocks, len(grid)))
    for b in range(item.n_blocks):
        for j, v in enumerate(grid):
            a = np.zeros(n, np.float32)
            a[b] = v
            out[b, j] = evaluate(model, robot, item, a, k, w_peak, w_settle)[0]
    return out


def cem(model, robot, item, n_points=O.RL_POINTS, iters=30, pop=64, elite=0.2,
        k=None, w_peak=W_PEAK, w_settle=W_SETTLE, seed=0):
    """Cross-entropy method on the same action space: the black-box ceiling.

    Not a deliverable -- it re-solves every path and amortizes nothing -- but it
    bounds what *any* zeroth-order searcher can get here. If CEM cannot beat the
    gradient lane, neither will PPO, and that is worth knowing in 15 seconds rather
    than after a training run.
    """
    low, high = action_space_bounds(n_points)
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros(len(low), np.float32), 0.4 * np.ones(len(low), np.float32)
    best = (-np.inf, mu.copy())
    n_elite = max(2, int(elite * pop))
    for _ in range(iters):
        a = np.clip(rng.normal(mu, sd, (pop, len(mu))), low, high).astype(np.float32)
        r = np.array([evaluate(model, robot, item, x, k, w_peak, w_settle)[0] for x in a])
        top = a[np.argsort(r)[-n_elite:]]
        mu, sd = top.mean(0), top.std(0) + 1e-3
        if r.max() > best[0]:
            best = (float(r.max()), a[int(np.argmax(r))].copy())
    return best


# --- training ----------------------------------------------------------------

def ppo_kwargs(steps: int):
    """PPO settings for a one-step episode with a dense, already-scaled reward.

    The unusual ones, and why:

    - ``gamma=0``  the episode terminates after one step, so the return *is* the
      reward and the advantage is ``r - V(s)``. Writing 0 says so, and stops anyone
      reading ``n_steps`` as a horizon.
    - ``ent_coef=0``  on a diagonal Gaussian the entropy is ``sum(log_std) + const``,
      so a positive coefficient is an unbounded reward for widening the policy. It
      would push the mean's mass outside the action box, where SB3 clips without
      correcting the log-prob. Exploration comes from ``log_std_init`` instead.
    - ``log_std_init=ln(0.35)``  the box is [-1, 1] and the useful knee of the action's
      tanh is near |a|=0.5, so 1 sd lands on the knee and 3 sd on the wall.
    - no ``VecNormalize``  the reward is already O(1) and scale-free by construction
      (the recorded path scores 0), and normalizing it would destroy exactly the
      comparability the objective was built for. Observations are normalized by the
      bank, whose statistics are saved with the data rather than in a wrapper.
    """
    return dict(
        n_steps=64, batch_size=128, n_epochs=10,
        # Decayed to zero over the run: the reward is deterministic given the action,
        # so late training is refinement, not exploration. Small to begin with, because
        # training starts from cloned solutions that are already good -- this is a
        # fine-tune, and 3e-4 walked away from the clone faster than it improved on it.
        learning_rate=lambda progress_remaining: 1e-4 * progress_remaining,
        gamma=0.0, gae_lambda=1.0, clip_range=0.2, ent_coef=0.0,
        vf_coef=0.5, max_grad_norm=0.5, target_kl=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                           # sigma 0.12, not 0.35: the feasible region of this action
                           # space is narrow, and wide exploration spends most samples
                           # on paths that break a joint limit -- which is what made
                           # the first rollouts score -17 against a cloned +0.56.
                           log_std_init=float(np.log(0.12)), ortho_init=True),
        use_sde=False, device="cpu", verbose=0, seed=0)


_W = {}


def _bc_init(bank_path, model_path, robot, threads: int = 2):
    """Load one model and bank per worker process, once.

    Two threads, not one: unlike a rollout worker this is doing a whole CEM search,
    and nothing else is competing for the machine while it runs.
    """
    from train_distillation_model import DistillModel
    torch.set_num_threads(threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    m = DistillModel.load(model_path)
    for p in m.parameters():
        p.requires_grad_(False)
    _W.update(model=m, robot=Robot(robot), bank=PathBank.load(bank_path))


def _bc_one(job):
    i, n_points, k, w_peak, w_settle, iters, pop = job
    it = _W["bank"].items[i]
    reward, a = cem(_W["model"], _W["robot"], it, n_points=n_points, iters=iters,
                    pop=pop, k=k, w_peak=w_peak, w_settle=w_settle, seed=i)
    return i, a, reward


def _bc_fingerprint(bank_path, model_path, robot, n_items, n_points, k, w_peak,
                    w_settle, iters, pop, seed) -> dict:
    """Everything a cached set of demonstrations depends on.

    An action vector only means something relative to the map that decodes it and the
    objective that scored it, and neither is stored in the vector. Reusing targets
    across a change to either would clone solutions from a different problem -- which
    would look like a training failure, not a stale cache. So the fingerprint covers
    the data, the action space and the objective, and any difference recomputes.
    """
    digest = lambda p: hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]
    return {"bank": digest(bank_path), "model": digest(model_path), "robot": robot,
            "n_items": n_items, "n_points": n_points, "iters": iters, "pop": pop,
            "seed": seed, "k": O.K if k is None else k, "w_peak": w_peak,
            "w_settle": w_settle, "alpha": O.ALPHA, "limit": O.LIMIT, "K": O.K,
            "t_gain": O.T_GAIN, "t_shift": O.T_SHIFT, "a_gain": O.A_GAIN,
            "b_max": O.B_MAX}


def bc_targets(bank_path, model_path, robot, n_items, n_points=0, k=None,
               w_peak=W_PEAK, w_settle=W_SETTLE, procs=6, iters=20, pop=48, seed=0,
               cache: str = None):
    """``(obs, actions, rewards)``: good actions to imitate, found by search.

    The targets come from CEM rather than from the gradient lane, because CEM already
    searches *in the policy's own action space* -- so its answer is an action vector,
    with no projection step and no projection error. (Cloning the gradient lane would
    mean fitting its 16x6 offsets and 24 slices back onto 12 block scales by least
    squares, and inheriting whatever that fit could not represent.)

    These are demonstrations, not labels: CEM optimizes each path on its own and knows
    nothing about the ring terms, so cloning it only puts the policy in the right
    region. What PPO does afterwards is the part that has to generalize.
    """
    from concurrent.futures import ProcessPoolExecutor
    fp = _bc_fingerprint(bank_path, model_path, robot, n_items, n_points, k, w_peak,
                         w_settle, iters, pop, seed)
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        old = json.loads(str(z["fingerprint"]))
        stale = {kk for kk in set(old) | set(fp) if old.get(kk) != fp.get(kk)}
        if not stale:
            print(f"  reusing {cache}: {len(z['rew'])} targets, "
                  f"mean reward {z['rew'].mean():.3f}")
            return z["obs"], z["act"], z["rew"]
        print(f"  {cache} is stale ({', '.join(sorted(stale))} changed), re-searching")

    bank = PathBank.load(bank_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bank.items), size=min(n_items, len(bank.items)), replace=False)
    jobs = [(int(i), n_points, k, w_peak, w_settle, iters, pop) for i in idx]
    out = []
    with ProcessPoolExecutor(max_workers=procs, initializer=_bc_init,
                             initargs=(bank_path, model_path, robot)) as ex:
        for n, (i, a, r) in enumerate(ex.map(_bc_one, jobs), 1):
            out.append((bank.normalize(bank.items[i].obs), a, r))
            if n % 25 == 0:
                print(f"    {n}/{len(jobs)} solved, mean reward so far "
                      f"{np.mean([x[2] for x in out]):.3f}")
    obs, act, rew = (np.stack([o[j] for o in out]) for j in range(3))
    obs, act, rew = (obs.astype(np.float32), act.astype(np.float32),
                     rew.astype(np.float32))
    print(f"  {len(out)} targets, mean reward {rew.mean():.3f}, "
          f"{(rew > 0).mean():.0%} better than the recorded path")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, obs=obs, act=act, rew=rew,
                 fingerprint=np.array(json.dumps(fp)))
        print(f"  cached to {cache}")
    return obs, act, rew


def bc(agent, obs, act, rew, epochs: int = 300, lr: float = 1e-3):
    """Regress the policy's mean onto ``act`` and its value head onto ``rew``.

    Both heads, not just the policy. A cloned mean sitting on top of a random value
    function produces meaningless advantages, and PPO's first updates undo the clone
    before the critic has caught up -- which is the usual reason a warm start appears
    not to help.
    """
    policy = agent.policy
    x = torch.as_tensor(obs)
    ya, yr = torch.as_tensor(act), torch.as_tensor(rew)[:, None]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for ep in range(epochs):
        feats = policy.extract_features(x)
        if isinstance(feats, tuple):
            fa, fv = feats
        else:
            fa = fv = feats
        mu = policy.action_net(policy.mlp_extractor.forward_actor(fa))
        v = policy.value_net(policy.mlp_extractor.forward_critic(fv))
        loss = ((mu - ya) ** 2).mean() + ((v - yr) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ep % 100 == 0 or ep == epochs - 1:
            with torch.no_grad():
                print(f"    bc epoch {ep:4d}  action mse {((mu-ya)**2).mean():.4f}  "
                      f"value mse {((v-yr)**2).mean():.4f}")
    return agent


def norm_path(agent_path: str) -> str:
    """Where a policy's observation normalizer lives, given the policy's path."""
    return agent_path.rsplit(".", 1)[0] + ".norm.npz"


def build_vec(bank_path, model_path, robot, n_points, k, w_peak, w_settle, n_envs):
    """``SubprocVecEnv`` of ``n_envs`` scorers.

    Processes rather than a batched environment: the gap model is small enough that
    one path's forward pass does not fill a core, and its 1x1 convolutions fall off a
    performance cliff at batch 16, so batching would cost more than it saves.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    envs = [make_env(bank_path, model_path, robot, n_points, k, w_peak, w_settle, i)
            for i in range(n_envs)]
    return VecMonitor(SubprocVecEnv(envs, start_method="spawn"))


def train(bank_path, model_path, robot, steps, n_points=O.RL_POINTS, k=None,
          w_peak=W_PEAK, w_settle=W_SETTLE, n_envs=6, warm=None, out=None, run=None,
          bc_items: int = 0, bc_procs: int = 6, bc_cache: str = None):
    """Train a policy over the bank and save it. Returns the agent."""
    from stable_baselines3 import PPO
    # Search for the demonstrations *before* the rollout workers exist. Building the
    # VecEnv first leaves six processes idle, each holding a model and a bank, while
    # six more do the searching -- twelve processes contending for four performance
    # cores, which measured ~6x slower than the search costs on its own.
    demos = None
    if bc_items:
        print(f"  cloning {bc_items} searched solutions before PPO")
        demos = bc_targets(bank_path, model_path, robot, bc_items, n_points,
                           k, w_peak, w_settle, bc_procs, cache=bc_cache)

    venv = build_vec(bank_path, model_path, robot, n_points, k, w_peak, w_settle, n_envs)
    agent = PPO("MlpPolicy", venv, tensorboard_log=run, **ppo_kwargs(steps))
    if warm:
        agent.policy.load_state_dict(PPO.load(warm, device="cpu").policy.state_dict())
        print(f"  warm-started from {warm}")
    if demos:
        bc(agent, *demos)
    agent.learn(total_timesteps=steps, progress_bar=False)
    if out:
        agent.save(out)
        # The normalizer is part of the policy, not part of whatever data it is later
        # pointed at. Saved beside the weights so an evaluation on a different bank
        # cannot silently renormalize the inputs.
        st = PathBank.load(bank_path).stats()
        np.savez(norm_path(out), **st)
    venv.close()
    return agent


# --- applying a trained policy ----------------------------------------------

def agent_solve(agent, model, robot, bank, item, k=None, w_peak=W_PEAK,
                w_settle=W_SETTLE):
    """``(q, T, reward, terms)`` for one path, in one forward pass.

    ``bank`` is not optional: it carries the observation normalizer the policy was
    trained under. Feeding raw observations to a policy trained on normalized ones
    fails silently -- it simply returns near-neutral actions -- so the statistics
    travel with the data rather than in a sidecar beside the weights.
    """
    a, _ = agent.predict(bank.normalize(item.obs), deterministic=True)
    reward, terms, q, T = evaluate(model, robot, item, a, k, w_peak, w_settle)
    return q, T, reward, terms


def _train_or_eval(args):
    """``train`` fits a policy over the bank; ``eval`` scores a saved one on paths.

    ``eval`` reports the policy against the two references that matter: the recorded
    path (reward 0 by construction) and a CEM search over the same action space, which
    is what any black-box optimizer could reach given a search per path. The policy's
    claim is not that it beats CEM but that it gets close in one forward pass instead
    of thousands of evaluations, so both the score and the wall clock are printed.
    """
    from stable_baselines3 import PPO
    from train_distillation_model import DistillModel

    torch.set_num_threads(4)
    robot, bank = Robot(args.robot), PathBank.load(args.bank)
    n_act = O.B_MAX + args.rl_points * N_JOINTS

    if args.command == "train":
        out = args.agent or f"models/ppo-{args.robot.lower()}.zip"
        print(f"training on {len(bank.items)} items, {n_act}-dim action "
              f"({args.rl_points} offset points), {args.steps} steps on "
              f"{args.n_envs} workers")
        t0 = time.perf_counter()
        train(args.bank, args.model, args.robot, args.steps, args.rl_points, args.k,
              args.w_peak, args.w_settle, args.n_envs, args.warm, out, args.run,
              bc_items=args.bc, bc_procs=args.n_envs,
              bc_cache=args.bc_cache or f"models/bc-{args.robot.lower()}.npz")
        print(f"wrote {out} in {(time.perf_counter() - t0) / 60:.1f} min")
        return

    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    agent = PPO.load(args.agent, device="cpu")
    np_path = norm_path(args.agent)
    if os.path.exists(np_path):
        bank.adopt(dict(np.load(np_path)))
        print(f"  using the normalizer from {np_path}")
    else:
        print(f"  WARNING: {np_path} is missing, so this bank's own statistics are "
              f"used. If this is not the bank the policy trained on, the inputs are "
              f"wrong and the result is meaningless.")
    whole = probe_set([it for it in bank.items if "[" not in it.name], args.paths)
    print(f"ALPHA={O.ALPHA} LIMIT={O.LIMIT} K={O.K}   recorded path scores 0\n")
    print(f"{'path':16s} {'pause%':>6s} | {'PPO':>7s} {'cut%':>6s} {'err x':>6s} "
          f"{'pen':>7s} {'s':>6s} | {'CEM':>7s} {'s':>5s} | {'PPO/CEM':>7s}")
    rows = []
    for it in whole:
        t0 = time.perf_counter()
        q, T, reward, terms = agent_solve(agent, model, robot, bank, it, args.k,
                                          args.w_peak, args.w_settle)
        ps = time.perf_counter() - t0
        t0 = time.perf_counter()
        best, _ = cem(model, robot, it, n_points=args.rl_points, k=args.k,
                      w_peak=args.w_peak, w_settle=args.w_settle)
        cs = time.perf_counter() - t0
        pause = float((O.movable(it.q).numpy() < 0.5).mean()) * 100
        rows.append((reward, best, ps, cs))
        print(f"{it.name:16s} {pause:5.1f}% | {reward:7.3f} {(1-T/it.T0)*100:5.1f}% "
              f"{terms['err']/it.baseline(args.k):6.3f} {terms['pen']:7.5f} {ps:6.3f} "
              f"| {best:7.3f} {cs:5.1f} | {reward/best if best > 1e-9 else float('nan'):7.1%}")
    r = np.array(rows)
    print(f"\n  PPO median {np.median(r[:,0]):.3f} in {np.median(r[:,2]):.3f} s   "
          f"CEM median {np.median(r[:,1]):.3f} in {np.median(r[:,3]):.1f} s")
    print(f"  policy reaches {np.median(r[:,0]/np.maximum(r[:,1],1e-9)):.0%} of the "
          f"search's result, {np.median(r[:,3])/max(np.median(r[:,2]),1e-9):.0f}x faster")
    print(f"  beats the recorded path on {(r[:,0] > 0).sum()}/{len(r)} paths")


def _apply(args):
    """Write runnable ``.path`` files: the recorded motion and the optimized one.

    Both go out through ``convert.write_path``, the same writer ``send.py`` reads, so
    the pair differs only in the trajectory -- which is what makes the hardware
    comparison a comparison. The predicted improvement is printed alongside, because
    that is the number the robot is being asked to confirm or refute.
    """
    from convert import write_path
    from stable_baselines3 import PPO
    from train_distillation_model import DistillModel

    torch.set_num_threads(4)
    robot, bank = Robot(args.robot), PathBank.load(args.bank)
    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    agent = PPO.load(args.agent, device="cpu")
    npz = norm_path(args.agent)
    if os.path.exists(npz):
        bank.adopt(dict(np.load(npz)))
    else:
        raise SystemExit(f"{npz} missing: the policy's normalizer must travel with it")

    by = {it.name: it for it in bank.items}
    names = args.paths_named or [it.name for it in
                                 probe_set([i for i in bank.items if "[" not in i.name],
                                           args.paths)]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{'path':18s} {'T0':>7s} {'T':>7s} {'cut%':>6s} {'err x':>6s} {'pen':>7s} "
          f"{'reward':>7s}")
    for name in names:
        if name not in by:
            print(f"  {name}: not in the bank, skipped")
            continue
        it = by[name]
        q, T, reward, terms = agent_solve(agent, model, robot, bank, it, args.k,
                                          args.w_peak, args.w_settle)
        base = os.path.join(args.out_dir, name)
        write_path(f"{base}.baseline.path", it.q, DT)
        write_path(f"{base}.ppo.path", q, DT)
        flag = "  ** INFEASIBLE, do not run" if terms["pen"] > 1e-9 else ""
        print(f"{name:18s} {it.T0:7.3f} {T:7.3f} {(1-T/it.T0)*100:5.1f}% "
              f"{terms['err']/it.baseline(args.k):6.3f} {terms['pen']:7.5f} "
              f"{reward:+7.3f}{flag}")
    print(f"\nwrote {2 * len(names)} files to {args.out_dir}/. Run a pair on the robot:")
    n0 = names[0]
    print(f"  python send.py --robot-ip <ip> --path {args.out_dir}/{n0}.baseline.path "
          f"--loop 5 --out baseline.csv")
    print(f"  python send.py --robot-ip <ip> --path {args.out_dir}/{n0}.ppo.path "
          f"--loop 5 --out ppo.csv")
    print(f"  python analysis.py --csv baseline.csv ppo.csv --model {args.model}")
    print("\nThe model predicts the cut above at the cost in `err x`. The recording is "
          "what says whether it held.")


def probe_set(items, n: int):
    """``n`` paths spread over the distinct trajectories, not the first ``n`` by name.

    The recordings are named ``T01_fast_r1``..., so taking a prefix of the sorted
    list samples one trajectory six times and says nothing about the rest -- and how
    much a path can gain here is mostly a property of the trajectory (how much of its
    cycle is spent standing still), which is exactly the axis that slice collapses.
    """
    groups = {}
    for it in items:
        groups.setdefault(it.name.split("_")[0], []).append(it)
    out, order = [], sorted(groups)
    while len(out) < n and any(groups[g] for g in order):
        for g in order:                      # one per trajectory, then go round again
            if groups[g] and len(out) < n:
                out.append(groups[g].pop(0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["preflight", "train", "eval", "apply"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--robot", required=True, choices=list(Robot.MODELS))
    ap.add_argument("--k", type=float, default=None,
                    help="sd of model uncertainty added to the error (default optimize.K)")
    ap.add_argument("--w-peak", type=float, default=W_PEAK)
    ap.add_argument("--w-settle", type=float, default=W_SETTLE)
    ap.add_argument("--paths", type=int, default=8, help="how many whole paths to probe")
    ap.add_argument("--rl-points", type=int, default=O.RL_POINTS,
                    help="offset control points; 0 is retiming only (12-dim action)")
    ap.add_argument("--steps", type=int, default=200_000, help="PPO timesteps")
    ap.add_argument("--n-envs", type=int, default=6, help="SubprocVecEnv workers")
    ap.add_argument("--warm", default=None, help="policy to warm-start from")
    ap.add_argument("--agent", default=None, help="policy to write (train) or read (eval)")
    ap.add_argument("--run", default=None, help="tensorboard log dir")
    ap.add_argument("--out-dir", default="optimized",
                    help="apply: where to write the .path files")
    ap.add_argument("--paths-named", nargs="+", default=None,
                    help="apply: specific bank items, e.g. T09_medium_r1")
    ap.add_argument("--bc", type=int, default=0,
                    help="clone this many searched solutions before PPO (0 = off)")
    ap.add_argument("--bc-cache", default=None,
                    help="reuse searched demonstrations from here (default "
                         "models/bc-<robot>.npz); recomputed if the bank, model, "
                         "action space or objective changed")
    args = ap.parse_args()

    if args.command == "apply":
        return _apply(args)
    if args.command in ("train", "eval"):
        return _train_or_eval(args)

    from train_distillation_model import DistillModel
    torch.set_num_threads(4)
    model = DistillModel.load(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    robot, bank = Robot(args.robot), PathBank.load(args.bank)
    whole = probe_set([it for it in bank.items if "[" not in it.name], args.paths)
    base = baseline_loss(args.w_peak, args.w_settle)
    print(f"baseline loss {base:.3f} for every path, by construction "
          f"(reward 0); {len(bank.items)} items, probing {len(whole)} whole paths\n")

    print(f"{'path':22s} {'blk':>3s} {'T0':>6s} {'pause%':>7s} {'ceiling':>8s} "
          f"{'a=0 reward':>11s} {'best 1-D':>9s} {'CEM':>7s} {'CEM s':>6s}")
    rows = []
    for it in whole:
        still = O.movable(it.q).numpy() < 0.5
        r0 = evaluate(model, robot, it, np.zeros(O.B_MAX + O.RL_POINTS * N_JOINTS,
                                                 np.float32),
                      args.k, args.w_peak, args.w_settle)[0]
        sw = sweep(model, robot, it, args.k, args.w_peak, args.w_settle)
        t0 = time.perf_counter()
        best, _ = cem(model, robot, it, k=args.k, w_peak=args.w_peak,
                      w_settle=args.w_settle)
        secs = time.perf_counter() - t0
        rows.append((it, r0, sw.max(), best))
        print(f"{it.name:22s} {it.n_blocks:3d} {it.T0:6.2f} {still.mean()*100:6.1f}% "
              f"{base - ceiling(it):8.3f} {r0:11.2e} {sw.max():9.3f} {best:7.3f} {secs:6.1f}")

    print(f"\n  a=0 reward is 0 to {max(abs(r) for _, r, _, _ in rows):.1e} "
          f"-- the neutral action is the recorded path")
    print(f"  CEM beats the recorded path on {sum(b > 0 for *_, b in rows)}/{len(rows)} paths, "
          f"median gain {np.median([b for *_, b in rows]):.3f}")


if __name__ == "__main__":
    main()
