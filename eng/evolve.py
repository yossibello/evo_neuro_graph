# eng/evolve.py
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
import multiprocessing as mp
import os, numpy as np

# Policies
from eng.policies import LinearPolicy
try:
    # Optional: if you added a stronger MLP policy, this import will succeed.
    from eng.policies_mlp import MLPPolicy  # noqa: F401
    HAS_MLP = True
except Exception:
    HAS_MLP = False

import numpy as np
import copy

def mutate_inplace(policy, sigma: float):
    """
    Apply Gaussian noise to all weights and biases of a policy in-place.
    Works for both LinearPolicy and MLPPolicy.
    """
    if hasattr(policy, "W"):  # linear
        policy.W += np.random.randn(*policy.W.shape) * sigma
        policy.b += np.random.randn(*policy.b.shape) * sigma

    elif hasattr(policy, "layers"):  # MLP style (list of (W,b) pairs)
        for (W, b) in policy.layers:
            W += np.random.randn(*W.shape) * sigma
            b += np.random.randn(*b.shape) * sigma

    elif hasattr(policy, "params"):  # generic field
        for p in policy.params:
            p += np.random.randn(*p.shape) * sigma

    else:
        raise ValueError("Unknown policy type for mutation")

    return policy

# ----------------------------
# Config
# ----------------------------
@dataclass
class GAConfig:
    pop_size: int = 128
    elites: int = 8
    episodes: int = 16
    max_steps: int = 200
    generations: int = 200

    # Evolution hyperparams
    mutation_sigma: float = 0.12
    sigma_decay: float = 0.98          # anneal mutation per generation
    mutation_sigma_floor: float = 0.06   # ← ADD THIS
    crossover_rate: float = 0.35

    # RNG
    seed: int = 0

    # Policy kind
    policy: str = "linear"              # "linear" or "mlp" (if available)

    # Fitness shaping
    success_bonus: float = 0.25         # bonus * success_rate added to avg reward

    # Parallelism
    processes: Optional[int] = None     # None -> use mp.cpu_count()

    init_policy: str | None = None



# ----------------------------
# Helpers
# ----------------------------
def make_policy(kind: str):
    if kind == "linear":
        return LinearPolicy()
    if kind == "mlp":
        if not HAS_MLP:
            raise ValueError("MLP requested but eng/policies_mlp.py not found.")
        from eng.policies_mlp import MLPPolicy
        return MLPPolicy()
    raise ValueError(f"Unknown policy kind: {kind}")


def evaluate_policy(policy,
                    env_maker: Callable[[], Any],
                    episodes: int,
                    max_steps: int,
                    rng: np.random.RandomState) -> Tuple[float, float, float]:
    """
    Greedy evaluation (no exploration) as requested.
    Returns (fitness, avg_reward, success_rate).
    """
    total = 0.0
    successes = 0
    for _ in range(episodes):
        env = env_maker()
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
        done = False
        steps = 0
        ep_reward = 0.0
        while not done and steps < max_steps:
            logits = policy(obs)          # works for LinearPolicy and MLPPolicy
            action = int(np.argmax(logits))
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1
        total += ep_reward
        if ep_reward >= 1.0 - 1e-6:   # reached G at least once in episode
            successes += 1

    avg_reward = total / max(1, episodes)
    success_rate = successes / max(1, episodes)
    fitness = avg_reward + success_rate * 0.6
    return fitness, avg_reward, success_rate

from eng.io_policies import load_policy_npz  # put this near the top of evolve.py


def _load_policy_npz(path, policy_kind):
    """
    Thin wrapper that delegates to the unified loader in eng.io_policies.
    Ignores policy_kind, since the .npz already encodes 'kind'.
    """
    return load_policy_npz(path)

def save_policy_npz(p, path):
    import numpy as np

    if hasattr(p, "as_dict"):
        np.savez_compressed(path, **p.as_dict())
        return

    # Linear variants
    if hasattr(p, "W") and hasattr(p, "b"):
        np.savez_compressed(path, W=p.W, b=p.b, kind="linear")
        return

    # MLP variants
    if hasattr(p, "layers"):  # list[(W,b), ...]
        params = {}
        for i, (W, b) in enumerate(p.layers):
            params[f"W{i}"] = W
            params[f"b{i}"] = b
        np.savez_compressed(path, **params, kind="mlp")
        return

    if hasattr(p, "params"):  # list of arrays: [W1,b1,W2,b2,W3,b3]
        arrs = list(p.params)
        params = {}
        # try to pair as (W,b)
        j = 0
        k = 0
        while j < len(arrs):
            W = arrs[j]; b = arrs[j+1] if j+1 < len(arrs) else None
            params[f"W{k}"] = W
            if b is not None:
                params[f"b{k}"] = b
            j += 2; k += 1
        np.savez_compressed(path, **params, kind="mlp")
        return

    # Dict-like (some mutate/crossover return dicts)
    if isinstance(p, dict):
        keys = set(p.keys())
        if {"W","b"}.issubset(keys):
            np.savez_compressed(path, W=p["W"], b=p["b"], kind="linear")
            return
        # accept W0/b0,... pattern
        if any(k.startswith("W0") for k in keys) or any(k.startswith("W") for k in keys):
            np.savez_compressed(path, **p, kind=p.get("kind","mlp"))
            return

    # Last resort: try common attribute names W1,b1,...
    cand = {}
    for i in range(6):  # up to 6 layers if needed
        Wi = getattr(p, f"W{i}", None)
        bi = getattr(p, f"b{i}", None)
        if Wi is not None:
            cand[f"W{i}"] = Wi
        if bi is not None:
            cand[f"b{i}"] = bi
    if cand:
        np.savez_compressed(path, **cand, kind="mlp")
        return

    # Couldn’t detect format
    import warnings
    warnings.warn(f"[save_policy_npz] Unknown policy type: {type(p)}; attributes={dir(p)[:20]}")


# -------- Parallel worker --------
def _eval_worker(args):
    """
    Worker entrypoint for multiprocessing.
    args:
      - policy: picklable policy object
      - episodes, max_steps
      - seed: int
      - env_ctor: top-level callable (e.g., tasks.tinygrid.TinyGrid class)
      - env_kwargs: dict of kwargs for env_ctor
    """
    policy, episodes, max_steps, seed, env_ctor, env_kwargs = args
    rng = np.random.RandomState(seed)

    def env_maker():
        return env_ctor(**(env_kwargs or {}))

    return evaluate_policy(policy, env_maker, episodes, max_steps, rng)


# ----------------------------
# Genetic operators
# ----------------------------
def mutate(policy, sigma: float, rng: np.random.RandomState):
    """
    Create a mutated clone of `policy`.
    Supports:
      - LinearPolicy with W, b
      - MLPPolicy with .layers = [(W, b), ...]
      - (optionally) older policies with .params list
    """
    child = policy.clone()

    # LinearPolicy: W, b
    if hasattr(child, "W") and hasattr(child, "b"):
        child.W += rng.normal(0.0, sigma, size=child.W.shape).astype(child.W.dtype)
        child.b += rng.normal(0.0, sigma, size=child.b.shape).astype(child.b.dtype)
        return child

    # MLPPolicy: layers = list of (W, b)
    if hasattr(child, "layers"):
        new_layers = []
        for (W, b) in child.layers:
            Wm = W + rng.normal(0.0, sigma, size=W.shape).astype(W.dtype)
            bm = b + rng.normal(0.0, sigma, size=b.shape).astype(b.dtype)
            new_layers.append((Wm, bm))
        child.layers = new_layers
        return child

    # Fallback: legacy .params list
    if hasattr(child, "params"):
        new_params = []
        for p in child.params:
            new_params.append(p + rng.normal(0.0, sigma, size=p.shape).astype(p.dtype))
        child.params = new_params
        return child

    raise ValueError(f"Unknown policy type for mutation: {type(policy)}")


def crossover(p1, p2, rate: float, rng: np.random.RandomState):
    """
    Uniform crossover between two parents.
    Supports:
      - LinearPolicy with W, b
      - MLPPolicy with .layers = [(W, b), ...]
      - (optionally) older policies with .params
    """
    c = p1.clone()

    if rng.rand() >= rate:
        return c  # no crossover, just a clone

    # LinearPolicy: W, b
    if hasattr(c, "W") and hasattr(c, "b") and hasattr(p2, "W") and hasattr(p2, "b"):
        maskW = rng.rand(*c.W.shape) < 0.5
        c.W[maskW] = p2.W[maskW]
        maskb = rng.rand(*c.b.shape) < 0.5
        c.b[maskb] = p2.b[maskb]
        return c

    # MLPPolicy: layers = list of (W, b)
    if hasattr(c, "layers") and hasattr(p2, "layers"):
        new_layers = []
        for (Wa, ba), (Wb, bb) in zip(c.layers, p2.layers):
            # crossover weights
            maskW = rng.rand(*Wa.shape) < 0.5
            Wc = np.where(maskW, Wb, Wa)
            # crossover biases
            maskb = rng.rand(*ba.shape) < 0.5
            bc = np.where(maskb, bb, ba)
            new_layers.append((Wc, bc))
        c.layers = new_layers
        return c

    # Fallback: legacy .params list
    if hasattr(c, "params") and hasattr(p2, "params"):
        new_params = []
        for a, b in zip(c.params, p2.params):
            mask = rng.rand(*a.shape) < 0.5
            new_params.append(np.where(mask, b, a))
        c.params = new_params
        return c

    raise ValueError(f"Unknown policy type for crossover: {type(p1)}, {type(p2)}")


# ----------------------------
# Main GA
# ----------------------------
def run_ga(
    # For best compatibility across OS/process start methods, **prefer** passing a
    # top-level constructor and kwargs (both picklable).
    # Example usage from your train script:
    #   from tasks.tinygrid import TinyGrid
    #   winner, hist = run_ga(env_ctor=TinyGrid, env_kwargs={"max_steps": 200}, cfg=...)
    env_ctor: Optional[Callable[..., Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,

    # Legacy path (kept for backward compatibility):
    env_factory: Optional[Callable[[], Any]] = None,

    cfg: GAConfig = GAConfig(),
):
    """
    Returns (winner_policy, history)
    History is a list of dicts with per-generation stats.
    """
    # Validate environment maker
    if env_ctor is None:
        if env_factory is None:
            raise ValueError("Provide either env_ctor/env_kwargs or env_factory.")
        # Try to infer ctor/kwargs from a factory. NOTE: If env_factory is a lambda on macOS/Windows,
        # it may not be picklable. In that case, switch your train script to pass env_ctor/kwargs.
        def env_maker_local():
            return env_factory()
        have_local_factory = True
    else:
        have_local_factory = False

    rng = np.random.RandomState(cfg.seed)
    if cfg.init_policy:
        base = _load_policy_npz(cfg.init_policy, cfg.policy)
        pop = [base.clone()]
        # fill rest with mutated copies around the champion
        for _ in range(cfg.pop_size - 1):
            child = base.clone()
            mutate_inplace(child, sigma=cfg.mutation_sigma)
            pop.append(child)
    else:
        pop = [make_policy(cfg.policy) for _ in range(cfg.pop_size)]

    fitness = np.zeros(cfg.pop_size, dtype=np.float32)
    avg_rewards = np.zeros(cfg.pop_size, dtype=np.float32)
    success_rates = np.zeros(cfg.pop_size, dtype=np.float32)

    sigma = cfg.mutation_sigma
    history: List[Dict[str, float]] = []

    # Multiprocessing context (spawn works everywhere)
    ctx = mp.get_context("spawn")
    n_proc = cfg.processes or mp.cpu_count()

    for gen in range(1, cfg.generations + 1):
        # ---- Parallel evaluation ----
        jobs = []
        for i, pol in enumerate(pop):
            seed_i = int(rng.randint(0, 2**31 - 1))
            if have_local_factory:
                # Fallback path: use single-process evaluation to avoid pickling lambda/env_factory
                # (We still batch this loop, so performance will be lower.)
                f, r, s = evaluate_policy(pol, env_maker_local, cfg.episodes, cfg.max_steps, np.random.RandomState(seed_i))
                fitness[i], avg_rewards[i], success_rates[i] = f, r, s
            else:
                jobs.append((pol, cfg.episodes, cfg.max_steps, seed_i, env_ctor, env_kwargs or {}))

        if not have_local_factory:
            with ctx.Pool(processes=n_proc) as pool:
                results = pool.map(_eval_worker, jobs)
            for i, (f, r, s) in enumerate(results):
                fitness[i], avg_rewards[i], success_rates[i] = f, r, s

        # ---- Selection & logging ----
        order = np.argsort(fitness)[::-1]
        elites = [pop[i] for i in order[:cfg.elites]]
        best_idx = int(order[0])
        stats = {
            "gen": gen,
            "best_f": float(fitness[best_idx]),
            "best_r": float(avg_rewards[best_idx]),
            "best_s": float(success_rates[best_idx]),
            "mean_f": float(fitness.mean()),
            "mean_r": float(avg_rewards.mean()),
            "mean_s": float(success_rates.mean()),
            "sigma": float(sigma),
        }
        history.append(stats)
        print(
            f"Gen {gen:03d} | "
            f"best_f {stats['best_f']:+.3f} (r {stats['best_r']:+.3f}, s {stats['best_s']:.2f}) | "
            f"mean_f {stats['mean_f']:+.3f} (r {stats['mean_r']:+.3f}, s {stats['mean_s']:.2f}) | "
            f"sigma {stats['sigma']:.4f} | procs {n_proc}"
        )

        # ---- Checkpoints every N generations ----
        # You can hardcode N or later add a cfg.checkpoint_interval
        if gen % 20 == 0:
            os.makedirs("artifacts/checkpoints", exist_ok=True)
            K = 3  # save top-3
            for rank in range(min(K, len(order))):
                idx = order[rank]
                p = pop[idx]
                bf = float(fitness[idx])
                path = f"artifacts/checkpoints/gen{gen:04d}_rank{rank+1}_f{bf:+.3f}.npz"
                save_policy_npz(p, path)


        # ---- Reproduction ----
        new_pop: List[Any] = elites.copy()
        while len(new_pop) < cfg.pop_size:
            p1, p2 = rng.choice(elites), rng.choice(pop)
            child = crossover(p1, p2, cfg.crossover_rate, rng)
            child = mutate(child, sigma, rng)
            new_pop.append(child)
        pop = new_pop
        sigma *= cfg.sigma_decay
        sigma = max(sigma, cfg.mutation_sigma_floor)  # NEW

   

    # Final pick of winner (deterministic greedy eval)
    # We evaluate single-process here for simplicity; you can parallelize similarly if desired.
    final_fit = np.zeros(cfg.pop_size, dtype=np.float32)
    for i, pol in enumerate(pop):
        if have_local_factory:
            f, _, _ = evaluate_policy(pol, env_maker_local, cfg.episodes * 2, cfg.max_steps, np.random.RandomState(cfg.seed + i + 123))
        else:
            # Make a local maker that uses env_ctor directly
            def _m():
                return env_ctor(**(env_kwargs or {}))
            f, _, _ = evaluate_policy(pol, _m, cfg.episodes * 2, cfg.max_steps, np.random.RandomState(cfg.seed + i + 123))
        final_fit[i] = f

    winner = pop[int(np.argmax(final_fit))]
    return winner, history
