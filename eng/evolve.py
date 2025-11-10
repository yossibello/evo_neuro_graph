# eng/evolve.py
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
import multiprocessing as mp

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
            action = policy.act(obs, explore=False)  # <- greedy
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

def _load_policy_npz(path, policy_kind):
    import numpy as np
    d = np.load(path)
    if policy_kind == "mlp":
        from eng.policies_mlp import MLPPolicy
        params = [d["W1"], d["b1"], d["W2"], d["b2"], d["W3"], d["b3"]]
        return MLPPolicy(params)
    else:
        from eng.policies import LinearPolicy
        return LinearPolicy(d["W"], d["b"])


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
    child = policy.clone()
    # Linear policy has W, b; MLP policy has params list
    if hasattr(child, "W"):
        child.W += rng.normal(0.0, sigma, size=child.W.shape).astype(child.W.dtype)
        child.b += rng.normal(0.0, sigma, size=child.b.shape).astype(child.b.dtype)
    else:
        # Assume .params contains arrays
        new_params = []
        for p in child.params:
            new_params.append(p + rng.normal(0.0, sigma, size=p.shape).astype(p.dtype))
        child.params = new_params
    return child


def crossover(p1, p2, rate: float, rng: np.random.RandomState):
    c = p1.clone()
    if rng.rand() < rate:
        if hasattr(c, "W"):
            # Linear: per-weight uniform crossover
            maskW = rng.rand(*c.W.shape) < 0.5
            c.W[maskW] = p2.W[maskW]
            maskb = rng.rand(*c.b.shape) < 0.5
            c.b[maskb] = p2.b[maskb]
        else:
            # MLP (params list)
            new_params = []
            for a, b in zip(c.params, p2.params):
                mask = rng.rand(*a.shape) < 0.5
                new_params.append(np.where(mask, b, a))
            c.params = new_params
    return c


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
