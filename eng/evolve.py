# eng/evolve.py
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np
import multiprocessing as mp
import os, numpy as np
from eng.policies_graph import GraphPolicy  # NEW


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

def select_action(policy, obs):
    """
    Greedy action selection for any policy that returns logits.
    Works for LinearPolicy, MLPPolicy, GraphPolicy, etc.
    """
    logits = policy(obs)              # policy must be callable
    return int(np.argmax(logits))

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
    if kind == "graph":  # NEW
        return GraphPolicy()
    raise ValueError(f"Unknown policy kind: {kind}")


import numpy as np

def evaluate_policy(policy,
                    env_maker,
                    episodes: int,
                    max_steps: int,
                    rng: np.random.RandomState):
    """
    Returns (fitness, avg_reward, strict_success_rate).
    strict_success = used_key == True AND agent == goal_pos at end of episode.
    """
    total_reward = 0.0

    strict_successes = 0      # key+door+goal in correct order
    goal_hits = 0             # reached goal regardless of key usage
    key_episodes = 0          # picked up key at some point (or used it)
    door_open_eps = 0         # used_key == True

    for _ in range(episodes):
        env = env_maker()
        obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = select_action(policy,obs)
            obs, reward, done, _info = env.step(action)
            ep_reward += reward
            steps += 1

        total_reward += ep_reward

        # End-of-episode stats
        agent_pos = getattr(env, "agent", None)
        goal_pos  = getattr(env, "goal_pos", None)
        has_key   = getattr(env, "has_key", False)
        used_key  = getattr(env, "used_key", False)

        at_goal = (agent_pos == goal_pos)
        if at_goal:
            goal_hits += 1
        if has_key or used_key:
            key_episodes += 1
        if used_key:
            door_open_eps += 1
        if at_goal and used_key:
            strict_successes += 1

    avg_reward   = total_reward / max(1, episodes)
    strict_sr    = strict_successes / max(1, episodes)
    goal_rate    = goal_hits        / max(1, episodes)
    key_rate     = key_episodes     / max(1, episodes)
    door_rate    = door_open_eps    / max(1, episodes)

    # ---- Fitness shaping ----
    # - avg_reward: general behavior
    # - +0.2 * key_rate: encourage picking up key
    # - +0.3 * door_rate: encourage opening door
    # - +0.5 * goal_rate: encourage hitting goal at all
    # - +1.0 * strict_sr: bonus for fully correct sequence
    fitness = (
        avg_reward
        + 0.2 * key_rate
        + 0.3 * door_rate
        + 0.5 * goal_rate
        + 1.0 * strict_sr
    )

    return fitness, avg_reward, strict_sr
from eng.io_policies import load_policy_npz as _io_load_policy_npz

def _load_policy_npz(path, policy_kind: str):
    # Ignore policy_kind and let the npz metadata decide
    return _io_load_policy_npz(path)

def save_policy_npz(p, path):
    import numpy as np
    from eng.policies_graph import GraphPolicy

    # If the policy has a custom as_dict():
    if hasattr(p, "as_dict"):
        np.savez_compressed(path, **p.as_dict())
        return

    # Linear policy
    if hasattr(p, "W") and hasattr(p, "b"):
        np.savez_compressed(path, W=p.W, b=p.b, kind="linear")
        return

    # GraphPolicy
    if isinstance(p, GraphPolicy):
        np.savez_compressed(
            path,
            graph_params=p.node_params,
            num_registers=np.array(p.num_registers, dtype=np.int32),
            kind="graph",
        )
        return

    # MLP-style list of params
    if hasattr(p, "layers"):  # list[(W,b), ...]
        params = {}
        for i, (W, b) in enumerate(p.layers):
            params[f"W{i}"] = W
            params[f"b{i}"] = b
        params["kind"] = "mlp"
        np.savez_compressed(path, **params)
        return

    if hasattr(p, "params"):  # list of arrays: [W1,b1,W2,b2,...]
        arrs = list(p.params)
        params = {}
        j = 0
        k = 0
        while j < len(arrs):
            W = arrs[j]
            b = arrs[j+1] if j+1 < len(arrs) else None
            params[f"W{k}"] = W
            if b is not None:
                params[f"b{k}"] = b
            j += 2; k += 1
        params["kind"] = "mlp"
        np.savez_compressed(path, **params)
        return

    raise ValueError(f"[save_policy_npz] Unknown policy type: {type(p)}")

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
from eng.policies_graph import GraphPolicy
import numpy as np

def mutate(policy, sigma: float, rng: np.random.RandomState):
    """
    Clone the policy and add Gaussian noise to its parameters.
    Supports:
      - GraphPolicy (node_params with index re-quantization)
      - LinearPolicy (W, b)
      - Generic MLP-style policies exposing `.params` list of arrays.
    """
    child = policy.clone()

    # ----- GraphPolicy -----
    if isinstance(child, GraphPolicy):
        # numeric mutation on node_params
        node_params = child.node_params.copy()
        node_params += rng.normal(0.0, sigma, size=node_params.shape).astype(node_params.dtype)

        # re-quantize and clamp index columns: src0, src1, dst
        max_idx = child.num_registers - 1
        min_idx = -child.num_registers

        for col in (0, 1, 2):  # src0, src1, dst
            ints = np.round(node_params[:, col]).astype(np.int32)
            ints = np.clip(ints, min_idx, max_idx)
            node_params[:, col] = ints

        # assign back and rebuild cache
        child.node_params = node_params
        child._rebuild_cache()
        return child

    # ----- LinearPolicy -----
    if hasattr(child, "W") and hasattr(child, "b"):
        child.W += rng.normal(0.0, sigma, size=child.W.shape).astype(child.W.dtype)
        child.b += rng.normal(0.0, sigma, size=child.b.shape).astype(child.b.dtype)
        return child

    # ----- Generic MLP / others with .params list -----
    if hasattr(child, "params"):
        new_params = []
        for p in child.params:
            p = np.asarray(p, dtype=np.float32)
            new_params.append(
                p + rng.normal(0.0, sigma, size=p.shape).astype(p.dtype)
            )
        child.params = new_params  # policy's setter should rebuild any caches
        return child

    raise ValueError(
        f"mutate: Unknown policy type {type(child)} "
        "(no node_params, no (W,b), no params)"
    )

import numpy as np
from eng.policies_graph import GraphPolicy


def crossover(p1, p2, rate: float, rng: np.random.RandomState):
    """
    Crossover between two policies.

    - GraphPolicy: elementwise uniform crossover on node_params.
    - LinearPolicy: per-weight uniform crossover on W, b.
    - Generic MLP-like with .params: elementwise uniform crossover.
    """
    # Start from a clone of parent1
    child = p1.clone()

    # If crossover is skipped by rate, just return the clone
    if rng.rand() >= rate:
        return child

    # --------------------------------------------------------
    # 1) GraphPolicy
    # --------------------------------------------------------
    if isinstance(child, GraphPolicy) and isinstance(p2, GraphPolicy):
        A = child.node_params      # (Na, D)
        B = p2.node_params         # (Nb, D)

        if A.shape != B.shape:
            # Easiest is to require same shape for now:
            # if shapes differ, just keep parent1's params
            return child

        mask = rng.rand(*A.shape) < 0.5
        child_params = np.where(mask, B, A).astype(A.dtype)

        child.node_params = child_params
        child._rebuild_cache()
        return child

    # --------------------------------------------------------
    # 2) LinearPolicy (W, b)
    # --------------------------------------------------------
    if hasattr(child, "W") and hasattr(child, "b") and \
       hasattr(p2, "W") and hasattr(p2, "b"):

        if child.W.shape != p2.W.shape or child.b.shape != p2.b.shape:
            # shape mismatch → do nothing
            return child

        maskW = rng.rand(*child.W.shape) < 0.5
        child.W = np.where(maskW, p2.W, child.W)

        maskb = rng.rand(*child.b.shape) < 0.5
        child.b = np.where(maskb, p2.b, child.b)
        return child

    # --------------------------------------------------------
    # 3) Generic params-based (MLP etc.)
    # --------------------------------------------------------
    if hasattr(child, "params") and hasattr(p2, "params"):
        new_params = []
        for a, b in zip(child.params, p2.params):
            a = np.asarray(a)
            b = np.asarray(b)
            if a.shape != b.shape:
                # mismatch -> keep a
                new_params.append(a)
            else:
                mask = rng.rand(*a.shape) < 0.5
                new_params.append(np.where(mask, b, a))

        child.params = new_params
        return child

    # --------------------------------------------------------
    # 4) Fallback: unknown policy type → no crossover
    # --------------------------------------------------------
    return child
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
      # ---------- population init ----------
    sigma = cfg.mutation_sigma
    if cfg.init_policy:
        base = _load_policy_npz(cfg.init_policy, cfg.policy)
        print(f"[DEBUG] init_policy={cfg.init_policy}, type={type(base)}")

        pop = [base.clone()]
        # fill rest with mutated copies around the champion
        for _ in range(cfg.pop_size - 1):
            child = base.clone()
            mutate_inplace(child, sigma=sigma)
            pop.append(child)

        # debug: measure base fitness once
        def _m():
            if env_ctor is not None:
                return env_ctor(**(env_kwargs or {}))
            else:
                return env_factory()

        f0, r0, s0 = evaluate_policy(
            base, _m, cfg.episodes, cfg.max_steps,
            np.random.RandomState(cfg.seed + 1234),
        )
        print(f"[DEBUG] init_policy fitness {f0:+.3f} (r {r0:+.3f}, s {s0:.2f})")
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

    # Create pool ONCE (if we can use env_ctor)
    pool = None
    if not have_local_factory:
        pool = ctx.Pool(processes=n_proc)

    try:
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

    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
