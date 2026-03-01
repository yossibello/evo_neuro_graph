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

def select_action(policy, obs, rng: np.random.RandomState, eps: float = 0.05) -> int:
    logits = policy(obs)
    # occasional random action for exploration
    if rng.rand() < eps:
        return int(rng.randint(0, logits.shape[0]))
    return int(np.argmax(logits))

def mutate_inplace(policy, sigma: float):
    """
    Apply Gaussian noise to all weights and biases of a policy in-place.
    Works for LinearPolicy, MLPPolicy, and GraphPolicy.
    """
    from eng.policies_graph import GraphPolicy, NUM_OPS

    if isinstance(policy, GraphPolicy):
        # Structural: discrete rewire + op flip
        N = policy.node_params.shape[0]
        nr = policy.num_registers
        for i in range(N):
            if np.random.rand() < 0.10:
                policy.node_params[i, 0] = np.random.randint(0, nr)
            if np.random.rand() < 0.10:
                policy.node_params[i, 1] = np.random.randint(0, nr)
            if np.random.rand() < 0.10:
                policy.node_params[i, 2] = np.random.randint(0, nr)
            if np.random.rand() < 0.05:
                policy.node_params[i, 3] = np.random.randint(0, NUM_OPS)
        # Parametric: continuous Gaussian on weights/gate only
        for col in (4, 5, 6, 7):
            policy.node_params[:, col] += np.random.randn(N) * sigma
        # Learning rates: smaller perturbation to keep them stable
        if policy.node_params.shape[1] > 8:
            for col in (8, 9, 10):
                policy.node_params[:, col] += np.random.randn(N) * sigma * 0.3
        policy._rebuild_cache()

    elif hasattr(policy, "W"):  # linear
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
    pop_size: int = 256
    elites: int = 48
    episodes: int = 64
    max_steps: int = 200
    generations: int = 200

    # Evolution hyperparams
    mutation_sigma: float = 0.12
    sigma_decay: float = 0.995         # SLOW decay — keep mutations potent longer
    mutation_sigma_floor: float = 0.10  # higher floor — never go micro
    crossover_rate: float = 0.30        # more genetic mixing

    # Anti-stagnation
    stagnation_window: int = 15         # react faster to stagnation
    sigma_restart_mult: float = 2.0     # multiply sigma on restart (legacy, escalation replaces this)
    sigma_restart_cap: float = 2.0      # restart sigma capped at this * mutation_sigma (0.24)
    fresh_inject_frac: float = 0.08     # more fresh random injections each gen

    # Extinction events (biological mass extinction / creative destruction)
    extinction_every_n_stag: int = 1    # extinction on EVERY stagnation restart
    extinction_kill_frac: float = 0.35  # base kill fraction (escalates)

    # Behavioral diversity (fitness sharing)
    diversity_enabled: bool = False      # DISABLED: overhead + corrupts selection; organic diversity via extinction is enough
    diversity_num_seeds: int = 4         # fixed seeds for behavioral fingerprint
    diversity_sharing_sigma: float = 0.3 # sharing radius (lower = more selective, 0.3 = only penalize near-clones)

    # Tournament selection
    tournament_k: int = 4               # tournament size for parent selection

    # Elite re-evaluation (noise reduction)
    reeval_factor: int = 3              # 1 = disabled; >1 = re-evaluate top candidates with N*episodes

    # RNG
    seed: int = 0

    # Policy kind
    policy: str = "linear"              # "linear" or "mlp" (if available)

    # Fitness shaping
    success_bonus: float = 0.25         # bonus * success_rate added to avg reward

    # Parallelism
    processes: Optional[int] = None     # None -> use mp.cpu_count()

    init_policy: str | None = None

    # Graph policy config
    graph_nodes: int = 64
    graph_ticks: int = 5
    graph_registers: int = 128
    graph_memory: int = 16

    # ── Biological evolution parameters ──────────────────────────

    # Self-adaptive sigma: each individual evolves its own mutation rate
    self_adaptive_sigma: bool = True
    tau_sigma: float = 0.10             # log-normal adaptation rate (τ)
    sigma_cap: float = 0.30             # max per-individual sigma (was 0.50 — too destructive)

    # Novelty search: reward behavioral diversity, not just fitness
    novelty_enabled: bool = True
    novelty_weight: float = 0.3         # weight of novelty in selection fitness
    novelty_k: int = 10                 # k-nearest neighbors for novelty score
    novelty_archive_size: int = 500     # max archived behavioral fingerprints

    # Cambrian explosion: periodic radical diversity injection
    cambrian_period: int = 40           # every N gens, inject radical variants
    cambrian_inject_frac: float = 0.15  # fraction of pop replaced

    # Biological mutation operators (GraphPolicy only)
    p_node_duplication: float = 0.06    # gene duplication
    p_node_deletion: float = 0.04       # gene loss / pseudogenization
    p_dormant_reactivation: float = 0.10 # reactivate silent nodes
    p_regulatory_burst: float = 0.12    # epigenetic learning rate reset
    p_block_transpose: float = 0.05     # chromosomal rearrangement
    p_weight_homeostasis: float = 0.15  # synaptic scaling
    p_neuromodulation: float = 0.04     # broadcast neuron creation


# ----------------------------
# Helpers
# ----------------------------
def make_policy(kind: str, cfg: 'GAConfig | None' = None):
    if kind == "linear":
        return LinearPolicy(
            W=np.random.randn(5, 61).astype(np.float32) * 0.1,
            b=np.zeros(5, dtype=np.float32),
        )
    if kind == "mlp":
        if not HAS_MLP:
            raise ValueError("MLP requested but eng/policies_mlp.py not found.")
        from eng.policies_mlp import MLPPolicy
        return MLPPolicy()
    if kind == "graph":
        kw = {}
        if cfg:
            kw = dict(
                num_registers=cfg.graph_registers,
                num_nodes=cfg.graph_nodes,
                num_ticks=cfg.graph_ticks,
                num_memory=cfg.graph_memory,
            )
        return GraphPolicy(**kw)
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

        # Reset memory between episodes (not between steps!)
        if hasattr(policy, 'reset_memory'):
            policy.reset_memory()

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = select_action(policy, obs, rng)   # ✅ make sure this is using the helper
            obs, reward, done, _info = env.step(action)
            # Hebbian plasticity: reward-modulated lifetime learning
            if hasattr(policy, 'hebbian_update'):
                policy.hebbian_update(reward)
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

    # ----------------------
    # MUCH stronger shaping
    # ----------------------
    # Idea:
    # - strict_sr dominates once it appears (full solution)
    # - goal_rate is also a strong signal (even if key usage is wrong at first)
    # - door_rate, key_rate help bootstrap learning toward the right sequence
    #
    fitness = (
        0.1 * avg_reward   # small weight; don't let tiny step costs dominate
        + 0.5 * key_rate
        + 1.0 * door_rate
        + 3.0 * goal_rate
        + 8.0 * strict_sr
        + 5.0 * strict_sr ** 2   # quadratic bonus: steeper gradient at higher success
    )

    return fitness, avg_reward, strict_sr


def _behavioral_fingerprint(policy, env_ctor, env_kwargs, n_seeds=4, max_steps=50):
    """
    Compute a behavioral fingerprint: a short vector of actions the policy
    takes on fixed seeds. Used for diversity / fitness sharing.

    Like how in nature, two animals might have similar genes but different
    hunting strategies — we want to protect BEHAVIORAL diversity, not just
    genetic diversity.
    """
    actions = []
    for seed in range(n_seeds):
        env = env_ctor(**(env_kwargs or {}))
        obs = env.reset(seed=seed * 1000 + 42)
        if hasattr(policy, 'reset_memory'):
            policy.reset_memory()
        rng = np.random.RandomState(seed)
        for _ in range(min(max_steps, 10)):  # just first 10 steps
            logits = policy(obs)
            action = int(np.argmax(logits))
            actions.append(action)
            obs, _, done, _ = env.step(action)
            if done:
                actions.extend([0] * (10 - len(actions) % 10))  # pad
                break
    return np.array(actions, dtype=np.float32)


def _fitness_sharing(fitness_arr, fingerprints, sharing_sigma=0.1):
    """
    Fitness sharing: divide each individual's fitness by its niche count.
    Individuals with identical behavior share their fitness — this
    pressures the population to maintain diverse strategies.

    Like ecological niches: if 50 animals eat the same food, they compete
    harder than 5 animals each eating different foods.
    """
    n = len(fingerprints)
    if n == 0:
        return fitness_arr

    # Normalize fingerprints for distance computation
    fp_matrix = np.array(fingerprints, dtype=np.float32)  # (n, D)
    # Pairwise Hamming-like distance (fraction of different actions)
    shared_fitness = fitness_arr.copy()
    for i in range(n):
        # Count how many others behave similarly
        if fp_matrix.shape[0] < 2:
            break
        dists = np.mean(fp_matrix[i] != fp_matrix, axis=1)  # (n,)
        # Sharing function: sh(d) = 1 if d < sigma, else 0
        niche_count = np.sum(dists < sharing_sigma)
        niche_count = max(1.0, niche_count)
        shared_fitness[i] = fitness_arr[i] / niche_count

    return shared_fitness


def _compute_novelty(fingerprints, archive, k=10):
    """
    Compute novelty score for each individual based on behavioral distance
    to k-nearest neighbors in the combined population + archive.

    Higher novelty = more behaviorally different from anything seen before.
    Like ecological niche discovery — rewarding pioneers who explore
    uncharted behavioral territory.
    """
    n = len(fingerprints)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    fp_matrix = np.array(fingerprints, dtype=np.float32)

    # Combine current population with recent archive
    if archive:
        archive_matrix = np.array(archive[-200:], dtype=np.float32)
        all_fps = np.vstack([fp_matrix, archive_matrix])
    else:
        all_fps = fp_matrix

    novelty_scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        # Hamming-like distance (fraction of differing actions)
        dists = np.mean(fp_matrix[i] != all_fps, axis=1)
        dists[i] = float('inf')  # exclude self
        # k-nearest neighbor average distance
        k_actual = min(k, len(dists) - 1)
        if k_actual > 0:
            nearest = np.partition(dists, k_actual)[:k_actual]
            novelty_scores[i] = float(np.mean(nearest))

    return novelty_scores


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

def mutate(policy, sigma: float, rng: np.random.RandomState, stag_pressure: float = 0.0):
    """
    Clone the policy and add Gaussian noise to its parameters.
    stag_pressure in [0, 1]: when high, structural mutation rates increase.
    Supports:
      - GraphPolicy (node_params with index re-quantization)
      - LinearPolicy (W, b)
      - Generic MLP-style policies exposing `.params` list of arrays.
    """
    child = policy.clone()

    # ─── GraphPolicy (BIOLOGICAL EVOLUTION) ───────────────────────
    if isinstance(child, GraphPolicy):
        from eng.policies_graph import NUM_OPS
        node_params = child.node_params.copy()
        N = node_params.shape[0]
        nr = child.num_registers

        # ── 1. Structural point mutations (adaptive to stagnation) ──
        # Like SNPs in regulatory DNA regions
        sp = min(1.0, max(0.0, stag_pressure))
        p_rewire  = 0.12 + 0.18 * sp    # 0.12→0.30 (higher base!)
        p_op_flip = 0.06 + 0.14 * sp    # 0.06→0.20

        for i in range(N):
            if rng.rand() < p_rewire:
                node_params[i, 0] = rng.randint(0, nr)
            if rng.rand() < p_rewire:
                node_params[i, 1] = rng.randint(0, nr)
            if rng.rand() < p_rewire:
                node_params[i, 2] = rng.randint(0, nr)
            if rng.rand() < p_op_flip:
                node_params[i, 3] = rng.randint(0, NUM_OPS)

        # ── 2. Parametric mutations (Gaussian on continuous params) ──
        for col in (4, 5, 6, 7):   # w0, w1, bias, gate
            node_params[:, col] += rng.normal(0.0, sigma, size=N).astype(np.float32)
        # Learning rate mutations
        if node_params.shape[1] > 8:
            for col in (8, 9, 10):
                node_params[:, col] += rng.normal(0.0, sigma * 0.3, size=N).astype(np.float32)

        # ══════════════════════════════════════════════════════════════
        # BIOLOGICAL MUTATION OPERATORS
        # ══════════════════════════════════════════════════════════════

        # ── 3. Gene duplication ──
        # THE major source of novelty in real biology (opsins → color
        # vision, Hox genes → body plan diversity). Copy a strong node
        # to a new slot with slight divergence (neofunctionalization).
        if rng.rand() < 0.06:
            gate_vals = 1.0 / (1.0 + np.exp(-np.clip(node_params[:, 7], -10, 10)))
            strong = np.where(gate_vals > 0.5)[0]
            if len(strong) > 0:
                donor = rng.choice(strong)
                target = rng.randint(0, N)
                if target != donor:
                    node_params[target] = node_params[donor].copy()
                    node_params[target, 2] = rng.randint(0, nr)  # new destination
                    node_params[target, 4:8] += rng.normal(0, sigma * 0.5, size=4).astype(np.float32)

        # ── 4. Pseudogenization (gene silencing) ──
        # Sometimes less is more. Silence a noisy node.
        if rng.rand() < 0.04:
            victim = rng.randint(0, N)
            node_params[victim, 7] = rng.uniform(-3.0, -1.0)  # sigmoid → near 0

        # ── 5. Dormant node reactivation ──
        # Like transposon reactivation or de novo gene birth.
        # Wake up dead nodes with fresh random wiring.
        if rng.rand() < 0.10:
            gate_vals = 1.0 / (1.0 + np.exp(-np.clip(node_params[:, 7], -10, 10)))
            dormant = np.where(gate_vals < 0.15)[0]
            if len(dormant) > 0:
                n_wake = min(rng.randint(1, 5), len(dormant))
                wake_idxs = rng.choice(dormant, size=n_wake, replace=False)
                for wi in wake_idxs:
                    node_params[wi, 7] = rng.uniform(1.0, 2.5)
                    node_params[wi, 0] = rng.randint(0, nr)
                    node_params[wi, 1] = rng.randint(0, nr)
                    node_params[wi, 2] = rng.randint(0, nr)
                    node_params[wi, 3] = rng.randint(0, NUM_OPS)
                    node_params[wi, 4] = rng.normal(0, 0.3)
                    node_params[wi, 5] = rng.normal(0, 0.3)
                    node_params[wi, 6] = rng.normal(0, 0.1)

        # ── 6. Epigenetic regulatory burst ──
        # Radically reset learning rates on a node cluster.
        # Same genome, completely different learning dynamics.
        if rng.rand() < 0.12:
            n_burst = max(1, N // 6)
            burst_idxs = rng.choice(N, size=n_burst, replace=False)
            for col in (8, 9, 10):
                node_params[burst_idxs, col] = rng.normal(0, 0.05, size=n_burst).astype(np.float32)

        # ── 7. Chromosomal rearrangement (block transposition) ──
        # Swap entire blocks of adjacent nodes, like inversions
        # or translocations in real chromosomes.
        if rng.rand() < 0.05:
            block_sz = rng.randint(2, max(3, N // 6))
            a = rng.randint(0, max(1, N - block_sz))
            b = rng.randint(0, max(1, N - block_sz))
            if a != b:
                tmp = node_params[a:a+block_sz].copy()
                node_params[a:a+block_sz] = node_params[b:b+block_sz]
                node_params[b:b+block_sz] = tmp

        # ── 8. Synaptic homeostasis ──
        # Like synaptic scaling in real neurons — prevents runaway
        # excitation by normalizing extreme weights back toward zero.
        if rng.rand() < 0.15:
            for col in (4, 5, 6):
                extreme = np.abs(node_params[:, col]) > 3.0
                if np.any(extreme):
                    node_params[extreme, col] *= 0.6

        # ── 9. Neuromodulatory broadcast wiring ──
        # Create hub neurons that project to many targets, like
        # dopamine/serotonin neurons in real brains.
        if rng.rand() < 0.04:
            hub = rng.randint(0, N)
            hub_dst = int(node_params[hub, 2]) % nr
            n_listeners = rng.randint(3, min(8, N))
            listeners = rng.choice(N, size=n_listeners, replace=False)
            for li in listeners:
                if rng.rand() < 0.5:
                    node_params[li, 0] = hub_dst
                else:
                    node_params[li, 1] = hub_dst

        child.node_params = node_params.astype(np.float32)
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
            return child

        N = A.shape[0]
        # Node-level crossover: swap entire nodes (preserves src/weight coupling)
        mask = rng.rand(N, 1) < 0.5    # (N, 1) broadcasts to (N, D)
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

    best_ever_f = -float('inf')
    best_ever_policy = None

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

        # Initialize best_ever from init_policy
        best_ever_f = f0
        best_ever_policy = base.clone()
        os.makedirs("artifacts", exist_ok=True)
        save_policy_npz(best_ever_policy, "artifacts/best_ever_policy.npz")
    else:
        pop = [make_policy(cfg.policy, cfg) for _ in range(cfg.pop_size)]

    # ---- Stagnation tracking ----
    gens_since_improvement = 0
    stagnation_restarts = 0  # count how many times we've restarted sigma
    hall_of_fame = []  # stores (fitness, policy) of all-time bests
    HOF_SIZE = 5

    # Seed hall of fame with init_policy if loaded
    if best_ever_policy is not None:
        hall_of_fame.append((best_ever_f, best_ever_policy.clone()))

    fitness = np.zeros(cfg.pop_size, dtype=np.float32)
    avg_rewards = np.zeros(cfg.pop_size, dtype=np.float32)
    success_rates = np.zeros(cfg.pop_size, dtype=np.float32)

    # ── Self-adaptive sigma: per-individual mutation strength ──
    pop_sigmas = np.full(cfg.pop_size, sigma, dtype=np.float32)
    tau_sigma = getattr(cfg, 'tau_sigma', 0.10)
    sigma_cap = getattr(cfg, 'sigma_cap', 0.50)

    # ── Novelty archive: behavioral fingerprints of past individuals ──
    novelty_archive: List[np.ndarray] = []
    novelty_max = getattr(cfg, 'novelty_archive_size', 500)

    history: List[Dict[str, float]] = []

    # Multiprocessing context (spawn works everywhere)
    ctx = mp.get_context("spawn")
    n_proc = cfg.processes or mp.cpu_count()

    # Create pool ONCE (if we can use env_ctor)
    pool = None
    if not have_local_factory:
        pool = ctx.Pool(processes=n_proc)

    def _tournament_select(pop_list, fit_arr, k, _rng):
        """Tournament selection: pick k random individuals, return the best."""
        idxs = _rng.randint(0, len(pop_list), size=k)
        best_i = idxs[np.argmax(fit_arr[idxs])]
        return pop_list[best_i]

    def _tournament_select_idx(fit_arr, k, _rng):
        """Tournament selection returning INDEX (for self-adaptive sigma)."""
        idxs = _rng.randint(0, len(fit_arr), size=k)
        return int(idxs[np.argmax(fit_arr[idxs])])

    try:
        for gen in range(1, cfg.generations + 1):
            # ---- Parallel evaluation ----
            # Use a shared seed batch so elites get re-evaluated on same seeds
            # (reduces fitness noise between generations)
            eval_seed_base = int(rng.randint(0, 2**31 - 1))
            jobs = []
            for i, pol in enumerate(pop):
                seed_i = (eval_seed_base + i * 7919) % (2**31)
                if have_local_factory:
                    f, r, s = evaluate_policy(pol, env_maker_local, cfg.episodes, cfg.max_steps, np.random.RandomState(seed_i))
                    fitness[i], avg_rewards[i], success_rates[i] = f, r, s
                else:
                    jobs.append((pol, cfg.episodes, cfg.max_steps, seed_i, env_ctor, env_kwargs or {}))

            if not have_local_factory:
                results = pool.map(_eval_worker, jobs)
                for i, (f, r, s) in enumerate(results):
                    fitness[i], avg_rewards[i], success_rates[i] = f, r, s

            # ---- Elite re-evaluation (noise reduction) ----
            # Re-evaluate top candidates with more episodes to get stable fitness
            if cfg.reeval_factor > 1:
                prelim_order = np.argsort(fitness)[::-1]
                n_reeval = min(3 * cfg.elites, cfg.pop_size)
                reeval_idxs = prelim_order[:n_reeval]
                reeval_eps = cfg.episodes * (cfg.reeval_factor - 1)  # additional episodes

                reeval_seed_base = int(rng.randint(0, 2**31 - 1))
                if have_local_factory:
                    for j, idx in enumerate(reeval_idxs):
                        seed_re = (reeval_seed_base + j * 6271) % (2**31)
                        f2, r2, s2 = evaluate_policy(
                            pop[idx], env_maker_local, reeval_eps, cfg.max_steps,
                            np.random.RandomState(seed_re),
                        )
                        w1, w2 = cfg.episodes, reeval_eps
                        wt = w1 + w2
                        fitness[idx] = (w1 * fitness[idx] + w2 * f2) / wt
                        avg_rewards[idx] = (w1 * avg_rewards[idx] + w2 * r2) / wt
                        success_rates[idx] = (w1 * success_rates[idx] + w2 * s2) / wt
                else:
                    reeval_jobs = []
                    for j, idx in enumerate(reeval_idxs):
                        seed_re = (reeval_seed_base + j * 6271) % (2**31)
                        reeval_jobs.append((pop[idx], reeval_eps, cfg.max_steps, seed_re, env_ctor, env_kwargs or {}))
                    reeval_results = pool.map(_eval_worker, reeval_jobs)
                    for j, idx in enumerate(reeval_idxs):
                        f2, r2, s2 = reeval_results[j]
                        w1, w2 = cfg.episodes, reeval_eps
                        wt = w1 + w2
                        fitness[idx] = (w1 * fitness[idx] + w2 * f2) / wt
                        avg_rewards[idx] = (w1 * avg_rewards[idx] + w2 * r2) / wt
                        success_rates[idx] = (w1 * success_rates[idx] + w2 * s2) / wt

            # ---- Selection & logging ----
            # Elites are ALWAYS chosen by RAW fitness (best performers survive)
            order = np.argsort(fitness)[::-1]
            elites = [pop[i] for i in order[:cfg.elites]]
            best_idx = int(order[0])
            gen_best_f = float(fitness[best_idx])

            # ---- Behavioral diversity (for parent selection only) ----
            # Diversity pressure only affects tournament selection for offspring,
            # NOT elite selection. This prevents "diverse but bad" from replacing
            # actual top performers, while still encouraging the breeding pool
            # to explore different strategies.
            if cfg.diversity_enabled and env_ctor is not None:
                try:
                    fingerprints = []
                    for pol in pop:
                        fp = _behavioral_fingerprint(
                            pol, env_ctor, env_kwargs,
                            n_seeds=cfg.diversity_num_seeds,
                        )
                        fingerprints.append(fp)
                    selection_fitness = _fitness_sharing(
                        fitness, fingerprints, cfg.diversity_sharing_sigma
                    )
                except Exception:
                    selection_fitness = fitness
            else:
                selection_fitness = fitness

            # ---- Novelty search: reward behavioral pioneers ----
            _novelty_enabled = getattr(cfg, 'novelty_enabled', True)
            if _novelty_enabled and env_ctor is not None:
                try:
                    gen_fingerprints = []
                    for pol in pop:
                        fp = _behavioral_fingerprint(
                            pol, env_ctor, env_kwargs,
                            n_seeds=cfg.diversity_num_seeds,
                        )
                        gen_fingerprints.append(fp)

                    # Compute novelty: avg distance to k-nearest in pop + archive
                    novelty_scores = _compute_novelty(
                        gen_fingerprints, novelty_archive,
                        k=getattr(cfg, 'novelty_k', 10),
                    )

                    # Add fingerprints to archive (reservoir sampling)
                    for fp in gen_fingerprints:
                        if len(novelty_archive) < novelty_max:
                            novelty_archive.append(fp)
                        elif rng.rand() < 0.15:
                            idx_replace = rng.randint(0, len(novelty_archive))
                            novelty_archive[idx_replace] = fp

                    # Blend novelty into selection fitness
                    nw = getattr(cfg, 'novelty_weight', 0.3)
                    if novelty_scores.shape[0] == selection_fitness.shape[0]:
                        ns_std = max(float(novelty_scores.std()), 1e-8)
                        novelty_norm = (novelty_scores - novelty_scores.mean()) / ns_std
                        f_std = max(float(selection_fitness.std()), 1e-8)
                        novelty_norm *= f_std  # match fitness scale
                        selection_fitness = selection_fitness + nw * novelty_norm
                except Exception:
                    pass  # novelty is optional, don't crash

            # ---- Hall of Fame tracking ----
            if gen_best_f > best_ever_f:
                best_ever_f = gen_best_f
                gens_since_improvement = 0
                # Add to hall of fame
                best_ever_policy = pop[best_idx].clone()
                hall_of_fame.append((gen_best_f, best_ever_policy))
                hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                hall_of_fame = hall_of_fame[:HOF_SIZE]
                # Persist best_ever to disk immediately (never lose it)
                os.makedirs("artifacts", exist_ok=True)
                save_policy_npz(best_ever_policy, "artifacts/best_ever_policy.npz")
            else:
                gens_since_improvement += 1

            # ---- Stagnation detection & sigma restart + EXTINCTION ----
            stag_tag = ""
            stag_pressure = min(1.0, gens_since_improvement / max(1, cfg.stagnation_window))
            if gens_since_improvement >= cfg.stagnation_window:
                stagnation_restarts += 1
                old_sigma = sigma
                # Escalating restart: each restart is progressively more aggressive.
                # Restart #1: 1.5x initial (0.18), #2: +0.04 (0.22), #3: +0.04 (0.26)...
                # Like escalating environmental pressure — if a mild shock fails,
                # hit the population harder next time.
                base_restart = cfg.mutation_sigma * 1.5
                escalation  = 0.02 * stagnation_restarts  # gentler escalation (was 0.04)
                sigma_cap   = cfg.mutation_sigma * cfg.sigma_restart_cap
                sigma = min(base_restart + escalation, sigma_cap)
                gens_since_improvement = 0

                # Post-restart cooldown: decay faster for first few gens
                # so sigma comes back to productive range quickly
                # (applied after the restart, before next gen's decay)
                stag_tag = f" | STAG_RESTART #{stagnation_restarts} sigma {old_sigma:.4f}->{sigma:.4f}"

                # ---- EXTINCTION EVENT ----
                # Every Nth stagnation restart, trigger a mass extinction.
                # Like the asteroid that killed the dinosaurs — wipe out the
                # bottom portion of the population and fill with radically
                # different alternatives. The strong survive, the stuck die.
                if stagnation_restarts % cfg.extinction_every_n_stag == 0:
                    # Escalating extinction: each event kills more of the population.
                    # If the first mass extinction didn't help, the next one is harsher.
                    extinction_num = stagnation_restarts // cfg.extinction_every_n_stag
                    extra_kill = min(0.05 * (extinction_num - 1), 0.15)  # gentler escalation
                    eff_kill_frac = min(cfg.extinction_kill_frac + extra_kill, 0.55)
                    n_kill = int(cfg.pop_size * eff_kill_frac)
                    # Kill the weakest individuals
                    survivors = [pop[i] for i in order[:cfg.pop_size - n_kill]]

                    # Replace with a mix of:
                    # 1) Heavily mutated hall-of-fame (proven DNA, new expression)
                    # 2) Heavily mutated elites (local diversity explosion)
                    # 3) Completely fresh randoms (alien immigrants)
                    replacements: List[Any] = []
                    n_hof_mutants = n_kill // 3
                    n_elite_mutants = n_kill // 3
                    n_randoms = n_kill - n_hof_mutants - n_elite_mutants

                    # Heavily mutated HOF clones (3x sigma — radically different)
                    for i in range(n_hof_mutants):
                        if hall_of_fame:
                            src = hall_of_fame[i % len(hall_of_fame)][1]
                        else:
                            src = elites[0]
                        child = src.clone()
                        child = mutate(child, sigma * 3.0, rng, stag_pressure=1.0)
                        replacements.append(child)

                    # Heavily mutated elite clones (2x sigma)
                    for i in range(n_elite_mutants):
                        src = elites[i % len(elites)]
                        child = src.clone()
                        child = mutate(child, sigma * 2.0, rng, stag_pressure=0.8)
                        replacements.append(child)

                    # Fresh random individuals (completely new genetic material)
                    for _ in range(n_randoms):
                        replacements.append(make_policy(cfg.policy, cfg))

                    pop = survivors + replacements
                    # Update pop_sigmas: survivors keep theirs, replacements get high sigma
                    survivor_sigmas = [float(pop_sigmas[i]) for i in order[:cfg.pop_size - n_kill]]
                    replacement_sigmas = [cfg.mutation_sigma * 2.5] * len(replacements)
                    pop_sigmas = np.array(survivor_sigmas + replacement_sigmas, dtype=np.float32)
                    stag_tag += f" | EXTINCTION#{extinction_num} killed {n_kill}/{cfg.pop_size} ({eff_kill_frac:.0%}) (hof_mut={n_hof_mutants}, elite_mut={n_elite_mutants}, fresh={n_randoms})"

            # ---- Cambrian Explosion (periodic radical diversity burst) ----
            # Every N gens, inject radically different topologies regardless of
            # stagnation. Like the real Cambrian explosion: a sudden burst of
            # new body plans. Most will die, but some may find new niches.
            _cambrian_period = getattr(cfg, 'cambrian_period', 40)
            _cambrian_frac = getattr(cfg, 'cambrian_inject_frac', 0.15)
            if _cambrian_period > 0 and gen % _cambrian_period == 0 and gen > 1:
                n_cambrian = int(cfg.pop_size * _cambrian_frac)
                for ci in range(min(n_cambrian, len(order))):
                    worst_idx = order[-(ci + 1)]
                    pop[worst_idx] = make_policy(cfg.policy, cfg)
                    pop_sigmas[worst_idx] = cfg.mutation_sigma * 2.5
                stag_tag += f" | CAMBRIAN: {n_cambrian} radical variants"

            stats = {
                "gen": gen,
                "best_f": gen_best_f,
                "best_r": float(avg_rewards[best_idx]),
                "best_s": float(success_rates[best_idx]),
                "mean_f": float(fitness.mean()),
                "mean_r": float(avg_rewards.mean()),
                "mean_s": float(success_rates.mean()),
                "sigma": float(sigma),
                "mean_pop_sigma": float(pop_sigmas.mean()),
                "best_ever_f": best_ever_f,
                "stag_gens": gens_since_improvement,
            }
            history.append(stats)

            # Top-quartile mean (more informative than full-pop mean)
            q25 = max(1, cfg.pop_size // 4)
            top_q_f = float(fitness[order[:q25]].mean())

            print(
                f"Gen {gen:03d} | "
                f"best_f {stats['best_f']:+.3f} (r {stats['best_r']:+.3f}, s {stats['best_s']:.2f}) | "
                f"mean_f {stats['mean_f']:+.3f} top25%_f {top_q_f:+.3f} | "
                f"best_ever {best_ever_f:+.3f} stag {gens_since_improvement} | "
                f"sigma {stats['sigma']:.4f} | procs {n_proc}"
                f"{stag_tag}"
            )

            # ---- Checkpoints every N generations ----
            if gen % 20 == 0:
                os.makedirs("artifacts/checkpoints", exist_ok=True)
                K = 3  # save top-3
                for rank in range(min(K, len(order))):
                    idx = order[rank]
                    p = pop[idx]
                    bf = float(fitness[idx])
                    path = f"artifacts/checkpoints/gen{gen:04d}_rank{rank+1}_f{bf:+.3f}.npz"
                    save_policy_npz(p, path)


            # ---- Reproduction (biological, with self-adaptive sigma) ----
            new_pop: List[Any] = []
            new_sigmas: List[float] = []

            # Elites survive unmutated — their sigma persists
            for rank_i in range(cfg.elites):
                elite_orig_idx = order[rank_i]
                new_pop.append(elites[rank_i].clone())
                new_sigmas.append(float(pop_sigmas[elite_orig_idx]))

            # Soft-mutated elite offspring: fine-tuning best solutions
            n_soft = min(cfg.elites, cfg.pop_size - len(new_pop))
            for i in range(n_soft):
                soft_child = elites[i % len(elites)].clone()
                soft_sigma = sigma * 0.3
                soft_child = mutate(soft_child, soft_sigma, rng, stag_pressure=0.0)
                new_pop.append(soft_child)
                new_sigmas.append(soft_sigma)

            # Inject hall-of-fame members (persistent memory of best-ever)
            for _, hof_pol in hall_of_fame:
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(hof_pol.clone())
                    new_sigmas.append(sigma)

            # Inject fresh random individuals (alien immigrants — high sigma)
            n_fresh = int(cfg.pop_size * cfg.fresh_inject_frac)
            for _ in range(n_fresh):
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(make_policy(cfg.policy, cfg))
                    new_sigmas.append(cfg.mutation_sigma * 1.5)

            # Fill rest via tournament + crossover + mutation
            # Self-adaptive sigma: children inherit parents' mutation rates
            while len(new_pop) < cfg.pop_size:
                i1 = _tournament_select_idx(selection_fitness, cfg.tournament_k, rng)
                i2 = _tournament_select_idx(selection_fitness, cfg.tournament_k, rng)
                child = crossover(pop[i1], pop[i2], cfg.crossover_rate, rng)
                # Self-adaptive: inherit avg of parents' sigma, mutate via log-normal
                if getattr(cfg, 'self_adaptive_sigma', True):
                    child_sigma = 0.5 * (pop_sigmas[i1] + pop_sigmas[i2])
                    child_sigma *= float(np.exp(rng.normal(0, tau_sigma)))
                    child_sigma = float(np.clip(child_sigma, cfg.mutation_sigma_floor, sigma_cap))
                else:
                    child_sigma = sigma
                child = mutate(child, child_sigma, rng, stag_pressure=stag_pressure)
                new_pop.append(child)
                new_sigmas.append(child_sigma)

            pop = new_pop
            pop_sigmas = np.array(new_sigmas[:cfg.pop_size], dtype=np.float32)
            # Decay global sigma (baseline for soft-mutation and non-adaptive paths)
            sigma *= cfg.sigma_decay
            sigma = max(sigma, cfg.mutation_sigma_floor)

    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Return best_ever from Hall of Fame (most reliable champion)
    if hall_of_fame:
        winner = hall_of_fame[0][1]  # HOF is sorted by fitness, [0] = best
    else:
        winner = pop[0]  # fallback: first elite
    return winner, history
